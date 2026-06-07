"""
Tests for MOAR multi-step pipeline support.

Covers:
- update_pipeline: diff-based replacement map for directives across steps
- Node.op_to_step: operation-to-step mapping
- Equijoin dict format preservation in step operation lists
"""

import os
import tempfile
from copy import deepcopy

import pytest
import yaml

from docetl.moar.search_utils import update_pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_op(name, op_type="map"):
    return {"name": name, "type": op_type, "prompt": f"do {name}", "output": {"schema": {"result": "string"}}}


def _op_names_per_step(config):
    """Return {step_name: [op_name, ...]} for quick assertions."""
    result = {}
    for step in config.get("pipeline", {}).get("steps", []):
        ops = []
        for op in step.get("operations", []):
            ops.append(op if isinstance(op, str) else list(op.keys())[0])
        result[step["name"]] = ops
    return result


def _flat_op_names(config):
    return [op["name"] for op in config["operations"]]


# ---------------------------------------------------------------------------
# Fixtures: multi-step pipeline configs
# ---------------------------------------------------------------------------

@pytest.fixture
def two_step_config():
    """Pipeline: step1(extract, summarize) -> step2(classify, filter)."""
    return {
        "operations": [
            _make_op("extract"),
            _make_op("summarize"),
            _make_op("classify"),
            _make_op("filter"),
        ],
        "pipeline": {
            "steps": [
                {"name": "step1", "input": "docs", "operations": ["extract", "summarize"]},
                {"name": "step2", "input": "step1", "operations": ["classify", "filter"]},
            ],
            "output": {"type": "file", "path": "out.json"},
        },
    }


@pytest.fixture
def three_step_config():
    """Pipeline: prep(clean) -> analysis(analyze, score) -> output(format)."""
    return {
        "operations": [
            _make_op("clean"),
            _make_op("analyze"),
            _make_op("score"),
            _make_op("format"),
        ],
        "pipeline": {
            "steps": [
                {"name": "prep", "input": "raw", "operations": ["clean"]},
                {"name": "analysis", "input": "prep", "operations": ["analyze", "score"]},
                {"name": "output", "input": "analysis", "operations": ["format"]},
            ],
            "output": {"type": "file", "path": "out.json"},
        },
    }


@pytest.fixture
def equijoin_config():
    """Pipeline with an equijoin step: step1(map_a) -> join_step(eq_join) -> step3(map_c)."""
    return {
        "operations": [
            _make_op("map_a"),
            {"name": "eq_join", "type": "equijoin", "comparison_prompt": "compare"},
            _make_op("map_c"),
        ],
        "pipeline": {
            "steps": [
                {"name": "step1", "input": "data", "operations": ["map_a"]},
                {"name": "join_step", "input": "step1", "operations": [
                    {"eq_join": {"left": "step1", "right": "other"}}
                ]},
                {"name": "step3", "input": "join_step", "operations": ["map_c"]},
            ],
            "output": {"type": "file", "path": "out.json"},
        },
    }


# ===========================================================================
# Test: chaining directive (target op removed, replaced by 2 new ops)
# ===========================================================================

class TestChainingDirective:
    """Simulate chaining: one map op → two map ops."""

    def test_chaining_in_step1_preserves_step2(self, two_step_config):
        old_ops = two_step_config["operations"]
        # Chaining replaces "extract" with "extract_part1" + "extract_part2"
        new_ops = [
            _make_op("extract_part1"),
            _make_op("extract_part2"),
            _make_op("summarize"),
            _make_op("classify"),
            _make_op("filter"),
        ]
        result = update_pipeline(two_step_config, new_ops, ["extract"], old_ops_list=old_ops)
        step_ops = _op_names_per_step(result)

        assert step_ops["step1"] == ["extract_part1", "extract_part2", "summarize"]
        assert step_ops["step2"] == ["classify", "filter"]

    def test_chaining_in_step2_preserves_step1(self, two_step_config):
        old_ops = two_step_config["operations"]
        # Chaining replaces "classify" with "classify_v1" + "classify_v2"
        new_ops = [
            _make_op("extract"),
            _make_op("summarize"),
            _make_op("classify_v1"),
            _make_op("classify_v2"),
            _make_op("filter"),
        ]
        result = update_pipeline(two_step_config, new_ops, ["classify"], old_ops_list=old_ops)
        step_ops = _op_names_per_step(result)

        assert step_ops["step1"] == ["extract", "summarize"]
        assert step_ops["step2"] == ["classify_v1", "classify_v2", "filter"]

    def test_chaining_middle_step(self, three_step_config):
        old_ops = three_step_config["operations"]
        # Chain "analyze" into "analyze_a" + "analyze_b"
        new_ops = [
            _make_op("clean"),
            _make_op("analyze_a"),
            _make_op("analyze_b"),
            _make_op("score"),
            _make_op("format"),
        ]
        result = update_pipeline(three_step_config, new_ops, ["analyze"], old_ops_list=old_ops)
        step_ops = _op_names_per_step(result)

        assert step_ops["prep"] == ["clean"]
        assert step_ops["analysis"] == ["analyze_a", "analyze_b", "score"]
        assert step_ops["output"] == ["format"]


# ===========================================================================
# Test: gleaning directive (target kept, validator inserted after)
# ===========================================================================

class TestGleaningDirective:
    """Simulate gleaning: target op kept, validation op inserted after it."""

    def test_gleaning_in_step1(self, two_step_config):
        old_ops = two_step_config["operations"]
        # Gleaning on "extract": keeps "extract", adds "extract_gleaning_validator"
        new_ops = [
            _make_op("extract"),
            _make_op("extract_gleaning_validator"),
            _make_op("summarize"),
            _make_op("classify"),
            _make_op("filter"),
        ]
        result = update_pipeline(two_step_config, new_ops, ["extract"], old_ops_list=old_ops)
        step_ops = _op_names_per_step(result)

        assert step_ops["step1"] == ["extract", "extract_gleaning_validator", "summarize"]
        assert step_ops["step2"] == ["classify", "filter"]

    def test_gleaning_in_step2(self, two_step_config):
        old_ops = two_step_config["operations"]
        new_ops = [
            _make_op("extract"),
            _make_op("summarize"),
            _make_op("filter"),
            _make_op("filter_gleaning_validator"),
            _make_op("classify"),  # reordered to put filter before classify for test
        ]
        # Actually, let's keep original order and glean on "filter" (last in step2)
        new_ops = [
            _make_op("extract"),
            _make_op("summarize"),
            _make_op("classify"),
            _make_op("filter"),
            _make_op("filter_validator"),
        ]
        result = update_pipeline(two_step_config, new_ops, ["filter"], old_ops_list=old_ops)
        step_ops = _op_names_per_step(result)

        assert step_ops["step1"] == ["extract", "summarize"]
        assert step_ops["step2"] == ["classify", "filter", "filter_validator"]


# ===========================================================================
# Test: doc_chunking directive (adds gather before target)
# ===========================================================================

class TestDocChunkingDirective:
    """Simulate doc_chunking: split op + gather op inserted before target, target kept."""

    def test_chunking_inserts_before_target(self, two_step_config):
        old_ops = two_step_config["operations"]
        # doc_chunking on "summarize": inserts "summarize_split" before, keeps "summarize",
        # and adds "summarize_gather" after
        new_ops = [
            _make_op("extract"),
            _make_op("summarize_split"),
            _make_op("summarize"),
            _make_op("summarize_gather"),
            _make_op("classify"),
            _make_op("filter"),
        ]
        result = update_pipeline(two_step_config, new_ops, ["summarize"], old_ops_list=old_ops)
        step_ops = _op_names_per_step(result)

        assert step_ops["step1"] == ["extract", "summarize_split", "summarize", "summarize_gather"]
        assert step_ops["step2"] == ["classify", "filter"]


# ===========================================================================
# Test: equijoin preservation
# ===========================================================================

class TestEquijoinPreservation:
    """Equijoin dict format in step operations must survive update_pipeline."""

    def test_equijoin_untouched_by_other_step_change(self, equijoin_config):
        old_ops = equijoin_config["operations"]
        # Chain "map_a" into "map_a1" + "map_a2", equijoin should stay as dict
        new_ops = [
            _make_op("map_a1"),
            _make_op("map_a2"),
            {"name": "eq_join", "type": "equijoin", "comparison_prompt": "compare"},
            _make_op("map_c"),
        ]
        result = update_pipeline(equijoin_config, new_ops, ["map_a"], old_ops_list=old_ops)
        step_ops = _op_names_per_step(result)

        assert step_ops["step1"] == ["map_a1", "map_a2"]
        assert step_ops["step3"] == ["map_c"]
        # equijoin step preserved as dict
        join_ops = result["pipeline"]["steps"][1]["operations"]
        assert len(join_ops) == 1
        assert isinstance(join_ops[0], dict)
        assert "eq_join" in join_ops[0]

    def test_equijoin_step_target_replaced(self, equijoin_config):
        old_ops = equijoin_config["operations"]
        # Chain "map_c" in step3
        new_ops = [
            _make_op("map_a"),
            {"name": "eq_join", "type": "equijoin", "comparison_prompt": "compare"},
            _make_op("map_c1"),
            _make_op("map_c2"),
        ]
        result = update_pipeline(equijoin_config, new_ops, ["map_c"], old_ops_list=old_ops)
        step_ops = _op_names_per_step(result)

        assert step_ops["step1"] == ["map_a"]
        assert step_ops["join_step"] == ["eq_join"]  # preserved
        assert step_ops["step3"] == ["map_c1", "map_c2"]


# ===========================================================================
# Test: identity / no-op cases
# ===========================================================================

class TestEdgeCases:
    def test_none_new_ops_returns_unchanged(self, two_step_config):
        original = deepcopy(two_step_config)
        result = update_pipeline(two_step_config, None, ["extract"])
        assert result == original

    def test_no_change_preserves_everything(self, two_step_config):
        old_ops = two_step_config["operations"]
        same_ops = deepcopy(old_ops)
        result = update_pipeline(two_step_config, same_ops, ["extract"], old_ops_list=old_ops)
        step_ops = _op_names_per_step(result)

        assert step_ops["step1"] == ["extract", "summarize"]
        assert step_ops["step2"] == ["classify", "filter"]

    def test_old_ops_list_fallback(self, two_step_config):
        """When old_ops_list is None, falls back to orig_config['operations']."""
        new_ops = [
            _make_op("extract_v2"),
            _make_op("summarize"),
            _make_op("classify"),
            _make_op("filter"),
        ]
        result = update_pipeline(two_step_config, new_ops, ["extract"])
        step_ops = _op_names_per_step(result)

        assert step_ops["step1"] == ["extract_v2", "summarize"]
        assert step_ops["step2"] == ["classify", "filter"]

    def test_flat_operations_updated(self, two_step_config):
        """The flat operations list should be replaced with new_ops_list."""
        old_ops = two_step_config["operations"]
        new_ops = [
            _make_op("extract_a"),
            _make_op("extract_b"),
            _make_op("summarize"),
            _make_op("classify"),
            _make_op("filter"),
        ]
        result = update_pipeline(two_step_config, new_ops, ["extract"], old_ops_list=old_ops)
        assert _flat_op_names(result) == ["extract_a", "extract_b", "summarize", "classify", "filter"]


# ===========================================================================
# Test: successive directive applications (simulating MCTS depth > 1)
# ===========================================================================

class TestSuccessiveDirectives:
    """MCTS applies directives iteratively. Test chaining two directives."""

    def test_chain_then_glean(self, two_step_config):
        # Round 1: chain "extract" → "extract_a" + "extract_b"
        old_ops_1 = two_step_config["operations"]
        new_ops_1 = [
            _make_op("extract_a"),
            _make_op("extract_b"),
            _make_op("summarize"),
            _make_op("classify"),
            _make_op("filter"),
        ]
        config = update_pipeline(two_step_config, new_ops_1, ["extract"], old_ops_list=old_ops_1)

        # Round 2: glean "classify" → keeps "classify", adds "classify_val"
        old_ops_2 = config["operations"]
        new_ops_2 = [
            _make_op("extract_a"),
            _make_op("extract_b"),
            _make_op("summarize"),
            _make_op("classify"),
            _make_op("classify_val"),
            _make_op("filter"),
        ]
        config = update_pipeline(config, new_ops_2, ["classify"], old_ops_list=old_ops_2)

        step_ops = _op_names_per_step(config)
        assert step_ops["step1"] == ["extract_a", "extract_b", "summarize"]
        assert step_ops["step2"] == ["classify", "classify_val", "filter"]

    def test_chain_in_different_steps(self, three_step_config):
        # Round 1: chain "clean" in prep
        old_ops_1 = three_step_config["operations"]
        new_ops_1 = [
            _make_op("clean_a"),
            _make_op("clean_b"),
            _make_op("analyze"),
            _make_op("score"),
            _make_op("format"),
        ]
        config = update_pipeline(three_step_config, new_ops_1, ["clean"], old_ops_list=old_ops_1)

        # Round 2: chain "format" in output
        old_ops_2 = config["operations"]
        new_ops_2 = [
            _make_op("clean_a"),
            _make_op("clean_b"),
            _make_op("analyze"),
            _make_op("score"),
            _make_op("format_header"),
            _make_op("format_body"),
        ]
        config = update_pipeline(config, new_ops_2, ["format"], old_ops_list=old_ops_2)

        step_ops = _op_names_per_step(config)
        assert step_ops["prep"] == ["clean_a", "clean_b"]
        assert step_ops["analysis"] == ["analyze", "score"]
        assert step_ops["output"] == ["format_header", "format_body"]


# ===========================================================================
# Test: Node.op_to_step mapping
# ===========================================================================

class TestNodeOpToStep:
    """Test that Node correctly builds the op_to_step mapping."""

    def test_op_to_step_basic(self, tmp_path):
        config = {
            "default_model": "gpt-4o-mini",
            "operations": [
                _make_op("extract"),
                _make_op("summarize"),
                _make_op("classify"),
            ],
            "datasets": {"docs": {"type": "file", "path": "data.json"}},
            "pipeline": {
                "steps": [
                    {"name": "step1", "input": "docs", "operations": ["extract", "summarize"]},
                    {"name": "step2", "input": "step1", "operations": ["classify"]},
                ],
                "output": {"type": "file", "path": str(tmp_path / "out.json")},
            },
        }
        yaml_path = tmp_path / "pipeline.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config, f)

        from docetl.moar.Node import Node
        node = Node(str(yaml_path))

        assert node.op_to_step["extract"] == "step1"
        assert node.op_to_step["summarize"] == "step1"
        assert node.op_to_step["classify"] == "step2"

    def test_op_to_step_with_equijoin(self, tmp_path):
        config = {
            "default_model": "gpt-4o-mini",
            "operations": [
                _make_op("map_a"),
                {"name": "eq_join", "type": "equijoin", "comparison_prompt": "compare"},
            ],
            "datasets": {"data": {"type": "file", "path": "data.json"}},
            "pipeline": {
                "steps": [
                    {"name": "step1", "input": "data", "operations": ["map_a"]},
                    {"name": "join_step", "input": "step1", "operations": [
                        {"eq_join": {"left": "step1", "right": "other"}}
                    ]},
                ],
                "output": {"type": "file", "path": str(tmp_path / "out.json")},
            },
        }
        yaml_path = tmp_path / "pipeline.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config, f)

        from docetl.moar.Node import Node
        node = Node(str(yaml_path))

        assert node.op_to_step["map_a"] == "step1"
        assert node.op_to_step["eq_join"] == "join_step"

    def test_op_to_step_updates_after_directive(self, tmp_path):
        """After applying a directive via update_pipeline and creating a new Node,
        the new Node's op_to_step should reflect the updated operations."""
        config = {
            "default_model": "gpt-4o-mini",
            "operations": [
                _make_op("extract"),
                _make_op("classify"),
            ],
            "datasets": {"docs": {"type": "file", "path": "data.json"}},
            "pipeline": {
                "steps": [
                    {"name": "step1", "input": "docs", "operations": ["extract"]},
                    {"name": "step2", "input": "step1", "operations": ["classify"]},
                ],
                "output": {"type": "file", "path": str(tmp_path / "out.json")},
            },
        }

        old_ops = config["operations"]
        new_ops = [
            _make_op("extract_a"),
            _make_op("extract_b"),
            _make_op("classify"),
        ]
        updated = deepcopy(config)
        update_pipeline(updated, new_ops, ["extract"], old_ops_list=old_ops)

        yaml_path = tmp_path / "pipeline_updated.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(updated, f)

        from docetl.moar.Node import Node
        node = Node(str(yaml_path))

        assert node.op_to_step["extract_a"] == "step1"
        assert node.op_to_step["extract_b"] == "step1"
        assert node.op_to_step["classify"] == "step2"
        assert "extract" not in node.op_to_step


# ===========================================================================
# Test: multiple targets in a single directive application
# ===========================================================================

class TestMultipleTargets:
    """Some directives may target multiple operations at once."""

    def test_two_targets_same_step(self, two_step_config):
        old_ops = two_step_config["operations"]
        # Both "extract" and "summarize" are in step1, both get chained
        new_ops = [
            _make_op("extract_v2"),
            _make_op("summarize_v2"),
            _make_op("classify"),
            _make_op("filter"),
        ]
        result = update_pipeline(two_step_config, new_ops, ["extract", "summarize"], old_ops_list=old_ops)
        step_ops = _op_names_per_step(result)

        assert step_ops["step1"] == ["extract_v2", "summarize_v2"]
        assert step_ops["step2"] == ["classify", "filter"]

    def test_two_targets_different_steps(self, two_step_config):
        old_ops = two_step_config["operations"]
        # "extract" in step1, "classify" in step2 — both get chained
        new_ops = [
            _make_op("extract_a"),
            _make_op("extract_b"),
            _make_op("summarize"),
            _make_op("classify_a"),
            _make_op("classify_b"),
            _make_op("filter"),
        ]
        result = update_pipeline(two_step_config, new_ops, ["extract", "classify"], old_ops_list=old_ops)
        step_ops = _op_names_per_step(result)

        assert step_ops["step1"] == ["extract_a", "extract_b", "summarize"]
        assert step_ops["step2"] == ["classify_a", "classify_b", "filter"]
