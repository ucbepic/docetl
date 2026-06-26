"""MOAR integration tests.

Structural tests (no LLM calls): invalid directive candidates rejected
statically (before any execution), valid candidates canonicalized by the
rewrite rules in their saved YAMLs.

End-to-end tests (real LLM calls): run a full MOAR search loop against
3 models and verify it produces at least one frontier pipeline."""

import json
import os

import pytest
import yaml

from docetl.moar.MOARSearch import MOARSearch
from docetl.moar.Node import Node
from docetl.plan import InvalidCandidatePlan, validate_config


def root_yaml(tmp_path):
    config = {
        "datasets": {"docs": {"type": "file", "path": str(tmp_path / "docs.json")}},
        "default_model": "gpt-4o-mini",
        "operations": [
            {
                "name": "m",
                "type": "map",
                "prompt": "Summarize {{ input.text }}",
                "output": {"schema": {"summary": "string"}},
            },
            {
                "name": "f",
                "type": "code_filter",
                "code": "def transform(doc):\n    return doc['category'] == 'a'",
            },
        ],
        "pipeline": {
            "steps": [
                {"name": "s1", "input": "docs", "operations": ["m"]},
                {"name": "s2", "input": "s1", "operations": ["f"]},
            ],
            "output": {"type": "file", "path": str(tmp_path / "out.json")},
        },
    }
    path = tmp_path / "root.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return str(path)


def make_search(tmp_path):
    """A minimally-wired MOARSearch: enough state for instantiate_node."""
    search = object.__new__(MOARSearch)
    search.console = __import__("docetl.console", fromlist=["DOCETL_CONSOLE"]).DOCETL_CONSOLE
    search.sample_dataset_path = None
    search.output_dir = str(tmp_path / "candidates")
    search.directive_name_to_obj = {}
    return search


class TestCandidateValidation:
    def test_invalid_candidate_rejected_before_writing(self, tmp_path):
        node = Node(root_yaml(tmp_path))
        search = make_search(tmp_path)

        # Directive output that breaks the pipeline: the map vanished from
        # the ops list, so the step now references an undefined op... but
        # update_pipeline patches steps. Instead: a malformed replacement op
        # (map without prompt/output) — exactly what used to explode at
        # runtime in Node.execute_plan.
        new_ops_list = [
            {"name": "m", "type": "map"},  # invalid: no prompt/output/drop_keys
            node.parsed_yaml["operations"][1],
        ]
        with pytest.raises(InvalidCandidatePlan) as exc_info:
            search.instantiate_node(
                node, new_ops_list, "broken_directive", ["m"], "acc", []
            )
        assert any(issue.where == "m" for issue in exc_info.value.issues)
        # Rejected before any candidate yaml was written.
        candidate_dir = tmp_path / "candidates"
        assert not candidate_dir.exists() or not list(candidate_dir.glob("*.yaml"))

    def test_valid_candidate_written_and_canonicalized(self, tmp_path):
        node = Node(root_yaml(tmp_path))
        search = make_search(tmp_path)

        # A no-op "directive": same ops back. The selection pushdown applies
        # to the root shape (LLM map then disjoint code filter), so the
        # saved candidate must already carry the rewritten step structure.
        new_ops_list = list(node.parsed_yaml["operations"])
        child = search.instantiate_node(
            node, new_ops_list, "noop_directive", ["m"], "acc", []
        )
        assert os.path.exists(child.yaml_file_path)
        saved = yaml.safe_load(open(child.yaml_file_path))
        steps = saved["pipeline"]["steps"]
        assert len(steps) == 1
        assert steps[0]["operations"] == ["f", "m"]
        # Canonicalization is a fixpoint: the saved yaml validates clean and
        # re-rewriting changes nothing.
        from docetl.plan import apply_rewrites_to_config

        reparsed, applied = apply_rewrites_to_config(saved)
        assert reparsed is saved and not applied
        assert not [i for i in validate_config(saved) if i.level == "error"]


class TestRootValidationWarnings:
    def test_validate_config_flags_root_problems(self, tmp_path):
        # MOARSearch.__init__ logs these as warnings; here we check the
        # underlying detection on a root with a dangling op reference.
        config = {
            "datasets": {"docs": {"type": "file", "path": "x.json"}},
            "operations": [],
            "pipeline": {
                "steps": [{"name": "s1", "input": "docs", "operations": ["ghost"]}],
                "output": {"type": "file", "path": "o.json"},
            },
        }
        errors = [i for i in validate_config(config) if i.level == "error"]
        assert errors and "ghost" in errors[0].message


class TestCandidateRespectsPlanRewritesSetting:
    def test_opt_out_is_honored(self, tmp_path):
        """Regression: candidate canonicalization used to apply the default
        rules unconditionally, baking rewrites into saved YAMLs even when
        the root pipeline set plan_rewrites: false."""
        config = yaml.safe_load(open(root_yaml(tmp_path)))
        config["plan_rewrites"] = False
        opt_out_path = tmp_path / "root_optout.yaml"
        opt_out_path.write_text(yaml.safe_dump(config, sort_keys=False))

        node = Node(str(opt_out_path))
        search = make_search(tmp_path)
        child = search.instantiate_node(
            node, list(node.parsed_yaml["operations"]), "noop_directive", ["m"], "acc", []
        )
        saved = yaml.safe_load(open(child.yaml_file_path))
        steps = saved["pipeline"]["steps"]
        # Same shape as the root: NOT rewritten to the pushed-down form.
        assert [s["operations"] for s in steps] == [["m"], ["f"]]
        assert saved["plan_rewrites"] is False  # carried for the runner

    def test_rule_subset_is_honored(self, tmp_path):
        config = yaml.safe_load(open(root_yaml(tmp_path)))
        config["plan_rewrites"] = ["limit_pushdown"]  # not selection_pushdown
        subset_path = tmp_path / "root_subset.yaml"
        subset_path.write_text(yaml.safe_dump(config, sort_keys=False))

        node = Node(str(subset_path))
        search = make_search(tmp_path)
        child = search.instantiate_node(
            node, list(node.parsed_yaml["operations"]), "noop_directive", ["m"], "acc", []
        )
        saved = yaml.safe_load(open(child.yaml_file_path))
        assert [s["operations"] for s in saved["pipeline"]["steps"]] == [["m"], ["f"]]


# ── End-to-end MOAR test (real LLM calls) ────────────────────────────


MOAR_MODELS = ["gpt-4o-mini", "gpt-5-nano", "gpt-5-mini"]


@pytest.fixture
def moar_pipeline(tmp_path):
    """A tiny pipeline suitable for a real MOAR run."""
    data = [
        {"text": "The Eiffel Tower is located in Paris, France."},
        {"text": "Mount Fuji is the tallest mountain in Japan."},
        {"text": "The Great Wall of China stretches over 13,000 miles."},
    ]
    data_path = tmp_path / "docs.json"
    data_path.write_text(json.dumps(data))

    config = {
        "default_model": "gpt-4o-mini",
        "datasets": {
            "docs": {"type": "file", "path": str(data_path)},
        },
        "operations": [
            {
                "name": "extract_info",
                "type": "map",
                "prompt": (
                    "Extract the landmark and its location from the text.\n"
                    "Text: {{ input.text }}"
                ),
                "output": {
                    "schema": {
                        "landmark": "string",
                        "location": "string",
                    }
                },
                "optimize": True,
            },
        ],
        "pipeline": {
            "steps": [
                {
                    "name": "extraction",
                    "input": "docs",
                    "operations": ["extract_info"],
                }
            ],
            "output": {"type": "file", "path": str(tmp_path / "output.json")},
        },
    }
    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return str(yaml_path), str(data_path), str(tmp_path)


def _eval_landmarks(results_path: str) -> dict:
    """Score: fraction of results that have non-empty landmark + location."""
    with open(results_path) as f:
        results = json.load(f)
    if not results:
        return {"score": 0.0}
    good = sum(
        1
        for r in results
        if r.get("landmark", "").strip() and r.get("location", "").strip()
    )
    return {"score": good / len(results)}


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY required for real MOAR test",
)
class TestMOAREndToEnd:
    def test_moar_search_with_three_models(self, moar_pipeline):
        """Run a full (1-iteration) MOAR search with real LLM calls."""
        from docetl.moar.optimizer import MOAROptimizer

        yaml_path, data_path, save_dir = moar_pipeline

        optimizer = MOAROptimizer(
            pipeline=yaml_path,
            eval_fn=_eval_landmarks,
            metric_key="score",
            models=MOAR_MODELS,
            agent_model="gpt-4o-mini",
            max_iterations=1,
            save_dir=os.path.join(save_dir, "moar_output"),
            dataset_path=data_path,
            max_concurrent_agents=1,
        )

        result = optimizer.optimize()

        assert result.iterations >= 1
        assert len(result.all_plans) >= 1
        assert result.frontier, "Expected at least one frontier pipeline"

        best = result.best()
        assert best is not None
        assert best.accuracy >= 0.0
        assert best.cost >= 0.0
        assert os.path.exists(best.yaml_path)
