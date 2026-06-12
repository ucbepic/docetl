"""lift/lower round-trip: deep-equal, yaml-dump-equal, identity on no-op."""

import copy

import pytest
import yaml

from docetl.frame import from_list
from docetl.plan import (
    JoinNode,
    OpaqueNode,
    ScanNode,
    lift,
    lower,
    validate,
    validate_config,
)


def assert_roundtrip(config):
    snapshot = copy.deepcopy(config)
    plan = lift(config)
    lowered = lower(plan)
    assert lowered is config, "untouched plan must lower to the original object"
    assert config == snapshot, "lift/lower must not mutate the config"
    # And through a forced re-emit (simulate a rule marking steps dirty):
    for step in plan.steps:
        step.original = dict(step.original)  # break identity, keep equality
    re_emitted = lower(plan)
    assert re_emitted == snapshot
    assert yaml.safe_dump(re_emitted, sort_keys=True, default_flow_style=False) == yaml.safe_dump(
        snapshot, sort_keys=True, default_flow_style=False
    )
    return plan


def frame_config(frame):
    return frame._build_config(checkpoint=False)


class TestFrameBuiltPipelines:
    def test_every_op_type_chain(self):
        frame = (
            from_list([{"text": "a b", "tags": ["x"], "grp": "g"}])
            .map("m", prompt="p {{ input.text }}", output={"schema": {"summary": "string"}})
            .parallel_map(
                "pm",
                prompts=[{"prompt": "q {{ input.summary }}", "output_keys": ["alt"]}],
                output={"schema": {"alt": "string"}},
            )
            .filter("f", prompt="k {{ input.summary }}", output={"schema": {"keep": "boolean"}})
            .unnest("u", unnest_key="tags")
            .split("sp", split_key="summary", method="token_count", method_kwargs={"num_tokens": 10})
            .gather("g", content_key="summary_chunk", doc_id_key="sp_id", order_key="sp_chunk_num")
            .extract("ex", prompt="e {{ input.summary }}", document_keys=["summary"])
            .sample("sa", method="first", samples=2)
            .cluster("cl", embedding_keys=["summary"], summary_schema={"t": "string"}, summary_prompt="s {{ inputs }}")
            .resolve("rs", comparison_prompt="c {{ input1.summary }} {{ input2.summary }}",
                     resolution_prompt="r {{ inputs }}", output={"schema": {"summary": "string"}})
            .reduce("rd", reduce_key="grp", prompt="agg {{ inputs }}", output={"schema": {"total": "string"}})
            .code_map("cm", code="def transform(doc):\n    return {'n': 1}")
            .code_filter("cf", code="def transform(doc):\n    return True")
            .code_reduce("cr", reduce_key="grp", code="def transform(items):\n    return {'c': len(items)}")
        )
        assert_roundtrip(frame_config(frame))

    def test_equijoin_of_two_pipelines(self):
        left = from_list([{"k": 1}], name="l").map(
            "lm", prompt="x {{ input.k }}", output={"schema": {"a": "string"}})
        right = from_list([{"k": 2}], name="r").map(
            "rm", prompt="y {{ input.k }}", output={"schema": {"b": "string"}})
        joined = left.equijoin(right, "j", comparison_prompt="c {{ left.k }} {{ right.k }}").code_map(
            "post", code="def transform(doc):\n    return {'z': 1}")

        plan = assert_roundtrip(frame_config(joined))
        join_nodes = [n for n in plan.nodes() if isinstance(n, JoinNode)]
        assert len(join_nodes) == 1
        assert join_nodes[0].left_ref == "step_lm"
        assert join_nodes[0].right_ref == "step_rm"
        assert {type(i).__name__ for i in join_nodes[0].inputs} != {"ScanNode"}

    def test_equijoin_of_raw_datasets(self):
        left = from_list([{"k": 1}], name="l")
        right = from_list([{"k": 2}], name="r")
        joined = left.equijoin(right, "j", comparison_prompt="c {{ left.k }} {{ right.k }}")

        plan = assert_roundtrip(frame_config(joined))
        join = next(n for n in plan.nodes() if isinstance(n, JoinNode))
        assert all(isinstance(i, ScanNode) for i in join.inputs)

    def test_shared_ancestry_dag(self):
        base = from_list([{"k": 1}]).map("shared", prompt="s {{ input.k }}", output={"schema": {"a": "string"}})
        left = base.filter("lf", prompt="l {{ input.a }}", output={"schema": {"keep": "boolean"}})
        right = base.filter("rf", prompt="r {{ input.a }}", output={"schema": {"keep": "boolean"}})
        joined = left.equijoin(right, comparison_prompt="c {{ left.a }} {{ right.a }}")
        assert_roundtrip(frame_config(joined))

    def test_retrievers_pass_through(self):
        config = frame_config(
            from_list([{"q": "x"}]).map("m", prompt="p {{ input.q }}", output={"schema": {"a": "string"}})
        )
        config["retrievers"] = {"r1": {"type": "fts", "dataset": "docs", "query": "{{ input.q }}"}}
        assert_roundtrip(config)


class TestYamlShapedPipelines:
    def yaml_config(self, **overrides):
        config = {
            "datasets": {"docs": {"type": "file", "path": "docs.json"}},
            "default_model": "gpt-4o-mini",
            "some_unknown_top_level_key": {"nested": [1, 2]},
            "operations": [
                {
                    "name": "summarize",
                    "type": "map",
                    "prompt": "Summarize {{ input.text }}",
                    "output": {"schema": {"summary": "string"}},
                    "an_unknown_op_key": True,
                },
                {
                    "name": "keep",
                    "type": "code_filter",
                    "code": "def transform(doc):\n    return True",
                },
            ],
            "pipeline": {
                "steps": [
                    {
                        "name": "main",
                        "input": "docs",
                        "operations": ["summarize", "keep"],
                        "unknown_step_key": "kept",
                    },
                ],
                "output": {"type": "file", "path": "out.json"},
            },
        }
        config.update(overrides)
        return config

    def test_multi_op_step_with_unknown_keys(self):
        assert_roundtrip(self.yaml_config())

    def test_op_referenced_from_two_steps(self):
        config = self.yaml_config()
        config["pipeline"]["steps"] = [
            {"name": "first", "input": "docs", "operations": ["summarize"]},
            {"name": "second", "input": "docs", "operations": ["summarize", "keep"]},
        ]
        plan = assert_roundtrip(config)
        assert plan.references("summarize") == 2

    def test_unknown_op_type_is_opaque_not_fatal(self):
        config = self.yaml_config()
        config["operations"].append({"name": "mystery", "type": "totally_custom_plugin"})
        config["pipeline"]["steps"][0]["operations"].append("mystery")
        plan = assert_roundtrip(config)
        mystery = next(n for n in plan.nodes() if n.name == "mystery")
        assert isinstance(mystery, OpaqueNode)
        issues = validate(plan)
        assert any(i.level == "info" and i.where == "mystery" for i in issues)
        assert not any(i.level == "error" for i in issues)

    def test_callable_code_carried_by_reference(self):
        def transform(doc):
            return {"y": doc["x"]}

        config = self.yaml_config()
        config["operations"][1] = {"name": "keep", "type": "code_filter", "code": transform}
        plan = lift(config)
        assert lower(plan) is config
        node = next(n for n in plan.nodes() if n.name == "keep")
        assert node.op_config["code"] is transform


class TestRandomizedChains:
    @pytest.mark.parametrize("seed", range(5))
    def test_random_code_op_chains(self, seed):
        import random

        rng = random.Random(seed)
        frame = from_list([{"x": i, "grp": i % 2} for i in range(6)])
        for i in range(rng.randint(1, 8)):
            kind = rng.choice(["code_map", "code_filter", "sample", "unnest_ok"])
            if kind == "code_map":
                frame = frame.code_map(f"cm{i}", code=f"def transform(doc):\n    return {{'k{i}': {i}}}")
            elif kind == "code_filter":
                frame = frame.code_filter(f"cf{i}", code="def transform(doc):\n    return True")
            elif kind == "sample":
                frame = frame.sample(f"sa{i}", method="first", samples=3)
            else:
                frame = frame.code_map(f"u{i}", code="def transform(doc):\n    return {'tags': [1]}").unnest(
                    f"un{i}", unnest_key="tags"
                )
        assert_roundtrip(frame_config(frame))


class TestLiftStructuralIssues:
    def test_dangling_op_reference(self):
        issues = validate_config(
            {
                "datasets": {"d": {"type": "file", "path": "x.json"}},
                "operations": [],
                "pipeline": {"steps": [{"name": "s", "input": "d", "operations": ["ghost"]}]},
            }
        )
        assert any(i.level == "error" and "ghost" in i.message for i in issues)

    def test_dangling_input_reference(self):
        issues = validate_config(
            {
                "datasets": {},
                "operations": [
                    {"name": "c", "type": "code_map", "code": "def transform(doc):\n    return {}"}
                ],
                "pipeline": {"steps": [{"name": "s", "input": "nowhere", "operations": ["c"]}]},
            }
        )
        assert any(i.level == "error" and "nowhere" in i.message for i in issues)

    def test_invalid_op_config_caught_statically(self):
        # Runtime syntax_check failure (map without prompt/output) surfaced
        # before execution.
        issues = validate_config(
            {
                "datasets": {"d": {"type": "file", "path": "x.json"}},
                "operations": [{"name": "bad", "type": "map"}],
                "pipeline": {"steps": [{"name": "s", "input": "d", "operations": ["bad"]}]},
            }
        )
        assert any(i.level == "error" and i.where == "bad" for i in issues)

    def test_read_after_definite_removal(self):
        issues = validate_config(
            {
                "datasets": {"d": {"type": "file", "path": "x.json"}},
                "operations": [
                    {
                        "name": "dropper",
                        "type": "map",
                        "prompt": "p {{ input.text }}",
                        "output": {"schema": {"s": "string"}},
                        "drop_keys": ["raw"],
                    },
                    {
                        "name": "reader",
                        "type": "code_filter",
                        "code": "def transform(doc):\n    return doc['raw']",
                    },
                ],
                "pipeline": {
                    "steps": [{"name": "s", "input": "d", "operations": ["dropper", "reader"]}]
                },
            }
        )
        # Warning, not error: the read-set extractor has known false
        # positives (guarded reads), so this check must not reject plans.
        assert any(i.level == "warning" and "raw" in i.message and i.where == "reader" for i in issues)

    def test_import_docetl_plan_submodule(self):
        # The sys.modules replacement in docetl/__init__.py must not break
        # subpackage imports.
        import importlib

        module = importlib.import_module("docetl.plan")
        assert hasattr(module, "lift")


class TestJoinEntryUnknownKeys:
    def config(self):
        return {
            "datasets": {
                "l": {"type": "file", "path": "l.json"},
                "r": {"type": "file", "path": "r.json"},
            },
            "operations": [
                {
                    "name": "j",
                    "type": "equijoin",
                    "comparison_prompt": "same? {{ left.a }} {{ right.b }}",
                },
                {
                    "name": "keep",
                    "type": "code_filter",
                    "code": "def transform(doc):\n    return doc['x'] > 1",
                },
                {
                    "name": "m",
                    "type": "map",
                    "prompt": "p {{ input.text }}",
                    "output": {"schema": {"s": "string"}},
                },
            ],
            "pipeline": {
                "steps": [
                    {
                        "name": "joined",
                        "operations": [
                            # Unknown key inside the join entry must survive.
                            {"j": {"left": "l", "right": "r", "note": "KEEPME"}}
                        ],
                    },
                    {"name": "post", "input": "joined", "operations": ["m", "keep"]},
                ],
                "output": {"type": "file", "path": "out.json"},
            },
        }

    def test_roundtrip_preserves_join_entry_keys(self):
        import yaml as _yaml

        from docetl.plan import lift, lower

        config = self.config()
        lowered = lower(lift(config))
        assert lowered is config or _yaml.safe_dump(
            lowered, sort_keys=False
        ) == _yaml.safe_dump(config, sort_keys=False)

    def test_rewrite_elsewhere_keeps_join_entry_keys(self, monkeypatch):
        # Regression: when a rule fired anywhere, lower used to regenerate
        # the join entry as bare {left, right}, dropping unknown keys.
        monkeypatch.setattr(
            "docetl.plan.rules.pushdown._chain_has_llm", lambda plan, node: True
        )
        from docetl.plan import apply_rewrites_to_config

        config = self.config()
        rewritten, applied = apply_rewrites_to_config(config)
        assert applied, "the m→keep pushdown should fire"
        join_entry = rewritten["pipeline"]["steps"][0]["operations"][0]
        assert join_entry["j"]["note"] == "KEEPME"
