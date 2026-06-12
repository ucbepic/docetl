"""Runner-hook safety: zero hash churn when no rule fires, stable
checkpoints under rewriting, and untouched Frame memoization."""

import copy

import pytest

from docetl.api import Dataset, MapOp, Pipeline, PipelineOutput, PipelineStep
from docetl.frame import from_list
from docetl.runner import DSLRunner


def code_config(marker, tmp_path, intermediate_dir=None, **extra):
    """A code-only map→filter pipeline where no default rule fires
    (benefit gate: no LLM op)."""
    code = (
        "def transform(doc):\n"
        f"    open({str(marker)!r}, 'a').write('x')\n"
        "    return {'y': doc['x'] * 2}"
    )
    output = {"type": "file", "path": str(tmp_path / "out.json")}
    if intermediate_dir:
        output["intermediate_dir"] = str(intermediate_dir)
    config = {
        "datasets": {"docs": {"type": "memory", "path": [{"x": i} for i in range(4)]}},
        "operations": [
            {"name": "cm", "type": "code_map", "code": code},
            {"name": "cf", "type": "code_filter", "code": "def transform(doc):\n    return doc['x'] < 3"},
        ],
        "pipeline": {
            "steps": [
                {"name": "s1", "input": "docs", "operations": ["cm"]},
                {"name": "s2", "input": "s1", "operations": ["cf"]},
            ],
            "output": output,
        },
    }
    config.update(extra)
    return config


def rewritable_config(tmp_path, intermediate_dir=None):
    """An LLM map followed by a code filter on a disjoint field — the
    selection pushdown fires, but nothing here is ever executed."""
    output = {"type": "file", "path": str(tmp_path / "out.json")}
    if intermediate_dir:
        output["intermediate_dir"] = str(intermediate_dir)
    return {
        "datasets": {"docs": {"type": "memory", "path": [{"text": "t", "category": "a"}]}},
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
            "output": output,
        },
    }


class TestNoRuleFires:
    def test_config_object_identity_preserved(self, tmp_path):
        config = code_config(tmp_path / "m.log", tmp_path)
        runner = DSLRunner(config)
        assert runner.applied_rewrites == []
        assert runner.config is config

    def test_pipeline_object_identity_preserved(self, tmp_path):
        pipeline = Pipeline(
            name="p",
            datasets={"docs": Dataset(type="memory", path=[{"x": 1}])},
            operations=[
                MapOp(
                    name="m",
                    type="map",
                    prompt="p {{ input.x }}",
                    output={"schema": {"s": "string"}},
                )
            ],
            steps=[PipelineStep(name="s1", input="docs", operations=["m"])],
            output=PipelineOutput(type="file", path=str(tmp_path / "out.json")),
            default_model="gpt-4o-mini",
        )
        runner = DSLRunner(pipeline)
        assert runner.applied_rewrites == []
        assert runner.pipeline is pipeline

    def test_hashes_identical_with_flag_on_and_off(self, tmp_path):
        inter = tmp_path / "inter"
        on = DSLRunner(code_config(tmp_path / "a.log", tmp_path, inter))
        off = DSLRunner(
            code_config(tmp_path / "a.log", tmp_path, inter, plan_rewrites=False)
        )
        on_hashes = {k: dict(v) for k, v in on.step_op_hashes.items()}
        off_hashes = {k: dict(v) for k, v in off.step_op_hashes.items()}
        assert on_hashes == off_hashes

    def test_flag_off_blocks_rewrites(self, tmp_path):
        config = rewritable_config(tmp_path)
        config["plan_rewrites"] = False
        runner = DSLRunner(config)
        assert runner.applied_rewrites == []
        assert runner.config is config

    def test_rule_name_list_selects_rules(self, tmp_path):
        config = rewritable_config(tmp_path)
        config["plan_rewrites"] = ["limit_pushdown"]  # excludes selection pushdown
        runner = DSLRunner(config)
        assert runner.applied_rewrites == []
        assert runner.config is config


class TestRewrittenPipelines:
    def test_rewrite_fires_and_step_structure_updates(self, tmp_path):
        config = rewritable_config(tmp_path)
        snapshot = copy.deepcopy(config)
        runner = DSLRunner(config)
        assert [r.rule for r in runner.applied_rewrites] == ["selection_pushdown"]
        steps = runner.config["pipeline"]["steps"]
        assert len(steps) == 1
        assert steps[0]["operations"] == ["f", "m"]
        assert config == snapshot, "caller's config must not be mutated"

    def test_reload_of_rewritten_config_is_hash_stable(self, tmp_path):
        runner = DSLRunner(rewritable_config(tmp_path, tmp_path / "inter"))
        assert runner.applied_rewrites
        first = {k: dict(v) for k, v in runner.step_op_hashes.items()}
        rewritten = runner.config
        runner.reload(runner.config)
        assert runner.applied_rewrites == []  # fixpoint: nothing more fires
        assert runner.config is rewritten
        assert {k: dict(v) for k, v in runner.step_op_hashes.items()} == first

    def test_second_run_fully_cached(self, tmp_path, monkeypatch):
        # With the benefit gate patched, the limit pushdown fires on a
        # code-only pipeline; both runs key checkpoints on the rewritten
        # config, so the second run executes nothing.
        monkeypatch.setattr(
            "docetl.plan.rules.pushdown._chain_has_llm", lambda plan, node: True
        )
        marker = tmp_path / "runs.log"
        inter = tmp_path / "inter"

        def build():
            # docs → code_map → sample(first): the head hops the 1:1 map.
            # LimitPushdown is opt-in, so the test selects it explicitly.
            config = code_config(marker, tmp_path, inter)
            config["operations"] = [
                config["operations"][0],
                {"name": "head", "type": "sample", "method": "first", "samples": 2},
            ]
            config["pipeline"]["steps"] = [
                {"name": "s1", "input": "docs", "operations": ["cm"]},
                {"name": "s2", "input": "s1", "operations": ["head"]},
            ]
            config["plan_rewrites"] = ["selection_pushdown", "limit_pushdown"]
            return DSLRunner(config)

        first = build()
        assert first.applied_rewrites
        out1, _ = first.run()
        executed_first = len(marker.read_text())
        assert executed_first > 0

        second = build()
        out2, _ = second.run()
        assert out2 == out1
        assert len(marker.read_text()) == executed_first, "second run must be fully cached"


class TestFrameMemoization:
    def test_collect_twice_executes_once(self, tmp_path):
        marker = tmp_path / "runs.log"
        code = (
            "def transform(doc):\n"
            f"    open({str(marker)!r}, 'a').write('x')\n"
            "    return {'y': doc['x']}"
        )
        frame = from_list([{"x": i} for i in range(3)]).code_map("m", code=code)
        first = frame.collect()
        second = frame.collect()
        assert first == second
        assert len(marker.read_text()) == 3


class TestGlobalKillSwitch:
    def test_module_global_disables_rewrites_for_dict_configs(self, tmp_path, monkeypatch):
        """Regression: docetl.plan_rewrites=False used to reach only the
        Frame API; the runner gate now falls back to it for every
        in-process construction."""
        import docetl._config as _config

        config = rewritable_config(tmp_path)
        runner_on = DSLRunner(copy.deepcopy(config))
        assert runner_on.applied_rewrites, "sanity: pushdown fires by default"

        monkeypatch.setattr(_config, "plan_rewrites", False)
        cfg = copy.deepcopy(config)
        runner_off = DSLRunner(cfg)
        assert runner_off.applied_rewrites == []
        assert runner_off.config is cfg

    def test_config_key_overrides_module_global(self, tmp_path, monkeypatch):
        import docetl._config as _config

        monkeypatch.setattr(_config, "plan_rewrites", False)
        config = rewritable_config(tmp_path)
        config["plan_rewrites"] = True
        runner = DSLRunner(config)
        assert runner.applied_rewrites

    def test_misconfigured_setting_raises(self, tmp_path):
        config = rewritable_config(tmp_path)
        config["plan_rewrites"] = ["selection-pushdwn"]  # typo'd name
        with pytest.raises(ValueError, match="selection-pushdwn"):
            DSLRunner(config)


class TestPrepareConfigSingleLift:
    def test_one_lift_for_validate_and_rewrite(self, tmp_path, monkeypatch):
        import docetl.plan.prepare as prepare_mod
        from docetl.plan import prepare_config
        from docetl.plan.lift import lift as real_lift

        calls = []

        def counting_lift(config):
            calls.append(1)
            return real_lift(config)

        monkeypatch.setattr(prepare_mod, "lift", counting_lift)
        config = rewritable_config(tmp_path)
        out, issues, applied = prepare_config(config)
        assert len(calls) == 1
        assert applied and out is not config
        assert not [i for i in issues if i.level == "error"]
