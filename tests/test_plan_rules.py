"""Rewrite-rule tests: qualification conditions, graph surgery, and
LLM-free execution equivalence with counting instrumentation."""

import copy

import pytest

from docetl.plan import apply_rules, lift, lower
from docetl.plan.rewrite import push_below
from docetl.plan.rules.pushdown import LimitPushdown, SelectionPushdown
from docetl.runner import DSLRunner


def make_config(operations, steps, datasets=None, **extra):
    config = {
        "datasets": datasets or {"docs": {"type": "file", "path": "docs.json"}},
        "operations": operations,
        "pipeline": {
            "steps": steps,
            "output": {"type": "file", "path": "out.json"},
        },
    }
    config.update(extra)
    return config


def llm_map(name="m", prompt="Summarize {{ input.text }}", **extra):
    return {
        "name": name,
        "type": "map",
        "prompt": prompt,
        "output": {"schema": {"summary": "string"}},
        **extra,
    }


def code_filter(name="f", code="def transform(doc):\n    return doc['category'] == 'a'"):
    return {"name": name, "type": "code_filter", "code": code}


def two_step_config(map_op=None, filter_op=None):
    map_op = map_op or llm_map()
    filter_op = filter_op or code_filter()
    return make_config(
        [map_op, filter_op],
        [
            {"name": "s1", "input": "docs", "operations": [map_op["name"]]},
            {"name": "s2", "input": "s1", "operations": [filter_op["name"]]},
        ],
    )


class TestSelectionPushdownConditions:
    def fires(self, config):
        plan = lift(config)
        return bool(apply_rules(plan, rules=[SelectionPushdown()]))

    def test_fires_on_disjoint_fields(self):
        assert self.fires(two_step_config())

    def test_blocked_when_filter_reads_map_output(self):
        blocked = two_step_config(
            filter_op=code_filter(code="def transform(doc):\n    return len(doc['summary']) > 1")
        )
        assert not self.fires(blocked)

    def test_blocked_when_filter_reads_dropped_key(self):
        blocked = two_step_config(map_op=llm_map(drop_keys=["category"]))
        assert not self.fires(blocked)

    @pytest.mark.parametrize(
        "extra",
        [
            {"skip_on_error": True},
            {"limit": 3},
            {"validate": ["len(output['summary']) > 0"]},
            {"calibrate": True},
            {"tools": [{"code": "x", "function": {"name": "f"}}]},
        ],
    )
    def test_blocked_by_unsafe_map_configs(self, extra):
        assert not self.fires(two_step_config(map_op=llm_map(**extra)))

    def test_blocked_when_filter_reads_unknown(self):
        blocked = two_step_config(
            filter_op=code_filter(code="def transform(doc):\n    return bool(dict(doc))")
        )
        assert not self.fires(blocked)

    def test_blocked_without_llm_benefit(self):
        config = make_config(
            [
                {"name": "cm", "type": "code_map", "code": "def transform(doc):\n    return {'y': 1}"},
                code_filter(),
            ],
            [
                {"name": "s1", "input": "docs", "operations": ["cm"]},
                {"name": "s2", "input": "s1", "operations": ["cf" if False else "f"]},
            ],
        )
        assert not self.fires(config)

    def test_blocked_when_map_feeds_two_consumers(self):
        config = make_config(
            [llm_map(), code_filter("f1"), code_filter("f2")],
            [
                {"name": "s1", "input": "docs", "operations": ["m"]},
                {"name": "s2", "input": "s1", "operations": ["f1"]},
                {"name": "s3", "input": "s1", "operations": ["f2"]},
            ],
        )
        assert not self.fires(config)

    def test_blocked_when_op_config_shared_across_steps(self):
        config = make_config(
            [llm_map(), code_filter()],
            [
                {"name": "s1", "input": "docs", "operations": ["m"]},
                {"name": "s2", "input": "s1", "operations": ["f"]},
                {"name": "s3", "input": "docs", "operations": ["m", "f"]},
            ],
        )
        assert not self.fires(config)

    def test_llm_filter_with_annotation_needs_disjoint_writes(self):
        # _short_explanation survives on kept rows; the map must provably
        # not read or write it for the swap to stay equivalent.
        annotated_filter = {
            "name": "f",
            "type": "filter",
            "prompt": "keep? {{ input.category }}",
            "output": {"schema": {"keep": "boolean", "_short_explanation": "string"}},
        }
        assert self.fires(two_step_config(filter_op=annotated_filter))
        map_reading_annotation = llm_map(prompt="p {{ input._short_explanation }}")
        assert not self.fires(
            two_step_config(map_op=map_reading_annotation, filter_op=annotated_filter)
        )

    def test_fires_transitively_through_llm_chain(self):
        config = make_config(
            [
                llm_map("m1", prompt="a {{ input.text }}"),
                {
                    "name": "m2",
                    "type": "map",
                    "prompt": "b {{ input.summary }}",
                    "output": {"schema": {"extra": "string"}},
                },
                code_filter(),
            ],
            [
                {"name": "s1", "input": "docs", "operations": ["m1", "m2"]},
                {"name": "s2", "input": "s1", "operations": ["f"]},
            ],
        )
        plan = lift(config)
        applied = apply_rules(plan, rules=[SelectionPushdown()])
        assert len(applied) == 2  # hops below m2, then below m1
        names = [n.name for n in plan.nodes()]
        assert names == ["f", "m1", "m2"]

    def test_rewritten_config_shape(self):
        config = two_step_config()
        plan = lift(config)
        applied = apply_rules(plan)
        assert applied
        rewritten = lower(plan)
        assert rewritten is not config
        steps = rewritten["pipeline"]["steps"]
        assert len(steps) == 1  # emptied filter step dropped
        assert steps[0]["operations"] == ["f", "m"]
        # The flat ops list keeps its original order — only steps moved.
        assert [op["name"] for op in rewritten["operations"]] == ["m", "f"]


class TestLimitPushdownConditions:
    def first_sample(self, **extra):
        return {"name": "head", "type": "sample", "method": "first", "samples": 2, **extra}

    def config_with(self, sample_op):
        return make_config(
            [llm_map(), sample_op],
            [
                {"name": "s1", "input": "docs", "operations": ["m"]},
                {"name": "s2", "input": "s1", "operations": ["head"]},
            ],
        )

    def test_fires_for_first(self):
        plan = lift(self.config_with(self.first_sample()))
        assert apply_rules(plan, rules=[LimitPushdown()])
        assert [n.name for n in plan.nodes()] == ["head", "m"]

    def test_uniform_and_stratified_blocked(self):
        uniform = {"name": "head", "type": "sample", "method": "uniform", "samples": 2}
        assert not apply_rules(lift(self.config_with(uniform)), rules=[LimitPushdown()])
        stratified = self.first_sample(stratify_key="grp")
        assert not apply_rules(lift(self.config_with(stratified)), rules=[LimitPushdown()])

    def test_blocked_over_reordering_op(self):
        config = make_config(
            [
                {"name": "rk", "type": "rank", "prompt": "p", "input_keys": ["text"],
                 "direction": "desc"},
                self.first_sample(),
            ],
            [
                {"name": "s1", "input": "docs", "operations": ["rk"]},
                {"name": "s2", "input": "s1", "operations": ["head"]},
            ],
        )
        assert not apply_rules(lift(config), rules=[LimitPushdown()])


class TestExecutionEquivalence:
    """Run real (code-only) pipelines and prove the surgery preserves
    outputs while the upstream op touches fewer rows."""

    ROWS = [{"x": i, "category": "a" if i % 2 == 0 else "b"} for i in range(6)]

    def counting_map(self, marker):
        code = (
            "def transform(doc):\n"
            f"    open({str(marker)!r}, 'a').write('x')\n"
            "    return {'y': doc['x'] * 10}"
        )
        return {"name": "cm", "type": "code_map", "code": code}

    def base_config(self, marker, selection_op, tmp_path, suffix):
        out = tmp_path / f"out_{suffix}.json"
        return make_config(
            [self.counting_map(marker), selection_op],
            [
                {"name": "s1", "input": "docs", "operations": ["cm"]},
                {"name": "s2", "input": "s1", "operations": [selection_op["name"]]},
            ],
            datasets={"docs": {"type": "memory", "path": copy.deepcopy(self.ROWS)}},
        )

    def run_config(self, config):
        # plan_rewrites off so we execute exactly the given shape.
        output, _ = DSLRunner({**config, "plan_rewrites": False}).run()
        return output

    def manually_pushed(self, config):
        """Apply the push_below surgery directly (the rules' benefit gate
        requires an LLM op; here we assert the *surgery* is sound)."""
        plan = lift(config)
        selection = plan.steps[1].nodes[0]
        push_below(plan, selection, selection.inputs[0])
        rewritten = lower(plan)
        assert rewritten is not config
        return rewritten

    def test_filter_pushdown_equivalent_and_cheaper(self, tmp_path):
        marker_a, marker_b = tmp_path / "a.log", tmp_path / "b.log"
        keep_evens = {
            "name": "f",
            "type": "code_filter",
            "code": "def transform(doc):\n    return doc['category'] == 'a'",
        }
        original = self.base_config(marker_a, keep_evens, tmp_path, "orig")
        rewritten = self.manually_pushed(self.base_config(marker_b, keep_evens, tmp_path, "rw"))

        out_original = self.run_config(original)
        out_rewritten = self.run_config(rewritten)
        assert out_original == out_rewritten
        assert len(marker_a.read_text()) == 6  # map ran on every row
        assert len(marker_b.read_text()) == 3  # map ran only on kept rows

    def test_limit_pushdown_equivalent_and_cheaper(self, tmp_path):
        marker_a, marker_b = tmp_path / "a.log", tmp_path / "b.log"
        head = {"name": "head", "type": "sample", "method": "first", "samples": 2}
        original = self.base_config(marker_a, head, tmp_path, "orig")
        rewritten = self.manually_pushed(self.base_config(marker_b, head, tmp_path, "rw"))

        out_original = self.run_config(original)
        out_rewritten = self.run_config(rewritten)
        assert out_original == out_rewritten
        assert len(out_rewritten) == 2
        assert len(marker_a.read_text()) == 6
        assert len(marker_b.read_text()) == 2

    def test_rules_on_vs_off_identical_output(self, tmp_path, monkeypatch):
        # End to end through the runner hook: patch the benefit gate so the
        # real rule fires on a code-only pipeline, then compare outputs.
        marker_on, marker_off = tmp_path / "on.log", tmp_path / "off.log"
        head = {"name": "head", "type": "sample", "method": "first", "samples": 2}

        config_off = self.base_config(marker_off, head, tmp_path, "off")
        out_off, _ = DSLRunner({**config_off, "plan_rewrites": False}).run()

        monkeypatch.setattr(
            "docetl.plan.rules.pushdown._chain_has_llm", lambda plan, node: True
        )
        config_on = self.base_config(marker_on, head, tmp_path, "on")
        runner = DSLRunner(config_on)
        assert runner.applied_rewrites, "patched gate should let the rule fire"
        out_on, _ = runner.run()

        assert out_on == out_off
        assert len(marker_off.read_text()) == 6
        assert len(marker_on.read_text()) == 2
