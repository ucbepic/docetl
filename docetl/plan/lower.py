"""Lower a LogicalPlan back to a pipeline config dict.

The byte-stability contract: ``lower(lift(c))`` is deep-equal to ``c``
(and yaml-dump string-equal), and when nothing in the plan changed the
*original objects* are reused — so an untouched plan lowers to the very
same config object graph and checkpoint hashes cannot churn.

Steps are emitted from their verbatim original dicts, with ``operations``
and ``input`` overwritten only when membership actually changed. Op
configs are emitted in the original flat-list order regardless of step
moves (execution order lives in the steps), so rewrites that only move
ops between steps never reorder the ops list.
"""

from __future__ import annotations

from typing import Any

from docetl.plan.nodes import JoinNode
from docetl.plan.plan import LogicalPlan, StepGroup


def _step_entries(step: StepGroup) -> tuple[list[Any], str | None]:
    if step.is_join_headed:
        join = step.nodes[0]
        assert isinstance(join, JoinNode)
        # Rebuild from the verbatim original inner dict so unknown keys
        # survive; dict-unpacking updates left/right in their original
        # key positions.
        inner = {**join.entry_config, "left": join.left_ref, "right": join.right_ref}
        entries: list[Any] = [{join.name: inner}]
        entries.extend(n.name for n in step.nodes[1:])
        return entries, None
    return [n.name for n in step.nodes], step.input_ref


def lower(plan: LogicalPlan) -> dict[str, Any]:
    config = plan.config
    orig_steps = (config.get("pipeline") or {}).get("steps") or []

    new_steps: list[dict[str, Any]] = []
    for step in plan.steps:
        entries, input_ref = _step_entries(step)
        if entries == step.original.get(
            "operations"
        ) and input_ref == step.original.get("input"):
            new_steps.append(step.original)
            continue
        emitted = dict(step.original)  # shallow copy keeps unknown keys & key order
        emitted["operations"] = entries
        if input_ref is None:
            emitted.pop("input", None)
        else:
            emitted["input"] = input_ref
        new_steps.append(emitted)

    orig_ops = config.get("operations") or []
    new_ops: list[dict[str, Any]] = []
    emitted_names: set[str] = set()
    for op in orig_ops:
        name = op.get("name")
        current = plan.ops_by_name.get(name)
        # Reuse the original object unless a rule changed the config.
        new_ops.append(op if current is None or current == op else current)
        emitted_names.add(name)
    for name, op_config in plan.ops_by_name.items():
        if name not in emitted_names:
            new_ops.append(op_config)

    steps_unchanged = len(new_steps) == len(orig_steps) and all(
        new is old for new, old in zip(new_steps, orig_steps)
    )
    ops_unchanged = len(new_ops) == len(orig_ops) and all(
        new is old for new, old in zip(new_ops, orig_ops)
    )
    if steps_unchanged and ops_unchanged:
        return config

    new_config = dict(config)
    new_config["operations"] = new_ops
    pipeline = dict(config.get("pipeline") or {})
    pipeline["steps"] = new_steps
    new_config["pipeline"] = pipeline
    return new_config
