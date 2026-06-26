"""Lift a pipeline config dict into a LogicalPlan.

Mirrors the pipeline's step-resolution rules: each step's ``input``
names an earlier step or a dataset (earlier steps win); a step whose
first ``operations`` entry is
a ``{name: {left, right}}`` dict and that has no ``input`` is
equijoin-headed, with each side resolved step-or-dataset.

Never raises on malformed configs — structural problems are recorded as
issues on the returned plan so validation can report them all at once.
"""

from __future__ import annotations

import copy
from typing import Any

from docetl.plan.ir import (
    JoinNode,
    LogicalPlan,
    PlanIssue,
    PlanNode,
    ScanNode,
    StepGroup,
    make_node,
)
from docetl.utils import op_ref_name


def lift(config: dict[str, Any]) -> LogicalPlan:
    issues: list[PlanIssue] = []

    ops_by_name: dict[str, dict[str, Any]] = {}
    for op in config.get("operations") or []:
        name = op.get("name")
        if not name:
            issues.append(
                PlanIssue(
                    "error",
                    "<operations>",
                    f"operation without a name: {op.get('type', '?')}",
                )
            )
            continue
        if name in ops_by_name:
            issues.append(PlanIssue("error", name, "duplicate operation name"))
            continue
        ops_by_name[name] = copy.deepcopy(op)

    datasets = config.get("datasets") or {}
    steps_cfg = (config.get("pipeline") or {}).get("steps") or []

    steps: list[StepGroup] = []
    last_node_by_step: dict[str, PlanNode] = {}
    scans: dict[str, ScanNode] = {}

    def scan(dataset_name: str) -> ScanNode:
        node = scans.get(dataset_name)
        if node is None:
            node = ScanNode(
                name=f"scan_{dataset_name}",
                op_config={
                    "type": "scan",
                    "name": f"scan_{dataset_name}",
                    "dataset_name": dataset_name,
                },
                dataset_name=dataset_name,
            )
            scans[dataset_name] = node
        return node

    def resolve_ref(ref: str, where: str) -> PlanNode:
        if ref in last_node_by_step:
            return last_node_by_step[ref]
        if ref not in datasets:
            issues.append(
                PlanIssue(
                    "error",
                    where,
                    f"input {ref!r} is neither an earlier step nor a dataset",
                )
            )
        return scan(ref)

    def node_for(op_name: str, inputs: list[PlanNode], where: str) -> PlanNode:
        op_config = ops_by_name.get(op_name)
        if op_config is None:
            issues.append(
                PlanIssue(
                    "error",
                    where,
                    f"operation {op_name!r} is not defined in `operations`",
                )
            )
            op_config = {"name": op_name, "type": ""}
        return make_node(op_name, op_config, inputs)

    for idx, step_cfg in enumerate(steps_cfg):
        step_name = step_cfg.get("name") or f"<step {idx}>"
        if not step_cfg.get("name"):
            issues.append(PlanIssue("error", step_name, "step without a name"))
        if step_name in last_node_by_step:
            issues.append(PlanIssue("error", step_name, "duplicate step name"))
        entries = step_cfg.get("operations") or []
        if not entries:
            issues.append(PlanIssue("error", step_name, "step has no operations"))

        input_ref = step_cfg.get("input")
        join_headed = input_ref is None and entries and isinstance(entries[0], dict)

        group = StepGroup(name=step_name, original=step_cfg, input_ref=input_ref)
        upstream: PlanNode | None = None

        rest = entries
        if join_headed:
            entry = entries[0]
            join_name = op_ref_name(entry)
            join_cfg = entry.get(join_name) or {}
            left_ref = join_cfg.get("left", "")
            right_ref = join_cfg.get("right", "")
            left = resolve_ref(left_ref, f"{step_name}/{join_name}")
            right = resolve_ref(right_ref, f"{step_name}/{join_name}")
            join = node_for(join_name, [left, right], f"{step_name}/{join_name}")
            if isinstance(join, JoinNode):
                join.left_ref = left_ref
                join.right_ref = right_ref
                join.entry_config = join_cfg
            else:
                issues.append(
                    PlanIssue(
                        "error",
                        f"{step_name}/{join_name}",
                        f"step-entry join reference must be an equijoin, got {join.op_type!r}",
                    )
                )
            group.nodes.append(join)
            upstream = join
            rest = entries[1:]
        elif input_ref is not None:
            upstream = resolve_ref(input_ref, step_name)
        else:
            issues.append(
                PlanIssue(
                    "warning", step_name, "step has no input; treating as empty scan"
                )
            )
            upstream = scan("__empty__")

        for entry in rest:
            if not isinstance(entry, str):
                issues.append(
                    PlanIssue(
                        "error",
                        step_name,
                        f"operation entry {entry!r} should be a string (equijoin entries must come first, with no step input)",
                    )
                )
                continue
            node = node_for(
                entry, [upstream] if upstream else [], f"{step_name}/{entry}"
            )
            group.nodes.append(node)
            upstream = node

        steps.append(group)
        if upstream is not None and step_cfg.get("name"):
            last_node_by_step[step_name] = upstream

    return LogicalPlan(
        config=config, steps=steps, ops_by_name=ops_by_name, issues=issues
    )
