"""Static plan validation: structural integrity, per-op schema checks,
and field-dependency analysis — the runtime failures of
``Node.execute_plan`` surfaced before any execution cost is paid."""

from __future__ import annotations

from typing import Any

from docetl.plan.lift import lift
from docetl.plan.nodes import OpaqueNode
from docetl.plan.plan import LogicalPlan, PlanIssue
from docetl.plan.schema import propagate_schemas


class InvalidCandidatePlan(Exception):
    """A candidate plan failed static validation (used by MOAR to reject
    directive instantiations before executing them)."""

    def __init__(self, source: str, issues: list[PlanIssue]):
        self.source = source
        self.issues = issues
        details = "; ".join(str(i) for i in issues)
        super().__init__(f"invalid plan from {source}: {details}")


def validate(plan: LogicalPlan) -> list[PlanIssue]:
    issues = list(plan.issues)

    seen_configs: set[int] = set()
    for node in plan.nodes():
        if id(node.op_config) in seen_configs:
            continue  # an op referenced from several steps validates once
        seen_configs.add(id(node.op_config))

        if isinstance(node, OpaqueNode):
            issues.append(
                PlanIssue(
                    "info",
                    node.name,
                    f"unknown operation type {node.op_type!r}; treated as opaque "
                    "(conservative traits, never rewritten)",
                )
            )
            continue
        op_class = node.op_class
        if op_class is None:
            continue
        try:
            op_class.schema(**node.op_config)
        except Exception as e:
            issues.append(
                PlanIssue("error", node.name, f"invalid {node.op_type} config: {e}")
            )

    _, removed, schema_issues = propagate_schemas(plan)
    issues.extend(schema_issues)
    for node in plan.nodes():
        if not node.inputs:
            continue
        in_removed = removed.get(id(node.inputs[0]), frozenset())
        reads = node.fields_read
        if reads:
            missing = reads & in_removed
            if missing:
                # Warning, not error: the read-set extractor counts
                # guarded reads ({% if input.x is defined %}, |default)
                # as hard reads, and those templates are runtime-safe.
                # A check with known false positives must not reject
                # candidate plans.
                issues.append(
                    PlanIssue(
                        "warning",
                        node.name,
                        f"reads field(s) {sorted(missing)} that were removed upstream",
                    )
                )
    return issues


def validate_config(config: dict[str, Any]) -> list[PlanIssue]:
    """Lift + validate, with lift problems reported as issues, not raises."""
    try:
        plan = lift(config)
    except Exception as e:  # lift shouldn't raise; belt and suspenders
        return [PlanIssue("error", "<lift>", f"could not build plan: {e}")]
    return validate(plan)
