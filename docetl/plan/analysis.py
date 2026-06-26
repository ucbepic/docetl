"""Schema propagation and static validation over a LogicalPlan."""

from __future__ import annotations

from typing import Any

from docetl.plan.ir import (
    JoinNode,
    LogicalPlan,
    OpaqueNode,
    PlanIssue,
    PlanNode,
    ScanNode,
)

# ── Schema propagation ────────────────────────────────────────────────


def propagate_schemas(
    plan: LogicalPlan,
) -> tuple[dict[int, dict[str, str]], dict[int, frozenset[str]], list[PlanIssue]]:
    """Per-node output schema and definitely-removed field set, keyed by
    ``id(node)``, plus any issues found (join key collisions)."""
    schemas: dict[int, dict[str, str]] = {}
    removed: dict[int, frozenset[str]] = {}
    issues: list[PlanIssue] = []

    def visit(node: PlanNode) -> None:
        if id(node) in schemas:
            return
        for upstream in node.inputs:
            visit(upstream)

        if isinstance(node, ScanNode):
            schemas[id(node)] = {}
            removed[id(node)] = frozenset()
            return

        if isinstance(node, JoinNode) and len(node.inputs) == 2:
            left, right = node.inputs
            in_schema = dict(schemas[id(left)])
            for key, value in schemas[id(right)].items():
                if key in in_schema and in_schema[key] != value:
                    issues.append(
                        PlanIssue(
                            "warning",
                            node.name,
                            f"join sides both produce {key!r} with different types",
                        )
                    )
                in_schema[key] = value
            in_removed = removed[id(left)] & removed[id(right)]
        elif node.inputs:
            in_schema = schemas[id(node.inputs[0])]
            in_removed = removed[id(node.inputs[0])]
        else:
            in_schema, in_removed = {}, frozenset()

        op_class = node.op_class
        if op_class is not None:
            try:
                out_schema = op_class.transform_schema(in_schema, node.op_config)
            except Exception:
                out_schema = dict(in_schema)
        else:
            out_schema = dict(in_schema)

        written = node.fields_written
        if written is None:
            out_removed: frozenset[str] = frozenset()
        else:
            out_removed = (in_removed - written) | node.fields_removed
        schemas[id(node)] = out_schema
        removed[id(node)] = out_removed

    for node in plan.nodes():
        visit(node)
    return schemas, removed, issues


def output_schema(plan: LogicalPlan) -> dict[str, str]:
    """The schema of the plan's final output."""
    root = plan.root
    if root is None:
        return {}
    schemas, _, _ = propagate_schemas(plan)
    return schemas[id(root)]


# ── Validation ────────────────────────────────────────────────────────


class InvalidCandidatePlan(Exception):
    """A candidate plan failed static validation (used by MOAR)."""

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
            continue
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
                issues.append(
                    PlanIssue(
                        "warning",
                        node.name,
                        f"reads field(s) {sorted(missing)} that were removed upstream",
                    )
                )
    return issues


def validate_config(config: dict[str, Any]) -> list[PlanIssue]:
    """Lift + validate."""
    from docetl.plan.lift import lift

    try:
        plan = lift(config)
    except Exception as e:
        return [PlanIssue("error", "<lift>", f"could not build plan: {e}")]
    return validate(plan)
