"""Static output-schema propagation over a LogicalPlan.

Reuses each operation class's ``transform_schema`` (the existing single
source of truth, including the split/unnest/gather/extract/cluster/filter
overrides). Scans start from ``{}`` — the open-world convention shared
with ``Frame.schema()`` — so the only *sound* missing-field error is a
read of a field that was definitely removed upstream; that removal set is
propagated alongside the schemas.
"""

from __future__ import annotations

from docetl.plan.nodes import JoinNode, PlanNode, ScanNode
from docetl.plan.plan import LogicalPlan, PlanIssue

# Definite removals come from the fields_removed trait on the operation
# classes (drop_keys in the base, the decision key on filter, unnest_key
# on unnest_columns) — one description per op, next to its other traits.


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
            # Unknown writes may re-add anything: nothing stays
            # *definitely* removed.
            out_removed: frozenset[str] = frozenset()
        else:
            out_removed = (in_removed - written) | node.fields_removed
        schemas[id(node)] = out_schema
        removed[id(node)] = out_removed

    for node in plan.nodes():
        visit(node)
    return schemas, removed, issues


def output_schema(plan: LogicalPlan) -> dict[str, str]:
    """The schema of the plan's final output (parity with Frame.schema())."""
    root = plan.root
    if root is None:
        return {}
    schemas, _, _ = propagate_schemas(plan)
    return schemas[id(root)]
