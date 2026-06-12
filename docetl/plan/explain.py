"""Render a LogicalPlan as a plain indented tree, one line per node —
compact enough to embed in MOAR agent prompts."""

from __future__ import annotations

from docetl.operations.base import Cardinality
from docetl.plan.nodes import JoinNode, PlanNode, ScanNode
from docetl.plan.plan import LogicalPlan
from docetl.plan.schema import propagate_schemas

_CARDINALITY_SYMBOL = {
    Cardinality.ONE_TO_ONE: "1:1",
    Cardinality.SELECTION: "selection",
    Cardinality.MANY_TO_ONE: "N:1",
    Cardinality.ONE_TO_MANY: "1:N",
    Cardinality.MANY_TO_MANY: "M:N",
}


def _node_line(node: PlanNode, schemas: dict[int, dict[str, str]] | None) -> str:
    if isinstance(node, ScanNode):
        return f"scan({node.dataset_name})"
    tags = [node.op_type or "?", _CARDINALITY_SYMBOL[node.cardinality]]
    if node.is_llm:
        tags.append("llm")
    line = f"{node.name} [{' · '.join(tags)}]"
    if schemas is not None and node.inputs:
        in_schema = schemas.get(id(node.inputs[0]), {})
        out_schema = schemas.get(id(node), {})
        added = {
            k: v for k, v in out_schema.items() if in_schema.get(k) != v
        }
        dropped = sorted(k for k in in_schema if k not in out_schema)
        if added:
            line += " +{" + ", ".join(f"{k}: {v}" for k, v in sorted(added.items())) + "}"
        if dropped:
            line += " -{" + ", ".join(dropped) + "}"
    return line


def format_plan(plan: LogicalPlan, schemas: bool = True) -> str:
    """Output-rooted tree: each node's producers are indented beneath it."""
    schema_map = propagate_schemas(plan)[0] if schemas else None
    lines: list[str] = []
    printed: set[int] = set()

    def emit(node: PlanNode, depth: int) -> None:
        prefix = "  " * depth
        if id(node) in printed:
            lines.append(f"{prefix}{node.name} (shown above)")
            return
        printed.add(id(node))
        lines.append(prefix + _node_line(node, schema_map))
        inputs = node.inputs
        if isinstance(node, JoinNode) and len(inputs) == 2:
            lines.append(f"{prefix}  left ({node.left_ref}):")
            emit(inputs[0], depth + 2)
            lines.append(f"{prefix}  right ({node.right_ref}):")
            emit(inputs[1], depth + 2)
        else:
            for upstream in inputs:
                emit(upstream, depth + 1)

    root = plan.root
    if root is None:
        return "(empty plan)"
    emit(root, 0)
    return "\n".join(lines)
