"""Typed nodes of the logical-plan IR.

Every node wraps the original op config dict (``op_config``) as the
lossless source of truth — unknown keys, plugin extensions, and
Frame-injected callables ride along untouched. Subclasses exist purely
for typed pattern-matching in validation and rewrite rules; operator
semantics come from the trait classmethods on the operation classes
(see ``docetl.operations.base.BaseOperation``), resolved per node via
the operation registry so entry-point plugins participate too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from docetl.operations.base import Cardinality


def _resolve_op_class(op_type: str):
    from docetl.operations import get_operation

    try:
        return get_operation(op_type)
    except (KeyError, ValueError):
        return None


@dataclass(eq=False)
class PlanNode:
    """One operation in the plan DAG.

    ``inputs`` are upstream producers (cross-step edges included);
    ``step_name`` is the pipeline step this node currently belongs to.
    Nodes compare by identity — the same op config referenced from two
    steps becomes two nodes sharing one ``op_config`` dict.
    """

    name: str
    op_config: dict[str, Any]
    inputs: list["PlanNode"] = field(default_factory=list)
    step_name: str | None = None

    @property
    def op_type(self) -> str:
        return self.op_config.get("type", "")

    @property
    def op_class(self):
        return _resolve_op_class(self.op_type)

    # Trait accessors delegate to the operation class; unknown op types
    # get the conservative BaseOperation defaults (never rewritable).

    @property
    def cardinality(self) -> Cardinality:
        cls = self.op_class
        return cls.cardinality(self.op_config) if cls else Cardinality.MANY_TO_MANY

    @property
    def fields_read(self) -> frozenset[str] | None:
        cls = self.op_class
        return cls.fields_read(self.op_config) if cls else None

    @property
    def fields_written(self) -> frozenset[str] | None:
        cls = self.op_class
        return cls.fields_written(self.op_config) if cls else None

    @property
    def is_llm(self) -> bool:
        cls = self.op_class
        return cls.is_llm(self.op_config) if cls else False

    @property
    def is_deterministic(self) -> bool:
        cls = self.op_class
        return cls.is_deterministic(self.op_config) if cls else False

    @property
    def is_row_local(self) -> bool:
        cls = self.op_class
        return cls.is_row_local(self.op_config) if cls else False

    @property
    def preserves_order(self) -> bool:
        cls = self.op_class
        return cls.preserves_order(self.op_config) if cls else False

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


@dataclass(eq=False, repr=False)
class ScanNode(PlanNode):
    """Reads a named dataset. Synthesized by lift; never lowered to an
    op config (steps reference datasets via their ``input`` key)."""

    dataset_name: str = ""


@dataclass(eq=False, repr=False)
class ProjectionNode(PlanNode):
    """Row-wise transforms: map, parallel_map, code_map, extract, ..."""


@dataclass(eq=False, repr=False)
class SelectionNode(PlanNode):
    """Row subsetting: filter, code_filter, sample, topk."""


@dataclass(eq=False, repr=False)
class AggregateNode(PlanNode):
    """Group-by aggregation: reduce, code_reduce."""


@dataclass(eq=False, repr=False)
class ExpandNode(PlanNode):
    """Row reshaping/expansion: split, unnest, unnest_columns."""


@dataclass(eq=False, repr=False)
class ResolveNode(PlanNode):
    """Entity resolution (similarity self-join)."""


@dataclass(eq=False, repr=False)
class JoinNode(PlanNode):
    """Two-input equijoin. ``inputs`` is [left, right]; the original
    step-entry references are kept so lower can regenerate the
    ``{name: {left, right}}`` entry exactly."""

    left_ref: str = ""
    right_ref: str = ""


@dataclass(eq=False, repr=False)
class OpaqueNode(PlanNode):
    """An op type the registry can't resolve. Conservative traits apply,
    so rewrite rules never touch it."""


NODE_FAMILY: dict[str, type[PlanNode]] = {
    "map": ProjectionNode,
    "parallel_map": ProjectionNode,
    "code_map": ProjectionNode,
    "extract": ProjectionNode,
    "add_uuid": ProjectionNode,
    "gather": ProjectionNode,
    "cluster": ProjectionNode,
    "rank": ProjectionNode,
    "filter": SelectionNode,
    "code_filter": SelectionNode,
    "sample": SelectionNode,
    "topk": SelectionNode,
    "reduce": AggregateNode,
    "code_reduce": AggregateNode,
    "split": ExpandNode,
    "unnest": ExpandNode,
    "unnest_columns": ExpandNode,
    "resolve": ResolveNode,
    "equijoin": JoinNode,
}


def make_node(
    name: str,
    op_config: dict[str, Any],
    inputs: list[PlanNode],
    step_name: str | None,
) -> PlanNode:
    op_type = op_config.get("type", "")
    node_cls = NODE_FAMILY.get(op_type)
    if node_cls is None:
        # Registered-but-unclassified types (web_fetch, link_resolve, ...)
        # get the base node; unresolvable types get OpaqueNode.
        node_cls = PlanNode if _resolve_op_class(op_type) else OpaqueNode
    return node_cls(name=name, op_config=op_config, inputs=inputs, step_name=step_name)
