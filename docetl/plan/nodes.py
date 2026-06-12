"""Typed nodes of the logical-plan IR.

Every node wraps the original op config dict (``op_config``) as the
lossless source of truth — unknown keys, plugin extensions, and
Frame-injected callables ride along untouched. Operator semantics come
from the trait classmethods on the operation classes (see
``docetl.operations.base.BaseOperation``), resolved per node via the
operation registry so entry-point plugins participate too. There is
deliberately no per-op-type taxonomy here: rules and validation dispatch
on traits (and ``op_type`` where an op's identity genuinely matters), so
adding an operation means writing its traits and nothing else. The only
subclasses carry structure the base node can't: ScanNode (dataset
sources), JoinNode (two inputs), OpaqueNode (unresolvable type).
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

    ``inputs`` are upstream producers (cross-step edges included); step
    membership lives in ``StepGroup.nodes`` (see ``LogicalPlan.step_of``).
    Nodes compare by identity — the same op config referenced from two
    steps becomes two nodes sharing one ``op_config`` dict.
    """

    name: str
    op_config: dict[str, Any]
    inputs: list["PlanNode"] = field(default_factory=list)

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
    def fields_removed(self) -> frozenset[str]:
        cls = self.op_class
        return cls.fields_removed(self.op_config) if cls else frozenset()

    @property
    def is_llm(self) -> bool:
        cls = self.op_class
        return cls.is_llm(self.op_config) if cls else False

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
class JoinNode(PlanNode):
    """Two-input equijoin. ``inputs`` is [left, right]. ``entry_config``
    is the verbatim inner dict of the original ``{name: {left, right,
    ...}}`` step entry, so lower can regenerate it without dropping
    unknown keys."""

    left_ref: str = ""
    right_ref: str = ""
    entry_config: dict[str, Any] = field(default_factory=dict)


@dataclass(eq=False, repr=False)
class OpaqueNode(PlanNode):
    """An op type the registry can't resolve. Conservative traits apply,
    so rewrite rules never touch it."""


def make_node(
    name: str,
    op_config: dict[str, Any],
    inputs: list[PlanNode],
) -> PlanNode:
    op_type = op_config.get("type", "")
    if op_type == "equijoin":
        return JoinNode(name=name, op_config=op_config, inputs=inputs)
    node_cls = PlanNode if _resolve_op_class(op_type) else OpaqueNode
    return node_cls(name=name, op_config=op_config, inputs=inputs)
