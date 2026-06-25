"""Typed nodes and plan container for the logical-plan IR.

Every node wraps the original op config dict (``op_config``) as the
lossless source of truth. Operator semantics come from the trait
classmethods on the operation classes, resolved per node via the
operation registry so entry-point plugins participate too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from docetl.operations.base import Cardinality

# ── Node types ────────────────────────────────────────────────────────


def _resolve_op_class(op_type: str):
    from docetl.operations import get_operation

    try:
        return get_operation(op_type)
    except (KeyError, ValueError):
        return None


@dataclass(eq=False)
class PlanNode:
    """One operation in the plan DAG."""

    name: str
    op_config: dict[str, Any]
    inputs: list["PlanNode"] = field(default_factory=list)

    @property
    def op_type(self) -> str:
        return self.op_config.get("type", "")

    @property
    def op_class(self):
        return _resolve_op_class(self.op_type)

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
    """Reads a named dataset."""

    dataset_name: str = ""


@dataclass(eq=False, repr=False)
class JoinNode(PlanNode):
    """Two-input equijoin. ``inputs`` is [left, right]."""

    left_ref: str = ""
    right_ref: str = ""
    entry_config: dict[str, Any] = field(default_factory=dict)


@dataclass(eq=False, repr=False)
class OpaqueNode(PlanNode):
    """An op type the registry can't resolve. Conservative traits apply."""


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


# ── Plan container ────────────────────────────────────────────────────


@dataclass
class PlanIssue:
    """A validation finding. ``level`` is "error", "warning", or "info"."""

    level: str
    where: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.where}: {self.message}"


@dataclass(eq=False)
class StepGroup:
    """One pipeline step: its op nodes in execution order, plus the
    verbatim original step dict so lower can pass unknown keys through."""

    name: str
    original: dict[str, Any]
    input_ref: str | None
    nodes: list[PlanNode] = field(default_factory=list)

    @property
    def is_join_headed(self) -> bool:
        return bool(self.nodes) and isinstance(self.nodes[0], JoinNode)


@dataclass(eq=False)
class LogicalPlan:
    """A lifted pipeline config."""

    config: dict[str, Any]
    steps: list[StepGroup]
    ops_by_name: dict[str, dict[str, Any]]
    issues: list[PlanIssue] = field(default_factory=list)
    _consumer_map: dict[int, list[PlanNode]] | None = field(
        default=None, repr=False, compare=False
    )
    _ref_counts: dict[str, int] | None = field(default=None, repr=False, compare=False)

    @property
    def root(self) -> PlanNode | None:
        for step in reversed(self.steps):
            if step.nodes:
                return step.nodes[-1]
        return None

    def nodes(self) -> Iterator[PlanNode]:
        for step in self.steps:
            yield from step.nodes

    def invalidate(self) -> None:
        self._consumer_map = None
        self._ref_counts = None

    def consumers(self, node: PlanNode) -> list[PlanNode]:
        if self._consumer_map is None:
            cmap: dict[int, list[PlanNode]] = {}
            for n in self.nodes():
                for i in n.inputs:
                    cmap.setdefault(id(i), []).append(n)
            self._consumer_map = cmap
        return self._consumer_map.get(id(node), [])

    def references(self, op_name: str) -> int:
        if self._ref_counts is None:
            counts: dict[str, int] = {}
            for n in self.nodes():
                counts[n.name] = counts.get(n.name, 0) + 1
            self._ref_counts = counts
        return self._ref_counts.get(op_name, 0)

    def step_of(self, node: PlanNode) -> StepGroup | None:
        for step in self.steps:
            if any(n is node for n in step.nodes):
                return step
        return None

    def drop_step(self, step: StepGroup) -> None:
        assert not step.nodes, "only empty steps can be dropped"
        self.steps.remove(step)
        self.invalidate()
        for other in self.steps:
            if other.input_ref == step.name:
                other.input_ref = step.input_ref
            if other.is_join_headed:
                join = other.nodes[0]
                if join.left_ref == step.name:
                    join.left_ref = step.input_ref
                if join.right_ref == step.name:
                    join.right_ref = step.input_ref
