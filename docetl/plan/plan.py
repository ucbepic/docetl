"""The LogicalPlan container: step groups over a DAG of PlanNodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from docetl.plan.nodes import JoinNode, PlanNode


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
    """A lifted pipeline config.

    ``config`` is the original dict (kept by reference; non-step keys are
    reused verbatim at lower time). ``ops_by_name`` maps op name to the
    plan's working copy of that op config — shared by every node that
    references the name.
    """

    config: dict[str, Any]
    steps: list[StepGroup]
    ops_by_name: dict[str, dict[str, Any]]
    issues: list[PlanIssue] = field(default_factory=list)

    @property
    def root(self) -> PlanNode | None:
        for step in reversed(self.steps):
            if step.nodes:
                return step.nodes[-1]
        return None

    def nodes(self) -> Iterator[PlanNode]:
        """All op nodes in execution (topological) order."""
        for step in self.steps:
            yield from step.nodes

    def consumers(self, node: PlanNode) -> list[PlanNode]:
        return [n for n in self.nodes() if any(i is node for i in n.inputs)]

    def references(self, op_name: str) -> int:
        """How many step entries reference this op name. Rules must not
        rewrite an op whose config is shared across entries."""
        return sum(1 for n in self.nodes() if n.name == op_name)

    def step_of(self, node: PlanNode) -> StepGroup | None:
        for step in self.steps:
            if any(n is node for n in step.nodes):
                return step
        return None

    def drop_step(self, step: StepGroup) -> None:
        """Remove an emptied step and rewire references to it (step
        inputs and equijoin left/right) to its own input."""
        assert not step.nodes, "only empty steps can be dropped"
        self.steps.remove(step)
        for other in self.steps:
            if other.input_ref == step.name:
                other.input_ref = step.input_ref
            if other.is_join_headed:
                join = other.nodes[0]
                if join.left_ref == step.name:
                    join.left_ref = step.input_ref
                if join.right_ref == step.name:
                    join.right_ref = step.input_ref
