"""Equivalence-preserving rewrite engine.

Rules implement ``find`` (locate one applicable site) and ``apply_at``
(perform the graph surgery); ``apply_rules`` runs them to fixpoint.
``apply_rewrites_to_config`` is the config-level entry point with the
zero-hash-churn guarantee: when no rule fires, the *original config
object* is returned, so checkpoint hashing and Pipeline state see
literally nothing new.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from docetl.plan.lift import lift
from docetl.plan.lower import lower
from docetl.plan.nodes import PlanNode
from docetl.plan.plan import LogicalPlan


@dataclass(frozen=True)
class AppliedRewrite:
    rule: str
    description: str

    def __str__(self) -> str:
        return f"{self.rule}: {self.description}"


@runtime_checkable
class RewriteRule(Protocol):
    name: str

    def find(self, plan: LogicalPlan) -> PlanNode | None: ...

    def apply_at(self, plan: LogicalPlan, node: PlanNode) -> AppliedRewrite: ...


def push_below(plan: LogicalPlan, node: PlanNode, upstream: PlanNode) -> None:
    """Move *node* to execute immediately before its single input
    *upstream*, maintaining step membership and dropping emptied steps."""
    assert len(node.inputs) == 1 and node.inputs[0] is upstream

    for consumer in plan.consumers(node):
        consumer.inputs = [upstream if i is node else i for i in consumer.inputs]
    node.inputs = list(upstream.inputs)
    upstream.inputs = [node]

    src = plan.step_of(node)
    dst = plan.step_of(upstream)
    assert src is not None and dst is not None
    src.nodes.remove(node)
    dst.nodes.insert(next(i for i, n in enumerate(dst.nodes) if n is upstream), node)
    node.step_name = dst.name
    if not src.nodes:
        plan.drop_step(src)


def default_rules() -> list[RewriteRule]:
    from docetl.plan.rules.pushdown import LimitPushdown, SelectionPushdown

    return [SelectionPushdown(), LimitPushdown()]


def resolve_rules(spec: Any) -> list[RewriteRule]:
    """Resolve a ``plan_rewrites`` setting value into rule instances:
    True/None → all default rules; a list → that subset by name."""
    rules = default_rules()
    if spec is None or spec is True:
        return rules
    if isinstance(spec, (list, tuple, set)):
        wanted = set(spec)
        return [r for r in rules if r.name in wanted]
    return []


def apply_rules(
    plan: LogicalPlan,
    rules: list[RewriteRule] | None = None,
    max_passes: int = 20,
) -> list[AppliedRewrite]:
    """Apply *rules* to fixpoint (bounded by *max_passes* applications),
    mutating *plan* in place. Returns what fired, in order."""
    if rules is None:
        rules = default_rules()
    applied: list[AppliedRewrite] = []
    for _ in range(max_passes):
        fired = False
        for rule in rules:
            site = rule.find(plan)
            if site is not None:
                applied.append(rule.apply_at(plan, site))
                fired = True
                break
        if not fired:
            break
    return applied


def apply_rewrites_to_config(
    config: dict[str, Any], rules: list[RewriteRule] | None = None
) -> tuple[dict[str, Any], list[AppliedRewrite]]:
    """Rewrite a pipeline config. Identity short-circuit: returns the
    original object (and ``[]``) when nothing fires or the config has
    structural problems we won't rewrite through."""
    plan = lift(config)
    if any(issue.level == "error" for issue in plan.issues):
        return config, []
    applied = apply_rules(plan, rules=rules)
    if not applied:
        return config, []
    return lower(plan), applied
