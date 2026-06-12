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
    # Op types that must be present in a config for this rule to possibly
    # fire — lets apply_rewrites_to_config skip the lift entirely.
    trigger_op_types: frozenset[str]

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
    plan.invalidate()

    src = plan.step_of(node)
    dst = plan.step_of(upstream)
    assert src is not None and dst is not None
    src.nodes.remove(node)
    dst.nodes.insert(next(i for i, n in enumerate(dst.nodes) if n is upstream), node)
    if not src.nodes:
        plan.drop_step(src)


def default_rules() -> list[RewriteRule]:
    # LimitPushdown's exactness assumption (no silent row drops while
    # hopping LLM ops — see its docstring) is accepted as a default:
    # the failure mode only occurs in runs already losing rows to the
    # runtime's silent timeout-drop, so the rewrite adds no new badness.
    # Disable per pipeline with plan_rewrites: ["selection_pushdown"].
    from docetl.plan.rules.pushdown import LimitPushdown, SelectionPushdown

    return [SelectionPushdown(), LimitPushdown()]


def all_rules() -> list[RewriteRule]:
    """Every shipped rule (the registry ``resolve_rules`` selects from)."""
    from docetl.plan.rules.pushdown import LimitPushdown, SelectionPushdown

    return [SelectionPushdown(), LimitPushdown()]


def resolve_rules(spec: Any) -> list[RewriteRule]:
    """Resolve a ``plan_rewrites`` setting value into rule instances.

    True/None → the default rules; a string or list of strings → exactly
    those rules by name (opt-in rules included). Unknown names or other
    spec types raise — a misspelled setting must not silently disable
    the optimizations the user asked for.
    """
    if spec is None or spec is True:
        return default_rules()
    if isinstance(spec, str):
        spec = [spec]
    if isinstance(spec, (list, tuple, set)):
        by_name = {r.name: r for r in all_rules()}
        unknown = [n for n in spec if n not in by_name]
        if unknown:
            raise ValueError(
                f"plan_rewrites names unknown rule(s) {unknown}; "
                f"available rules: {sorted(by_name)}"
            )
        return [by_name[n] for n in spec]
    raise ValueError(
        "plan_rewrites must be true, false, a rule name, or a list of "
        f"rule names; got {spec!r}"
    )


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


def could_fire(config: dict[str, Any], rules: list[RewriteRule]) -> bool:
    """Cheap pre-check: a rule can only fire if one of its trigger op
    types appears in the config. Saves the lift (deepcopies, node
    construction) on every runner startup for pipelines with no
    selection-like ops."""
    present = {op.get("type") for op in config.get("operations") or []}
    return any(present & rule.trigger_op_types for rule in rules)


def apply_rewrites_to_config(
    config: dict[str, Any], rules: list[RewriteRule] | None = None
) -> tuple[dict[str, Any], list[AppliedRewrite]]:
    """Rewrite a pipeline config. Identity short-circuit: returns the
    original object (and ``[]``) when nothing fires or the config has
    structural problems we won't rewrite through."""
    if rules is None:
        rules = default_rules()
    if not rules or not could_fire(config, rules):
        return config, []
    plan = lift(config)
    if any(issue.level == "error" for issue in plan.issues):
        return config, []
    applied = apply_rules(plan, rules=rules)
    if not applied:
        return config, []
    return lower(plan), applied
