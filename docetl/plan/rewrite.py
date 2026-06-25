"""Equivalence-preserving rewrite engine and pushdown rules.

Rules implement ``find`` (locate one applicable site) and ``apply_at``
(perform the graph surgery); ``apply_rules`` runs them to fixpoint.
``apply_rewrites_to_config`` is the config-level entry point with the
zero-hash-churn guarantee: when no rule fires, the *original config
object* is returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from docetl.operations.base import Cardinality
from docetl.plan.ir import JoinNode, LogicalPlan, PlanNode, ScanNode


@dataclass(frozen=True)
class AppliedRewrite:
    rule: str
    description: str

    def __str__(self) -> str:
        return f"{self.rule}: {self.description}"


@runtime_checkable
class RewriteRule(Protocol):
    name: str
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


# ── Pushdown rules ────────────────────────────────────────────────────


def _transparent_hop(plan: LogicalPlan, node: PlanNode, upstream: PlanNode) -> bool:
    if isinstance(upstream, (ScanNode, JoinNode)):
        return False
    if upstream.cardinality != Cardinality.ONE_TO_ONE:
        return False
    if not (upstream.is_row_local and upstream.preserves_order):
        return False
    if plan.consumers(upstream) != [node]:
        return False
    if plan.references(upstream.name) != 1 or plan.references(node.name) != 1:
        return False
    return True


def _swappable_upstream(plan: LogicalPlan, node: PlanNode) -> PlanNode | None:
    if len(node.inputs) != 1:
        return None
    upstream = node.inputs[0]
    return upstream if _transparent_hop(plan, node, upstream) else None


def _chain_has_llm(plan: LogicalPlan, start: PlanNode) -> bool:
    node = start
    while True:
        if node.is_llm:
            return True
        if len(node.inputs) != 1:
            return False
        upstream = node.inputs[0]
        if not _transparent_hop(plan, node, upstream):
            return False
        node = upstream


class SelectionPushdown:
    """Push a filter below a 1:1 op that doesn't produce anything the
    filter reads."""

    name = "selection_pushdown"
    trigger_op_types = frozenset({"filter", "code_filter"})

    def find(self, plan: LogicalPlan) -> PlanNode | None:
        for node in plan.nodes():
            if node.op_type not in self.trigger_op_types:
                continue
            upstream = _swappable_upstream(plan, node)
            if upstream is None:
                continue
            reads = node.fields_read
            if reads is None:
                continue
            upstream_writes = upstream.fields_written
            if upstream_writes is None or reads & upstream_writes:
                continue
            writes = node.fields_written
            if writes is None:
                continue
            if writes:
                upstream_reads = upstream.fields_read
                if (
                    upstream_reads is None
                    or upstream_reads & writes
                    or upstream_writes & writes
                ):
                    continue
            if not _chain_has_llm(plan, upstream):
                continue
            return node
        return None

    def apply_at(self, plan: LogicalPlan, node: PlanNode) -> AppliedRewrite:
        upstream = node.inputs[0]
        push_below(plan, node, upstream)
        return AppliedRewrite(
            self.name,
            f"pushed {node.name} ({node.op_type}) below {upstream.name} "
            f"({upstream.op_type}), so {upstream.name} runs only on rows "
            f"{node.name} keeps",
        )


class LimitPushdown:
    """Pull a positional head (``sample`` with method "first") below a
    1:1, row-local, order-preserving op."""

    name = "limit_pushdown"
    trigger_op_types = frozenset({"sample"})

    def find(self, plan: LogicalPlan) -> PlanNode | None:
        for node in plan.nodes():
            if node.op_type != "sample":
                continue
            if node.op_config.get("method") != "first" or node.op_config.get(
                "stratify_key"
            ):
                continue
            upstream = _swappable_upstream(plan, node)
            if upstream is None:
                continue
            if not _chain_has_llm(plan, upstream):
                continue
            return node
        return None

    def apply_at(self, plan: LogicalPlan, node: PlanNode) -> AppliedRewrite:
        upstream = node.inputs[0]
        push_below(plan, node, upstream)
        return AppliedRewrite(
            self.name,
            f"pushed {node.name} (first-{node.op_config.get('samples')}) below "
            f"{upstream.name} ({upstream.op_type}), so {upstream.name} runs "
            f"only on the rows that survive the head",
        )


# ── Engine ────────────────────────────────────────────────────────────


def default_rules() -> list[RewriteRule]:
    return [SelectionPushdown(), LimitPushdown()]


def all_rules() -> list[RewriteRule]:
    return [SelectionPushdown(), LimitPushdown()]


def resolve_rules(spec: Any) -> list[RewriteRule]:
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
    present = {op.get("type") for op in config.get("operations") or []}
    return any(present & rule.trigger_op_types for rule in rules)


def apply_rewrites_to_config(
    config: dict[str, Any], rules: list[RewriteRule] | None = None
) -> tuple[dict[str, Any], list[AppliedRewrite]]:
    from docetl.plan.lift import lift
    from docetl.plan.lower import lower

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
