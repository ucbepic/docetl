"""Classical pushdown rules over the LLM-op plan.

Both rules move a selection-like node earlier so upstream LLM ops run on
fewer rows — pure cost wins with unchanged output. Every condition here
is required for equivalence; the traits they consult fail closed, so an
op the analysis can't see through simply never qualifies.
"""

from __future__ import annotations

from docetl.operations.base import Cardinality
from docetl.plan.nodes import JoinNode, PlanNode, ScanNode
from docetl.plan.plan import LogicalPlan
from docetl.plan.rewrite import AppliedRewrite, push_below


def _swappable_upstream(plan: LogicalPlan, node: PlanNode) -> PlanNode | None:
    """The single upstream op *node* could hop over, if structure allows:
    a 1:1, row-local, order-preserving op consumed only by *node*, with
    neither op's config shared across step entries."""
    if len(node.inputs) != 1:
        return None
    upstream = node.inputs[0]
    if isinstance(upstream, (ScanNode, JoinNode)):
        return None
    if upstream.cardinality != Cardinality.ONE_TO_ONE:
        return None
    if not (upstream.is_row_local and upstream.preserves_order):
        return None
    if plan.consumers(upstream) != [node]:
        return None
    if plan.references(upstream.name) != 1 or plan.references(node.name) != 1:
        return None
    return upstream


def _chain_has_llm(plan: LogicalPlan, start: PlanNode) -> bool:
    """Whether *start* or a 1:1 row-local order-preserving chain above it
    contains an LLM op — the benefit gate that keeps both rules from
    churning configs for free code-over-code swaps."""
    node = start
    while True:
        if node.is_llm:
            return True
        if (
            node.cardinality != Cardinality.ONE_TO_ONE
            or not node.is_row_local
            or not node.preserves_order
            or len(node.inputs) != 1
        ):
            return False
        upstream = node.inputs[0]
        if isinstance(upstream, (ScanNode, JoinNode)):
            return False
        if plan.consumers(upstream) != [node]:
            return False
        node = upstream


class SelectionPushdown:
    """Push a filter below a 1:1 op that doesn't produce (or disturb)
    anything the filter reads.

    Equivalence argument for swapping ``U; S`` to ``S; U``: U is 1:1,
    row-local, and order-preserving, so each row meets S with the same
    field values either way *provided* U writes nothing S reads (cond.
    R_S ∩ W_U = ∅). When S annotates kept rows (an LLM filter's
    _short_explanation), U's inputs gain those fields after the swap, so
    U must provably not read or overwrite them. An LLM filter qualifies:
    surviving rows render identical prompts in both orders, so its
    decisions (and cache hits) are unchanged — only U now runs on fewer
    rows.
    """

    name = "selection_pushdown"

    def find(self, plan: LogicalPlan) -> PlanNode | None:
        for node in plan.nodes():
            if node.op_type not in ("filter", "code_filter"):
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
    1:1, row-local, order-preserving op: the first N outputs of such an
    op are exactly its outputs on the first N inputs. Stratified samples
    are excluded (already non-order-preserving by trait), and ``uniform``
    is excluded outright — a different draw without a fixed seam.
    """

    name = "limit_pushdown"

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
