"""Classical pushdown rules over the LLM-op plan.

Both rules move a selection-like node earlier so upstream LLM ops run on
fewer rows. Every condition here is required for equivalence; the traits
they consult fail closed, so an op the analysis can't see through simply
never qualifies.
"""

from __future__ import annotations

from docetl.operations.base import Cardinality
from docetl.plan.nodes import JoinNode, PlanNode, ScanNode
from docetl.plan.plan import LogicalPlan
from docetl.plan.rewrite import AppliedRewrite, push_below


def _transparent_hop(plan: LogicalPlan, node: PlanNode, upstream: PlanNode) -> bool:
    """Whether *node* (whose single input is *upstream*) could hop over
    *upstream* as far as structure is concerned: upstream is a 1:1,
    row-local, order-preserving op consumed only by *node*, and neither
    op's config is shared across step entries (rewiring a shared node
    would silently edit the other occurrence)."""
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
    """The single upstream op *node* could hop over, if any."""
    if len(node.inputs) != 1:
        return None
    upstream = node.inputs[0]
    return upstream if _transparent_hop(plan, node, upstream) else None


def _chain_has_llm(plan: LogicalPlan, start: PlanNode) -> bool:
    """Whether *start* or a transparent chain above it contains an LLM op
    — the benefit gate that keeps both rules from churning configs for
    free code-over-code swaps."""
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
    """Push a filter below a 1:1 op that doesn't produce (or disturb)
    anything the filter reads.

    Equivalence argument for swapping ``U; S`` to ``S; U``: U is 1:1 in
    the at-most sense (see ``Cardinality``), row-local, and
    order-preserving, so each row meets S with the same field values
    either way *provided* U writes nothing S reads (cond. R_S ∩ W_U = ∅).
    Row-local drops by U commute with the swap: both orders keep exactly
    the rows that pass S's predicate AND survive U. When S annotates kept
    rows (an LLM filter's _short_explanation), U's inputs gain those
    fields after the swap, so U must provably not read or overwrite them.
    An LLM filter qualifies: surviving rows render identical prompts in
    both orders, so its decisions (and cache hits) are unchanged — only U
    now runs on fewer rows.
    """

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
    1:1, row-local, order-preserving op: the first N outputs of such an
    op are exactly its outputs on the first N inputs. Stratified samples
    are excluded (already non-order-preserving by trait), and ``uniform``
    is excluded outright — a different draw without a fixed seam.

    NOT a default rule. Unlike SelectionPushdown, this rewrite is count-
    and position-sensitive, and ONE_TO_ONE is only an at-most contract:
    an LLM op can silently drop a row on an exhausted timeout, in which
    case head-then-op yields N-1 rows on a different row set than
    op-then-head's N. Since hopping over LLM ops is also this rule's only
    benefit case (the gate below requires one), enabling it means
    accepting that failure-free assumption: opt in via
    ``plan_rewrites: ["selection_pushdown", "limit_pushdown"]``. The
    fully sound version of this optimization is fusing the head into the
    scan's load ``limit`` so the file read itself stops early — future
    work at the scan boundary, not a node swap.
    """

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
