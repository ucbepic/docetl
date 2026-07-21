"""
Insertion-only leaderboard of plans for judge-based MOAR evaluation.

``RankedPlans`` maintains a total order over evaluated plans (best to
worst) as a list of capacity-bounded buckets, B+-tree style. A new plan
is routed to a bucket by its 1-5 rating, placed within the bucket by
judge comparisons, and assigned an immutable surrogate score in (0, 1)
strictly between its neighbors' scores.

Bucket capacity is, operationally, the number of already-ranked plans
one batched judge call may cover — by default it is derived per insert
from the judge model's context window and the observed token sizes of
plan outputs (see ``PlanJudge.derive_bucket_capacity``), and buckets
merge/split structurally as it changes.

Two invariants hold across MOAR iterations:

1. The relative order of previously ranked plans never changes — a new
   plan can only be *inserted* between existing ones (and removals just
   drop an entry). Bucket splits are structural only.
2. Scores are immutable once assigned, and strictly decreasing in list
   order — so scores are an order-embedding of the leaderboard and MCTS
   rewards derived from them are ranking-based and stationary.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from docetl.console import DOCETL_CONSOLE
from docetl.moar.judge import PlanJudge, PlanUnits

SCORE_MARGIN_FRACTION = 0.05


@dataclass
class RankedEntry:
    node: Any
    rating: float
    score: float
    units: PlanUnits
    rating_details: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def node_id(self):
        try:
            return self.node.get_id()
        except Exception:
            return getattr(self.node, "id", None)


class RankedPlans:
    """Bucketed, insertion-only ranking of plans judged by a ``PlanJudge``."""

    def __init__(
        self,
        judge: PlanJudge,
        bucket_capacity: Optional[int] = None,
        output_dir: Optional[str] = None,
        console=None,
    ):
        """*bucket_capacity* pins the capacity when given; ``None`` (the
        default) derives it per insert from the judge's context window and
        the observed token sizes of ranked outputs — buckets merge when
        capacity grows and split when it shrinks, both structurally."""
        from docetl.moar.judge import MAX_BUCKET_CAPACITY

        self.judge = judge
        self._static_capacity = max(2, bucket_capacity) if bucket_capacity else None
        self.bucket_capacity = self._static_capacity or MAX_BUCKET_CAPACITY
        self.output_dir = output_dir
        self.console = console if console is not None else DOCETL_CONSOLE

        self.buckets: List[List[RankedEntry]] = []
        self.history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._seq = 0

    # ── public API ─────────────────────────────────────────────────

    def insert(self, node: Any, output_rows: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Judge a plan's outputs and slot it into the leaderboard.

        Returns ``(score, judge_cost)`` where score is the plan's
        surrogate accuracy in (0, 1). Serialized under a lock so
        concurrent simulations cannot interleave insertions.
        """
        with self._lock:
            cost_before = self.judge.total_cost
            order_before = self._order_ids()
            self._seq += 1
            seq = self._seq

            units = self.judge.build_units(output_rows)
            rating = self.judge.rate(units)

            capacity = self._effective_capacity(units)
            self.bucket_capacity = capacity
            bucket_idx, slot, placement_mode = self._locate(
                units, rating.mean, seq, capacity
            )
            score = self._assign_score(bucket_idx, slot, rating.mean)

            entry = RankedEntry(
                node=node,
                rating=rating.mean,
                score=score,
                units=units,
                rating_details=rating.per_unit,
            )
            if not self.buckets:
                self.buckets.append([entry])
            else:
                self.buckets[bucket_idx].insert(slot, entry)
                self._enforce_capacity(capacity)

            cost = self.judge.total_cost - cost_before
            self._record(entry, placement_mode, order_before, cost)
            self._check_invariants(order_before, entry)
            self._save()

            self.console.log(
                f"[bold blue]Judge ranked plan {entry.node_id}:[/bold blue] "
                f"rating {rating.mean:.2f}/5, position "
                f"{self._global_index_of(entry) + 1}/{len(self)}, "
                f"score {score:.4f} (${cost:.4f})"
            )
            return score, cost

    def remove(self, node: Any) -> None:
        """Drop a plan (e.g. a discarded multi-instance candidate).

        Removal cannot reorder the remaining entries.
        """
        with self._lock:
            for bucket in self.buckets:
                for entry in bucket:
                    if entry.node is node:
                        bucket.remove(entry)
                        break
            self.buckets = [b for b in self.buckets if b]
            self._save()

    def __len__(self) -> int:
        return sum(len(b) for b in self.buckets)

    def ordered_entries(self) -> List[RankedEntry]:
        return [entry for bucket in self.buckets for entry in bucket]

    # ── placement ──────────────────────────────────────────────────

    def _effective_capacity(self, new_units: PlanUnits) -> int:
        """Static capacity when pinned, else derived from the judge's
        context window and all observed output sizes (new plan included)."""
        if self._static_capacity is not None:
            return self._static_capacity
        all_units = [e.units for e in self.ordered_entries()] + [new_units]
        per_doc = all(u.aligned for u in all_units)
        return self.judge.derive_bucket_capacity(all_units, per_doc)

    def _locate(
        self, units: PlanUnits, rating: float, seq: int, capacity: int
    ) -> Tuple[int, int, str]:
        """Route by rating, place by comparisons, slide across bucket
        boundaries when the placement lands on an edge."""
        if not self.buckets:
            return 0, 0, "first"

        bucket_idx = self._route_by_rating(rating)
        moved_up = moved_down = False

        for _ in range(len(self.buckets) + 1):
            members = self.buckets[bucket_idx]
            placement = self.judge.place(
                units, [e.units for e in members], seq, max_batch=capacity
            )
            slot = placement.slot

            if slot == 0 and bucket_idx > 0 and not moved_down:
                bucket_idx -= 1
                moved_up = True
                continue
            if (
                slot == len(members)
                and bucket_idx < len(self.buckets) - 1
                and not moved_up
            ):
                bucket_idx += 1
                moved_down = True
                continue
            return bucket_idx, slot, placement.mode

        return bucket_idx, slot, placement.mode

    def _route_by_rating(self, rating: float) -> int:
        """Bucket of the existing entry whose rating is closest to
        *rating* (ties prefer the better-ranked entry)."""
        best_bucket = 0
        best_gap = float("inf")
        for b_idx, bucket in enumerate(self.buckets):
            for entry in bucket:
                gap = abs(entry.rating - rating)
                if gap < best_gap:
                    best_gap = gap
                    best_bucket = b_idx
        return best_bucket

    def _assign_score(self, bucket_idx: int, slot: int, rating: float) -> float:
        """Immutable surrogate score: the normalized rating clamped
        strictly inside the neighbors' score interval."""
        rating_norm = (rating - 1.0) / 4.0
        if not self.buckets:
            return min(max(rating_norm, 0.05), 0.95)

        ordered = self.ordered_entries()
        global_idx = sum(len(b) for b in self.buckets[:bucket_idx]) + slot
        upper = ordered[global_idx - 1].score if global_idx > 0 else 1.0
        lower = ordered[global_idx].score if global_idx < len(ordered) else 0.0

        interval = upper - lower
        if interval <= 1e-12:
            return (upper + lower) / 2.0
        margin = interval * SCORE_MARGIN_FRACTION
        return min(max(rating_norm, lower + margin), upper - margin)

    def _enforce_capacity(self, capacity: int) -> None:
        """Rebalance buckets to the current capacity.

        Merges adjacent buckets that fit together (dynamic mode only, so a
        capacity that grew doesn't leave the leaderboard fragmented) and
        splits any over-capacity bucket into balanced consecutive chunks.
        Both operations preserve the concatenated order — no LLM calls,
        no score changes.
        """
        if self._static_capacity is None:
            merged: List[List[RankedEntry]] = []
            for bucket in self.buckets:
                if merged and len(merged[-1]) + len(bucket) <= capacity:
                    merged[-1] = merged[-1] + bucket
                else:
                    merged.append(list(bucket))
            self.buckets = merged

        rebalanced: List[List[RankedEntry]] = []
        for bucket in self.buckets:
            if len(bucket) <= capacity:
                rebalanced.append(bucket)
            else:
                n_parts = -(-len(bucket) // capacity)
                size = -(-len(bucket) // n_parts)
                rebalanced.extend(
                    bucket[i : i + size] for i in range(0, len(bucket), size)
                )
        self.buckets = rebalanced

    # ── bookkeeping ────────────────────────────────────────────────

    def _order_ids(self) -> List[Any]:
        return [e.node_id for e in self.ordered_entries()]

    def _global_index_of(self, entry: RankedEntry) -> int:
        return self.ordered_entries().index(entry)

    def _check_invariants(
        self, order_before: List[Any], new_entry: RankedEntry
    ) -> None:
        order_after = self._order_ids()
        without_new = [i for i in order_after if i != new_entry.node_id]
        if without_new != order_before:
            self.console.log(
                "[bold red]Judge ranking invariant violated: existing plan "
                f"order changed ({order_before} -> {without_new})[/bold red]"
            )
        scores = [e.score for e in self.ordered_entries()]
        if any(scores[i] <= scores[i + 1] for i in range(len(scores) - 1)):
            self.console.log(
                "[bold red]Judge ranking invariant violated: scores not "
                f"strictly decreasing ({scores})[/bold red]"
            )

    def _record(
        self,
        entry: RankedEntry,
        placement_mode: str,
        order_before: List[Any],
        cost: float,
    ) -> None:
        self.history.append(
            {
                "seq": self._seq,
                "node_id": entry.node_id,
                "rating": entry.rating,
                "rating_details": entry.rating_details,
                "score": entry.score,
                "placement_mode": placement_mode,
                "capacity": self.bucket_capacity,
                "position": self._global_index_of(entry),
                "order_before": order_before,
                "order_after": self._order_ids(),
                "judge_cost": cost,
            }
        )

    def _save(self) -> None:
        if not self.output_dir:
            return
        try:
            payload = {
                "criteria": self.judge.criteria,
                "judge_model": self.judge.judge_model,
                "leaderboard": [
                    {
                        "position": i + 1,
                        "bucket": b_idx,
                        "node_id": entry.node_id,
                        "yaml_path": str(
                            getattr(entry.node, "yaml_file_path", "")
                        ),
                        "rating": entry.rating,
                        "score": entry.score,
                    }
                    for i, (b_idx, entry) in enumerate(
                        (b_idx, entry)
                        for b_idx, bucket in enumerate(self.buckets)
                        for entry in bucket
                    )
                ],
                "history": self.history,
            }
            path = os.path.join(self.output_dir, "ranking.json")
            with open(path, "w") as f:
                json.dump(payload, f, indent=2, default=str)
        except Exception as e:
            self.console.log(
                f"[yellow]Could not save ranking.json: {e}[/yellow]"
            )
