"""Model-cascade machinery with statistical guarantees.

Routes each item in a batch to a cheap *proxy* model or an expensive *oracle*
model, while guaranteeing a target accuracy / precision / recall holds with
probability ``1 - delta`` for any finite sample size.

This module delegates to the BARGAIN library (UC Berkeley EPIC lab,
https://github.com/ucbepic/BARGAIN) for the statistical core -- betting
confidence sequences, without-replacement adaptive sampling, and the
threshold searches for the accuracy / precision / recall guarantees. Thin
adapter classes bridge BARGAIN's Proxy/Oracle class interface with DocETL's
pluggable callable-based proxy/oracle pattern, and a unified
``CategoricalCascade`` API lets DocETL operators (filter, map-with-enum,
resolve, equijoin) share a single guarantee-bearing engine.

The engine is intentionally free of any DocETL imports so it can be unit
tested against synthetic proxy/oracle functions without making LLM calls.

Usage
-----
    spec = CascadeSpec(proxy_model="gpt-4o-mini", guarantee="recall",
                       target=0.95, delta=0.05, label_budget=300)
    cascade = CategoricalCascade(spec, proxy_predict, oracle_predict)
    result = cascade.run(items)
    result.labels          # final label per item (oracle where escalated)
    result.escalated       # bool per item: was the oracle used?
    result.stats           # CascadeStats (proxy/oracle calls, threshold, ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Optional

import numpy as np

from BARGAIN.bounds.betting_bounds import (
    test_if_true_mean_is_above_m as _seq_test_above,
    test_if_true_mean_is_below_m as _seq_test_below,
)
from BARGAIN.models.AbstractModels import Oracle as _BargainOracle, Proxy as _BargainProxy
from BARGAIN.process.BARGAIN_A import BARGAIN_A
from BARGAIN.process.BARGAIN_P import BARGAIN_P
from BARGAIN.process.BARGAIN_R import BARGAIN_R

Guarantee = str  # "accuracy" | "precision" | "recall"

ProxyPredict = Callable[[Any], "tuple[Hashable, float]"]
OraclePredict = Callable[[Any], Hashable]


# ---------------------------------------------------------------------------
# Adapter classes: bridge DocETL's callable protocol to BARGAIN's class API.
# ---------------------------------------------------------------------------
class _ProxyAdapter(_BargainProxy):
    """Wraps a ``ProxyPredict`` callable as a BARGAIN :class:`Proxy`."""

    def __init__(
        self,
        predict_fn: ProxyPredict,
        positive_label: Optional[Hashable] = None,
        console=None,
    ):
        super().__init__(verbose=False)
        self._predict_fn = predict_fn
        self._positive_label = positive_label
        self._console = console

    def proxy_func(self, data_record):
        label, score = self._predict_fn(data_record)
        if self._positive_label is not None:
            return (1 if label == self._positive_label else 0), score
        return label, score

    def get_preds_and_scores(self, indxs, data_records):
        uncached = sum(1 for x in indxs if x not in self.preds_dict)
        if self._console and uncached > 0:
            self._console.log(
                f"[dim]Cascade: scoring {uncached} items with proxy...[/dim]"
            )
        return super().get_preds_and_scores(indxs, data_records)

    def n_calls(self) -> int:
        return len(self.preds_dict)


class _OracleAdapter(_BargainOracle):
    """Wraps an ``OraclePredict`` callable as a BARGAIN :class:`Oracle`.

    For precision / recall modes (``positive_label is not None``), labels are
    mapped to binary {0, 1} so BARGAIN's threshold search works correctly.
    """

    def __init__(
        self,
        predict_fn: OraclePredict,
        positive_label: Optional[Hashable] = None,
        console=None,
    ):
        super().__init__(verbose=False)
        self._predict_fn = predict_fn
        self._positive_label = positive_label
        self._console = console

    def oracle_func(self, data_record, proxy_output):
        raw_label = self._predict_fn(data_record)
        if self._positive_label is not None:
            oracle_output = 1 if raw_label == self._positive_label else 0
        else:
            oracle_output = raw_label
        is_correct = oracle_output == proxy_output
        return is_correct, oracle_output

    def get_pred(self, data_records, indxs=None):
        if self._console and len(data_records) > 0:
            uncached = (
                sum(1 for i in range(len(data_records)) if indxs is None or indxs[i] not in self.preds_dict)
                if indxs is not None
                else len(data_records)
            )
            if uncached > 0:
                self._console.log(
                    f"[dim]Cascade: evaluating {uncached} items with oracle...[/dim]"
                )
        return super().get_pred(data_records, indxs)


# ---------------------------------------------------------------------------
# Public spec / result types.
# ---------------------------------------------------------------------------
@dataclass
class CascadeSpec:
    proxy_model: str
    guarantee: Guarantee  # "accuracy" | "precision" | "recall"
    target: float
    delta: float = 0.05
    label_budget: int = 400  # oracle calls for threshold learning (precision/recall)
    positive_label: Hashable = True  # which label counts as "positive" (P/R)
    negative_label: Hashable = False  # label assigned to non-positive items (P/R)
    n_thresholds: int = 20  # candidate thresholds considered (accuracy/precision)
    seed: Optional[int] = 0


@dataclass
class CascadeStats:
    n_items: int
    proxy_calls: int
    oracle_calls: int
    escalation_rate: float
    guarantee: Guarantee
    target: float
    delta: float


@dataclass
class CascadeResult:
    labels: list  # final label per item, in input order
    escalated: list  # bool per item: was the oracle used?
    stats: CascadeStats
    positive_indices: list = field(default_factory=list)


class GuaranteeNotSupportedError(ValueError):
    pass


class CategoricalCascade:
    """Guarantee-bearing proxy/oracle cascade over a list of items whose LLM
    output is a single categorical label.

    See module docstring for the proxy/oracle callable contract.
    """

    def __init__(
        self,
        spec: CascadeSpec,
        proxy_predict: ProxyPredict,
        oracle_predict: OraclePredict,
        *,
        console=None,
    ):
        if spec.guarantee not in ("accuracy", "precision", "recall"):
            raise GuaranteeNotSupportedError(
                f"unknown guarantee {spec.guarantee!r}; "
                "expected 'accuracy', 'precision' or 'recall'"
            )
        if not (0 < spec.target < 1):
            raise ValueError("target must be in (0, 1)")
        if not (0 < spec.delta < 1):
            raise ValueError("delta must be in (0, 1)")
        self.spec = spec
        self._proxy_predict = proxy_predict
        self._oracle_predict = oracle_predict
        self._console = console

    def run(self, items: list) -> CascadeResult:
        spec = self.spec
        if spec.seed is not None:
            np.random.seed(spec.seed)

        if len(items) == 0:
            return CascadeResult(
                labels=[],
                escalated=[],
                stats=CascadeStats(0, 0, 0, 0.0, spec.guarantee, spec.target, spec.delta),
                positive_indices=[],
            )

        positive_label = (
            spec.positive_label if spec.guarantee in ("precision", "recall") else None
        )

        proxy = _ProxyAdapter(
            self._proxy_predict, positive_label=positive_label, console=self._console
        )
        oracle = _OracleAdapter(
            self._oracle_predict, positive_label=positive_label, console=self._console
        )

        if self._console:
            self._console.log(
                f"[bold]Cascade[/bold] ({spec.guarantee}): "
                f"target={spec.target}, delta={spec.delta}, "
                f"{len(items)} items"
            )

        if spec.guarantee == "accuracy":
            return self._run_accuracy(items, proxy, oracle)
        if spec.guarantee == "precision":
            return self._run_precision(items, proxy, oracle)
        return self._run_recall(items, proxy, oracle)

    def _make_stats(self, n_items: int, proxy, oracle) -> CascadeStats:
        oracle_calls = oracle.get_number_preds()
        return CascadeStats(
            n_items=n_items,
            proxy_calls=proxy.n_calls(),
            oracle_calls=oracle_calls,
            escalation_rate=oracle_calls / n_items if n_items else 0.0,
            guarantee=self.spec.guarantee,
            target=self.spec.target,
            delta=self.spec.delta,
        )

    # ------------------------------------------------------------------
    # Accuracy guarantee via BARGAIN_A
    # ------------------------------------------------------------------
    def _run_accuracy(self, items, proxy, oracle) -> CascadeResult:
        bargain = BARGAIN_A(
            proxy,
            oracle,
            target=self.spec.target,
            delta=self.spec.delta,
            M=self.spec.n_thresholds,
            verbose=False,
            seed=None,
        )

        if self._console:
            self._console.log("[dim]Cascade: determining confidence threshold...[/dim]")

        labels, used_oracle = bargain.process(items, return_oracle_usage=True)

        if self._console:
            n_oracle = sum(used_oracle)
            self._console.log(
                f"[dim]Cascade: threshold found — {n_oracle}/{len(items)} "
                f"items escalated to oracle[/dim]"
            )

        return CascadeResult(
            labels=labels,
            escalated=used_oracle,
            stats=self._make_stats(len(items), proxy, oracle),
        )

    # ------------------------------------------------------------------
    # Precision guarantee via BARGAIN_P
    # ------------------------------------------------------------------
    def _run_precision(self, items, proxy, oracle) -> CascadeResult:
        bargain = BARGAIN_P(
            proxy,
            oracle,
            delta=self.spec.delta,
            target=self.spec.target,
            budget=self.spec.label_budget,
            M=self.spec.n_thresholds,
            eta=0,
            seed=None,
        )

        if self._console:
            self._console.log("[dim]Cascade: determining precision threshold...[/dim]")

        positive_indices = bargain.process(items)
        positive_set = set(int(i) for i in positive_indices)

        if self._console:
            self._console.log(
                f"[dim]Cascade: {len(positive_set)} items classified as positive[/dim]"
            )

        result_labels = [
            self.spec.positive_label if i in positive_set else self.spec.negative_label
            for i in range(len(items))
        ]
        escalated = [i in oracle.preds_dict for i in range(len(items))]
        return CascadeResult(
            labels=result_labels,
            escalated=escalated,
            stats=self._make_stats(len(items), proxy, oracle),
            positive_indices=sorted(positive_set),
        )

    # ------------------------------------------------------------------
    # Recall guarantee via BARGAIN_R (beta=0 uniform path)
    # ------------------------------------------------------------------
    def _run_recall(self, items, proxy, oracle) -> CascadeResult:
        bargain = BARGAIN_R(
            proxy,
            oracle,
            delta=self.spec.delta,
            target=self.spec.target,
            budget=self.spec.label_budget,
            beta=0,
            seed=None,
        )

        if self._console:
            self._console.log("[dim]Cascade: determining recall threshold...[/dim]")

        positive_indices = bargain.process(items)
        positive_set = set(int(i) for i in positive_indices)

        if self._console:
            self._console.log(
                f"[dim]Cascade: {len(positive_set)} items classified as positive[/dim]"
            )

        result_labels = [
            self.spec.positive_label if i in positive_set else self.spec.negative_label
            for i in range(len(items))
        ]
        escalated = [i in oracle.preds_dict for i in range(len(items))]
        return CascadeResult(
            labels=result_labels,
            escalated=escalated,
            stats=self._make_stats(len(items), proxy, oracle),
            positive_indices=sorted(positive_set),
        )
