"""Shared plumbing for running a binary model cascade inside an operator.

The statistical engine lives in ``cascade.py``; this module provides the
operator-facing pieces reused by filter / resolve / equijoin:

- :class:`CascadeConfig` -- the opt-in ``cascade:`` config block.
- :class:`CascadeMixin` -- builds the proxy/oracle adapters around an
  operator's prompt, runs the engine, and accumulates cost.

An operator supplies only: how to enumerate items, how to render an item's
prompt, the candidate label set, and an oracle callable. The guarantee default
is operator-dependent (filter->recall, resolve/equijoin->precision) and is
passed in at call time. Only binary predictions are supported (accuracy,
precision, recall, precision+recall guarantees via BARGAIN).
"""

import hashlib
import json
from typing import TYPE_CHECKING, Any, Callable, Optional

from pydantic import BaseModel, Field, model_validator

from docetl.operations.utils.cache import cache
from docetl.progress.tracker import active_tracker

if TYPE_CHECKING:
    from docetl.operations.utils.progress import RichLoopBar

_GUARANTEES = ("accuracy", "precision", "recall", "precision+recall")

# Default guarantee when the cascade block omits ``guarantee`` (per operator).
CASCADE_DEFAULT_GUARANTEE: dict[str, str] = {
    "filter": "recall",
    "resolve": "precision",
    "equijoin": "precision",
}


def format_cascade_plan_lines(
    cascade: dict[str, Any],
    *,
    op_type: str,
    oracle_model: str,
) -> list[str]:
    """Rich-markup lines for the cascade block in the query-plan panel."""
    cfg = cascade
    guarantee = cfg.get("guarantee") or CASCADE_DEFAULT_GUARANTEE.get(op_type, "recall")
    target = cfg["target"]
    delta = cfg.get("delta", 0.05)
    budget = cfg.get("label_budget", 400)
    if guarantee in ("recall", "precision", "precision+recall"):
        oracle_hint = f"≤{budget} oracle labels"
    else:
        oracle_hint = "escalated items"
    return [
        "[bold magenta]Cascade[/bold magenta]",
        f"  [dim]proxy[/dim]     [cyan]{cfg['proxy_model']}[/cyan]  [dim]· all items[/dim]",
        f"  [dim]oracle[/dim]    [cyan]{oracle_model}[/cyan]  [dim]· {oracle_hint}[/dim]",
        (
            f"  [dim]guarantee[/dim] [yellow]{guarantee}[/yellow] "
            f"[dim]≥[/dim][yellow]{target:.0%}[/yellow]  [dim]δ={delta}[/dim]"
        ),
    ]


def _build_score_hist(scores: list[float], n_bins: int = 20) -> list[int]:
    """Bucket proxy confidence scores into a fixed-width histogram."""
    hist = [0] * n_bins
    for s in scores:
        b = min(int(s * n_bins), n_bins - 1)
        hist[b] += 1
    return hist


def describe_cascade_stats(info: dict) -> dict[str, str]:
    """Produce human-readable description strings from a cascade info dict.

    Returns a dict with keys:
        oracle_desc:     what the oracle did ("12 calibration + 3 gap-verified = 15 total (budget 300)")
        threshold_desc:  threshold line ("0.847 proxy confidence" or "n/a — proxy not confident enough")
        result_desc:     result/escalation line ("450 proxy-accepted + 50 calibration samples → 500 items")

    Both the Rich console log (_report_cascade) and the TUI detail pane
    (_render_cascade_info) use this so the wording stays in sync.
    """
    guarantee = info.get("guarantee", "")
    is_calibrated = guarantee in ("precision", "recall", "precision+recall")
    oracle_calls = info.get("oracle_calls", 0)
    n_items = info.get("n_items", 0)
    served_by_proxy = info.get("served_by_proxy", n_items - oracle_calls)
    budget = info.get("label_budget", "?")
    cal = info.get("calibration_calls", oracle_calls)
    gap = info.get("gap_verified", 0)
    threshold = info.get("threshold")

    # Oracle description
    if guarantee == "precision+recall" and gap > 0:
        oracle_desc = (
            f"{cal} calibration + {gap} gap-verified "
            f"= {oracle_calls} total (budget {budget})"
        )
    elif is_calibrated:
        oracle_desc = f"{oracle_calls} sampled for calibration (budget {budget})"
    else:
        oracle_desc = f"{oracle_calls} escalated"

    # Threshold description
    if threshold is not None and threshold >= 0.01:
        threshold_desc = f"{threshold:.3f} proxy confidence"
    elif is_calibrated:
        threshold_desc = "n/a — proxy not confident enough"
    else:
        threshold_desc = "n/a"

    # Result description
    if guarantee == "precision+recall" and gap > 0:
        result_desc = (
            f"{served_by_proxy} proxy-accepted + {cal} calibration "
            f"+ {gap} gap-verified → {n_items} items"
        )
    elif is_calibrated:
        result_desc = (
            f"{served_by_proxy} proxy-accepted "
            f"+ {oracle_calls} calibration samples → {n_items} items"
        )
    else:
        esc_rate = info.get("escalation_rate", 0)
        result_desc = (
            f"{esc_rate:.0%} escalation · {served_by_proxy}/{n_items} served by proxy"
        )

    return {
        "oracle_desc": oracle_desc,
        "threshold_desc": threshold_desc,
        "result_desc": result_desc,
    }


class CascadeConfig(BaseModel):
    """Opt-in model-cascade configuration for an operator.

    Runs a cheap ``proxy_model`` on every item, learns a confidence threshold
    on a small oracle-labeled sample, trusts the proxy above it and escalates
    the rest to the operation's existing model (the oracle) -- preserving the
    chosen statistical ``guarantee`` w.p. ``1 - delta``. See
    ``docs/design/model-cascade.md``.

    ``guarantee`` defaults to ``None`` so each operator can apply its own
    natural default when the user omits it.
    """

    proxy_model: str
    guarantee: Optional[str] = None
    target: float = Field(..., gt=0, lt=1)
    delta: float = Field(0.05, gt=0, lt=1)
    label_budget: int = Field(400, gt=0)

    @model_validator(mode="after")
    def _check_guarantee(self):
        if self.guarantee is not None and self.guarantee not in _GUARANTEES:
            raise ValueError(
                f"cascade.guarantee must be one of {_GUARANTEES}; "
                f"got '{self.guarantee}'"
            )
        return self


class _CascadeProgress:
    """Proxy/oracle phase progress for model cascades.

    Each dot in the TUI grid = one input item. The proxy phase fills dots
    as responses arrive (one tick per item scored). When the oracle phase
    starts, the phase label changes but the grid keeps its state — oracle
    calls don't reset the grid. The console path uses separate tqdm bars
    for proxy and oracle since tqdm doesn't support label changes cleanly.
    """

    def __init__(
        self,
        console,
        *,
        n_items: int,
        proxy_model: str,
        oracle_model: str,
        label_budget: int,
        guarantee: str,
        status=None,
        proxy_scores: list[float] | None = None,
    ) -> None:
        import threading
        self._lock = threading.Lock()
        self.console = console
        self.n_items = n_items
        self.proxy_model = proxy_model
        self.oracle_model = oracle_model
        self.label_budget = label_budget
        self.guarantee = guarantee
        self._status = status
        self._proxy_scores = proxy_scores
        self._bar: RichLoopBar | None = None
        self._oracle_started = False
        self._proxy_ticks = 0
        self._tracker = active_tracker()
        if self._tracker is None and self._status is not None:
            self._status.stop()
        self._start_proxy()

    def _start_proxy(self) -> None:
        label = f"proxy ({self.proxy_model})"
        if self._tracker is not None:
            self._tracker.set_phase(self.n_items, label=label)
        else:
            from docetl.operations.utils.progress import RichLoopBar

            self._bar = RichLoopBar(
                total=self.n_items,
                desc=f"Cascade {label}",
                console=self.console,
                leave=False,
            )
            self._bar.__enter__()

    def tick_proxy(self) -> None:
        """Advance the grid by one item as a proxy response arrives."""
        with self._lock:
            if self._oracle_started:
                return
            self._proxy_ticks += 1
            if self._proxy_ticks <= self.n_items:
                self._tick()

    def tick_oracle(self) -> None:
        """Signal an oracle call.

        On first call, freezes the grid (dots stay green from proxy) and
        starts a new phase for the ops-list progress counter. Each
        subsequent call advances the oracle progress.
        """
        with self._lock:
            if not self._oracle_started:
                self._oracle_started = True
                label = f"oracle ({self.oracle_model})"
                oracle_total = min(self.n_items, self.label_budget)
                if self._tracker is not None:
                    self._tracker.freeze_grid()
                    info = {
                        "proxy_model": self.proxy_model,
                        "oracle_model": self.oracle_model,
                        "proxy_calls": self._proxy_ticks,
                        "label_budget": self.label_budget,
                        "guarantee": self.guarantee,
                    }
                    if self._proxy_scores:
                        info["score_hist"] = _build_score_hist(self._proxy_scores)
                        info["item_proxy_scores"] = list(self._proxy_scores)
                    self._tracker.set_cascade_info(info)
                    self._tracker.set_phase(oracle_total, label=label)
                else:
                    self._close_bar()
                    from docetl.operations.utils.progress import RichLoopBar

                    self._bar = RichLoopBar(
                        total=oracle_total,
                        desc=f"Cascade {label}",
                        console=self.console,
                        leave=False,
                    )
                    self._bar.__enter__()
            self._tick()

    def _tick(self) -> None:
        if self._tracker is not None:
            self._tracker.tick()
        elif self._bar is not None:
            self._bar.update()

    def _close_bar(self) -> None:
        if self._bar is not None:
            self._bar.__exit__(None, None, None)
            self._bar = None

    def finish(self) -> None:
        if self._tracker is not None:
            self._tracker.clear_phase()
        self._close_bar()
        if self._tracker is None and self._status is not None:
            self._status.start()


class CascadeMixin:
    """Mixin giving an operation a ``_run_binary_cascade`` helper.

    Relies on attributes provided by ``BaseOperation``: ``self.config``,
    ``self.runner`` (with ``.api``), and ``self.console``.
    """

    def _oracle_model_name(self) -> str:
        """The oracle model name, checking comparison_model (resolve/equijoin) first."""
        return self.config.get(
            "comparison_model",
            self.config.get("model", getattr(self, "default_model", "?")),
        )

    def _cascade_cfg(self) -> dict[str, Any]:
        """The cascade block as a plain dict (it may arrive as a dict or a
        validated :class:`CascadeConfig`)."""
        cfg = self.config["cascade"]
        if isinstance(cfg, BaseModel):
            return cfg.model_dump()
        return cfg

    def _cascade_cache_key(
        self,
        items: list,
        proxy_labels: list,
        guarantee: str,
        positive_label: Any,
        negative_label: Any,
    ) -> str:
        """Stable key over op identity + config + dataset signature.

        Calibration (threshold learning) and labeling are deterministic given
        the same operation config and items, so an identical re-run can reuse
        the cached result without re-paying the proxy/oracle calls.
        """
        material = {
            "v": 1,
            "name": self.config.get("name"),
            "cascade": self._cascade_cfg(),
            "guarantee": guarantee,
            "labels": [str(x) for x in proxy_labels],
            "positive": str(positive_label),
            "negative": str(negative_label),
            "prompt": self.config.get("prompt"),
            "comparison_prompt": self.config.get("comparison_prompt"),
            "model": self.config.get("model"),
            "comparison_model": self.config.get("comparison_model"),
            "output": self.config.get("output"),
            "items": items,
        }
        blob = json.dumps(material, sort_keys=True, default=str)
        return "cascade:" + hashlib.sha256(blob.encode()).hexdigest()

    def _report_cascade(
        self,
        op_label: str,
        stats,
        cost: float,
        cached_hit: bool,
        *,
        proxy_cost: float = 0.0,
        oracle_cost: float = 0.0,
        proxy_scores: list[float] | None = None,
        escalated: list[bool] | None = None,
        proxy_labels: list | None = None,
    ) -> None:
        """Log a cost/escalation summary and stash stats for programmatic use."""
        self.cascade_stats = stats
        served_by_proxy = stats.n_items - stats.oracle_calls
        tag = " [dim](cached)[/dim]" if cached_hit else ""

        cfg = self._cascade_cfg()
        proxy_model = cfg["proxy_model"]
        oracle_model = self._oracle_model_name()

        score_hist = _build_score_hist(proxy_scores) if proxy_scores else None

        tracker = active_tracker()
        if tracker is not None:
            info = {
                "proxy_model": proxy_model,
                "oracle_model": oracle_model,
                "guarantee": stats.guarantee,
                "target": stats.target,
                "delta": stats.delta,
                "label_budget": stats.label_budget,
                "proxy_calls": stats.proxy_calls,
                "oracle_calls": stats.oracle_calls,
                "escalation_rate": stats.escalation_rate,
                "served_by_proxy": served_by_proxy,
                "proxy_cost": proxy_cost,
                "oracle_cost": oracle_cost,
                "threshold": stats.threshold,
                "score_hist": score_hist,
                "cached": cached_hit,
                "calibration_calls": stats.calibration_calls,
                "gap_verified": stats.gap_verified,
            }
            if escalated is not None:
                info["item_escalated"] = escalated
            if proxy_scores is not None:
                info["item_proxy_scores"] = proxy_scores
            if proxy_labels is not None:
                info["item_proxy_labels"] = proxy_labels
            info["is_binary"] = True
            tracker.set_cascade_info(info)

        name = self.config.get("name", "?")
        target_pct = f"{stats.target:.0%}"

        descs = describe_cascade_stats({
            "guarantee": stats.guarantee,
            "oracle_calls": stats.oracle_calls,
            "n_items": stats.n_items,
            "served_by_proxy": served_by_proxy,
            "label_budget": stats.label_budget,
            "calibration_calls": stats.calibration_calls,
            "gap_verified": stats.gap_verified,
            "threshold": stats.threshold,
            "escalation_rate": stats.escalation_rate,
        })

        lines = [
            f"[bold magenta]Cascade[/bold magenta] {op_label} "
            f"[bold]'{name}'[/bold]{tag}",
        ]
        lines.append(
            f"           [dim]proxy[/dim]     [cyan]{proxy_model}[/cyan] "
            f"· {stats.proxy_calls} scored · [green]${proxy_cost:.4f}[/green]"
        )
        lines.append(
            f"           [dim]oracle[/dim]    [cyan]{oracle_model}[/cyan] "
            f"· {descs['oracle_desc']} · [green]${oracle_cost:.4f}[/green]"
        )
        lines.append(
            f"           [dim]guarantee[/dim] [yellow]{stats.guarantee} "
            f"≥ {target_pct}[/yellow]  [dim]δ={stats.delta}[/dim]"
        )
        lines.append(
            f"           [dim]threshold[/dim] [yellow]{descs['threshold_desc']}[/yellow]"
            if descs["threshold_desc"] != "n/a" and "n/a" not in descs["threshold_desc"]
            else f"           [dim]threshold[/dim] [dim]{descs['threshold_desc']}[/dim]"
        )
        is_calibrated = stats.guarantee in ("precision", "recall", "precision+recall")
        if is_calibrated:
            lines.append(f"           [dim]result[/dim]   {descs['result_desc']}")
        else:
            lines.append(f"           [dim]escalation[/dim] {descs['result_desc']}")
        lines.append(
            f"           [dim]total cost[/dim] [green]${cost:.4f}[/green]"
        )
        self.console.log("\n".join(lines))

        if getattr(stats, 'gap_truncated', False):
            self.console.log(
                f"           [bold yellow]⚠ gap verification capped by label_budget "
                f"— recall is best-effort. Increase label_budget for stronger recall."
                f"[/bold yellow]"
            )

        if stats.escalation_rate >= 0.95 and stats.n_items > 10:
            if is_calibrated:
                self.console.log(
                    f"           [bold yellow]⚠ calibration used {stats.oracle_calls} "
                    f"of {stats.label_budget} budget — threshold may be unreliable"
                    f"[/bold yellow]"
                )
            else:
                self.console.log(
                    f"           [bold yellow]⚠ escalated "
                    f"{stats.escalation_rate:.0%} of items to oracle — "
                    f"proxy saved almost no cost[/bold yellow]"
                )

    def _run_binary_cascade(
        self,
        *,
        items: list,
        render_messages: Callable[[Any], list[dict[str, str]]],
        proxy_labels: list,
        oracle_predict: Callable[[Any], "tuple[Any, float]"],
        default_guarantee: str,
        positive_label: Any = True,
        negative_label: Any = False,
        op_label: str = "cascade",
    ) -> "tuple[Any, float]":
        """Run the engine over ``items`` and return ``(CascadeResult, cost)``.

        Caches the result (keyed on op identity + config + dataset signature)
        so identical re-runs skip calibration/labeling. Honors the operation's
        ``bypass_cache``. Args:
            items: opaque items the adapters understand (records or pairs).
            render_messages: item -> chat messages for the proxy.
            proxy_labels: candidate labels rendered as the proxy's menu.
            oracle_predict: item -> ``(label, cost)`` using the operator's
                existing full-quality call.
            default_guarantee: applied when the config omits ``guarantee``.
            positive_label / negative_label: label space for precision/recall.
            op_label: short string for the summary log line.
        """
        # Imported lazily so numpy is only required when a cascade actually runs.
        from docetl.operations.utils.cascade import CascadeSpec, CategoricalCascade

        cfg = self._cascade_cfg()
        guarantee = cfg.get("guarantee") or default_guarantee

        bypass = self.config.get("bypass_cache", getattr(self, "bypass_cache", False))
        key = self._cascade_cache_key(
            items, proxy_labels, guarantee, positive_label, negative_label
        )
        if not bypass:
            with cache as c:
                cached = c.get(key)
            if cached is not None:
                result, total_cost, pc, oc, pscores, plabels, _ = cached
                self._report_cascade(
                    op_label, result.stats, total_cost, True,
                    proxy_cost=pc, oracle_cost=oc,
                    proxy_scores=pscores,
                    escalated=result.escalated,
                    proxy_labels=plabels,
                )
                return result, total_cost

        spec = CascadeSpec(
            proxy_model=cfg["proxy_model"],
            guarantee=guarantee,
            target=cfg["target"],
            delta=cfg.get("delta", 0.05),
            label_budget=cfg.get("label_budget", 400),
            positive_label=positive_label,
            negative_label=negative_label,
        )
        if guarantee in ("precision", "recall", "precision+recall") and spec.label_budget < 50:
            fallback = (
                "keep everything (no filtering)"
                if guarantee == "recall"
                else "return only oracle-confirmed positives (very few results)"
            )
            self.console.log(
                f"[bold yellow]Warning:[/bold yellow] cascade label_budget="
                f"{spec.label_budget} is very small. With fewer than ~50 oracle "
                f"samples the {guarantee} threshold search may not reach "
                f"confidence, causing the cascade to {fallback}. Consider "
                f"label_budget ≥ 100."
            )
        elif guarantee in ("precision", "recall", "precision+recall") and spec.label_budget < len(items) * 0.05:
            self.console.log(
                f"[bold yellow]Warning:[/bold yellow] cascade label_budget="
                f"{spec.label_budget} is small relative to {len(items)} items "
                f"({spec.label_budget / len(items):.0%}). The {guarantee} "
                f"guarantee may degrade — consider increasing label_budget."
            )

        import threading
        cost = {"proxy": 0.0, "oracle": 0.0}
        _lock = threading.Lock()
        proxy_scores_live: list[float] = []
        proxy_labels_live: list = []
        oracle_model = self._oracle_model_name()
        progress = _CascadeProgress(
            self.console,
            n_items=len(items),
            proxy_model=spec.proxy_model,
            oracle_model=oracle_model,
            label_budget=spec.label_budget,
            guarantee=guarantee,
            status=getattr(self, "status", None),
            proxy_scores=proxy_scores_live,
        )
        try:
            _proxy_cache: dict[str, tuple] = {}

            def proxy_predict(item):
                msgs = render_messages(item)
                cache_key = msgs[0]["content"] if msgs else ""
                with _lock:
                    cached = _proxy_cache.get(cache_key)
                if cached is not None:
                    return cached
                lbl, prob, c = self.runner.api._classify_with_logprob_with_cost(
                    spec.proxy_model, msgs, proxy_labels
                )
                p_pos = prob if lbl == positive_label else (1.0 - prob)
                with _lock:
                    cost["proxy"] += c
                    proxy_labels_live.append(lbl)
                    proxy_scores_live.append(p_pos)
                    _proxy_cache[cache_key] = (lbl, prob)
                progress.tick_proxy()
                return lbl, prob

            def _oracle(item):
                lbl, c = oracle_predict(item)
                with _lock:
                    cost["oracle"] += c
                progress.tick_oracle()
                return lbl

            cascade = CategoricalCascade(
                spec, proxy_predict, _oracle,
                max_threads=self.max_threads,
            )
            result = cascade.run(items)
        finally:
            progress.finish()

        total_cost = cost["proxy"] + cost["oracle"]
        self._report_cascade(
            op_label, result.stats, total_cost, False,
            proxy_cost=cost["proxy"], oracle_cost=cost["oracle"],
            proxy_scores=cascade.proxy_scores,
            escalated=result.escalated,
            proxy_labels=proxy_labels_live,
        )

        with cache as c:
            c.set(key, (result, total_cost, cost["proxy"], cost["oracle"],
                        cascade.proxy_scores, proxy_labels_live, True))
        return result, total_cost
