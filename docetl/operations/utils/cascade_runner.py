"""Shared plumbing for running a categorical model cascade inside an operator.

The statistical engine lives in ``cascade.py``; this module provides the
operator-facing pieces reused by filter / map / resolve / equijoin:

- :class:`CascadeConfig` -- the opt-in ``cascade:`` config block.
- :class:`CascadeMixin` -- builds the proxy/oracle adapters around an
  operator's prompt, runs the engine, and accumulates cost.

An operator supplies only: how to enumerate items, how to render an item's
prompt, the candidate label set, and an oracle callable. The guarantee default
is operator-dependent (filter->recall, map->accuracy, resolve/equijoin->
precision) and is passed in at call time.
"""

import hashlib
import json
from typing import TYPE_CHECKING, Any, Callable, Optional

from pydantic import BaseModel, Field, model_validator

from docetl.operations.utils.cache import cache
from docetl.progress.tracker import active_tracker

if TYPE_CHECKING:
    from docetl.operations.utils.progress import RichLoopBar

_GUARANTEES = ("accuracy", "precision", "recall")

# Default guarantee when the cascade block omits ``guarantee`` (per operator).
CASCADE_DEFAULT_GUARANTEE: dict[str, str] = {
    "filter": "recall",
    "map": "accuracy",
    "resolve": "precision",
    "equijoin": "precision",
}


def format_cascade_plan_lines(
    cascade: dict[str, Any] | BaseModel,
    *,
    op_type: str,
    oracle_model: str,
) -> list[str]:
    """Rich-markup lines for the cascade block in the query-plan panel."""
    cfg = cascade.model_dump() if isinstance(cascade, BaseModel) else cascade
    guarantee = cfg.get("guarantee") or CASCADE_DEFAULT_GUARANTEE.get(op_type, "recall")
    target = cfg["target"]
    delta = cfg.get("delta", 0.05)
    budget = cfg.get("label_budget", 400)
    if guarantee in ("recall", "precision"):
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
    target: float = Field(..., gt=0, le=1)
    delta: float = Field(0.05, gt=0, lt=1)
    label_budget: int = Field(400, gt=0)

    @model_validator(mode="after")
    def _check_guarantee(self):
        if self.guarantee is not None and self.guarantee not in _GUARANTEES:
            raise ValueError(
                "cascade.guarantee must be 'accuracy', 'precision', or "
                f"'recall'; got '{self.guarantee}'"
            )
        return self


class _CascadeProgress:
    """Proxy/oracle phase progress for model cascades.

    When the interactive TUI is active, feeds :class:`ProgressTracker` via
    :meth:`set_phase` / :meth:`tick`. Otherwise drives a tqdm bar on the Rich
    console and logs short phase headers.
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
    ) -> None:
        self.console = console
        self.n_items = n_items
        self.proxy_model = proxy_model
        self.oracle_model = oracle_model
        self.label_budget = label_budget
        self.guarantee = guarantee
        self._status = status
        self._bar: RichLoopBar | None = None
        self._oracle_started = False
        self._tracker = active_tracker()
        # Rich's live status spinner and tqdm fight over the terminal; map/filter
        # pause status for the same reason before driving a progress bar.
        if self._tracker is None and self._status is not None:
            self._status.stop()
        self._start_proxy()

    def _start_proxy(self) -> None:
        label = f"proxy ({self.proxy_model})"
        self._begin_phase(self.n_items, label)

    def tick_proxy(self) -> None:
        self._tick()

    def _oracle_total(self) -> int:
        """Upper bound on oracle calls for the live progress denominator."""
        if self.guarantee == "accuracy":
            return self.n_items
        return min(self.n_items, self.label_budget)

    def tick_oracle(self) -> None:
        if not self._oracle_started:
            self._oracle_started = True
            label = f"oracle ({self.oracle_model})"
            self._close_bar()
            self._begin_phase(self._oracle_total(), label)
        self._tick()

    def _begin_phase(self, total: int, label: str) -> None:
        if self._tracker is not None:
            self._tracker.set_phase(total, label=label)
            return
        from docetl.operations.utils.progress import RichLoopBar

        self._close_bar()
        self._bar = RichLoopBar(
            total=total,
            desc=f"Cascade {label}",
            console=self.console,
            leave=False,
        )
        self._bar.__enter__()

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
        self._close_bar()
        if self._tracker is not None:
            self._tracker.clear_phase()
        elif self._status is not None:
            self._status.start()


class CascadeMixin:
    """Mixin giving an operation a ``_run_categorical_cascade`` helper.

    Relies on attributes provided by ``BaseOperation``: ``self.config``,
    ``self.runner`` (with ``.api``), and ``self.console``.
    """

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
    ) -> None:
        """Log a cost/escalation summary and stash stats for programmatic use."""
        self.cascade_stats = stats
        served_by_proxy = stats.n_items - stats.oracle_calls
        tag = " [dim](cached)[/dim]" if cached_hit else ""

        cfg = self._cascade_cfg()
        proxy_model = cfg["proxy_model"]
        oracle_model = self.config.get("model", getattr(self, "default_model", "?"))

        tracker = active_tracker()
        if tracker is not None:
            tracker.set_cascade_info({
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
                "cached": cached_hit,
            })

        name = self.config.get("name", "?")
        target_pct = f"{stats.target:.0%}"
        is_calibrated = stats.guarantee in ("precision", "recall")

        lines = [
            f"[bold magenta]Cascade[/bold magenta] {op_label} "
            f"[bold]'{name}'[/bold]{tag}",
        ]
        lines.append(
            f"           [dim]proxy[/dim]     [cyan]{proxy_model}[/cyan] "
            f"· {stats.proxy_calls} scored · [green]${proxy_cost:.4f}[/green]"
        )
        if is_calibrated:
            lines.append(
                f"           [dim]oracle[/dim]    [cyan]{oracle_model}[/cyan] "
                f"· {stats.oracle_calls} sampled for calibration "
                f"(budget {stats.label_budget}) · [green]${oracle_cost:.4f}[/green]"
            )
        else:
            lines.append(
                f"           [dim]oracle[/dim]    [cyan]{oracle_model}[/cyan] "
                f"· {stats.oracle_calls} escalated · [green]${oracle_cost:.4f}[/green]"
            )
        lines.append(
            f"           [dim]guarantee[/dim] [yellow]{stats.guarantee} "
            f"≥ {target_pct}[/yellow]  [dim]δ={stats.delta}[/dim]"
        )
        if stats.threshold is not None:
            lines.append(
                f"           [dim]threshold[/dim] [yellow]{stats.threshold:.3f}[/yellow] "
                f"proxy confidence"
            )
        if is_calibrated:
            lines.append(
                f"           [dim]result[/dim]   {stats.n_items - served_by_proxy} "
                f"proxy-accepted + {stats.oracle_calls} calibration samples "
                f"→ {stats.n_items} items"
            )
        else:
            esc_pct = f"{stats.escalation_rate:.0%}"
            lines.append(
                f"           [dim]escalation[/dim] {esc_pct} "
                f"· {served_by_proxy}/{stats.n_items} served by proxy"
            )
        lines.append(
            f"           [dim]total cost[/dim] [green]${cost:.4f}[/green]"
        )
        self.console.log("\n".join(lines))

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

    def _run_categorical_cascade(
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
                if len(cached) == 4:
                    result, total_cost, pc, oc = cached
                else:
                    result, total_cost = cached
                    pc, oc = 0.0, 0.0
                self._report_cascade(
                    op_label, result.stats, total_cost, True,
                    proxy_cost=pc, oracle_cost=oc,
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
        if guarantee in ("precision", "recall") and spec.label_budget < 50:
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
        elif guarantee in ("precision", "recall") and spec.label_budget < len(items) * 0.05:
            self.console.log(
                f"[bold yellow]Warning:[/bold yellow] cascade label_budget="
                f"{spec.label_budget} is small relative to {len(items)} items "
                f"({spec.label_budget / len(items):.0%}). The {guarantee} "
                f"guarantee may degrade — consider increasing label_budget."
            )

        # Mutable accumulator; the engine drives the adapters sequentially.
        cost = {"proxy": 0.0, "oracle": 0.0}
        oracle_model = self.config.get("model", self.default_model)
        progress = _CascadeProgress(
            self.console,
            n_items=len(items),
            proxy_model=spec.proxy_model,
            oracle_model=oracle_model,
            label_budget=spec.label_budget,
            guarantee=guarantee,
            status=getattr(self, "status", None),
        )
        try:

            def proxy_predict(item):
                lbl, prob, c = self.runner.api._classify_with_logprob_with_cost(
                    spec.proxy_model, render_messages(item), proxy_labels
                )
                cost["proxy"] += c
                progress.tick_proxy()
                return lbl, prob

            def _oracle(item):
                lbl, c = oracle_predict(item)
                cost["oracle"] += c
                progress.tick_oracle()
                return lbl

            result = CategoricalCascade(spec, proxy_predict, _oracle).run(items)
        finally:
            progress.finish()

        total_cost = cost["proxy"] + cost["oracle"]
        self._report_cascade(
            op_label, result.stats, total_cost, False,
            proxy_cost=cost["proxy"], oracle_cost=cost["oracle"],
        )

        with cache as c:
            c.set(key, (result, total_cost, cost["proxy"], cost["oracle"]))
        return result, total_cost
