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

from typing import Any, Callable, Optional

from pydantic import BaseModel, Field, model_validator

_GUARANTEES = ("accuracy", "precision", "recall")


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

        Args:
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
        spec = CascadeSpec(
            proxy_model=cfg["proxy_model"],
            guarantee=cfg.get("guarantee") or default_guarantee,
            target=cfg["target"],
            delta=cfg.get("delta", 0.05),
            label_budget=cfg.get("label_budget", 400),
            positive_label=positive_label,
            negative_label=negative_label,
        )
        # Mutable accumulator; the engine drives the adapters sequentially.
        cost = {"total": 0.0}

        def proxy_predict(item):
            lbl, prob, c = self.runner.api._classify_with_logprob_with_cost(
                spec.proxy_model, render_messages(item), proxy_labels
            )
            cost["total"] += c
            return lbl, prob

        def _oracle(item):
            lbl, c = oracle_predict(item)
            cost["total"] += c
            return lbl

        result = CategoricalCascade(spec, proxy_predict, _oracle).run(items)

        s = result.stats
        self.console.log(
            f"[bold green]Cascade {op_label} "
            f"'{self.config.get('name', '?')}'[/bold green]: {s.n_items} items, "
            f"{s.oracle_calls} oracle / {s.proxy_calls} proxy calls (escalation "
            f"{s.escalation_rate:.0%}); guarantee={s.guarantee} target={s.target}, "
            f"delta={s.delta}"
        )
        return result, cost["total"]
