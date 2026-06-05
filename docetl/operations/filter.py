"""The `FilterOperation` class is a subclass of `BaseOperation` that implements a filtering operation on input data using a language model."""

import asyncio
from typing import Any, Optional

from litellm.utils import ModelResponse
from pydantic import BaseModel, Field, model_validator

from docetl.operations.map import MapOperation
from docetl.operations.utils import strict_render
from docetl.operations.utils.api import OutputMode


class CascadeConfig(BaseModel):
    """Opt-in model-cascade configuration for a filter operation.

    Runs a cheap ``proxy_model`` on every record, learns a confidence
    threshold on a small oracle-labeled sample, trusts the proxy above it and
    escalates the rest to the operation's ``model`` (the oracle) -- preserving
    the chosen statistical ``guarantee`` w.p. ``1 - delta``. See
    ``docs/design/model-cascade.md``.
    """

    proxy_model: str
    guarantee: str = "recall"  # filter's natural default: don't drop relevant docs
    target: float = Field(..., gt=0, le=1)
    delta: float = Field(0.05, gt=0, lt=1)
    label_budget: int = Field(400, gt=0)

    @model_validator(mode="after")
    def _check_guarantee(self):
        if self.guarantee not in ("accuracy", "precision", "recall"):
            raise ValueError(
                "cascade.guarantee must be 'accuracy', 'precision', or "
                f"'recall'; got '{self.guarantee}'"
            )
        return self


class FilterOperation(MapOperation):
    class schema(MapOperation.schema):
        type: str = "filter"
        prompt: str
        output: dict[str, Any]
        cascade: Optional[CascadeConfig] = None

        @model_validator(mode="after")
        def validate_filter_output_schema(self):
            # Check that schema exists and has the right structure for filtering
            schema_dict = self.output["schema"]

            # Filter out _short_explanation for validation
            schema = {k: v for k, v in schema_dict.items() if k != "_short_explanation"}
            if len(schema) != 1:
                raise ValueError(
                    "The 'schema' in 'output' configuration must have exactly one key-value pair that maps to a boolean value"
                )

            key, value = next(iter(schema.items()))
            if value not in ["bool", "boolean"]:
                raise TypeError(
                    f"The value in the 'schema' must be of type bool, got {value}"
                )

            return self

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._filter_key = next(
            iter(
                [
                    k
                    for k in self.config["output"]["schema"].keys()
                    if k != "_short_explanation"
                ]
            )
        )
        self._filter_is_build = False

    def _limit_applies_to_inputs(self) -> bool:
        return False

    def _handle_result(self, result: dict[str, Any]) -> tuple[dict | None, bool]:
        keep_record = bool(result.get(self._filter_key))
        result.pop(self._filter_key, None)

        if self._filter_is_build or keep_record:
            return result, keep_record
        return None, False

    def execute(
        self, input_data: list[dict], is_build: bool = False
    ) -> tuple[list[dict], float]:
        """
        Executes the filter operation on the input data.

        Args:
            input_data (list[dict]): A list of dictionaries to process.
            is_build (bool): Whether the operation is being executed in the build phase. Defaults to False.

        Returns:
            tuple[list[dict], float]: A tuple containing the filtered list of dictionaries
            and the total cost of the operation.
        """
        previous_state = self._filter_is_build
        self._filter_is_build = is_build
        try:
            # Model cascade is a batch-level rewrite (proxy-all -> calibrate ->
            # escalate). It is only meaningful for real filtering, not the
            # build/optimize phase, so fall back to the normal map path there.
            if self.config.get("cascade") and not is_build:
                return self._execute_cascade(input_data)
            return super().execute(input_data)
        finally:
            self._filter_is_build = previous_state

    def _cascade_config(self) -> dict[str, Any]:
        """Return the cascade block as a plain dict (it may arrive as a dict or
        a validated ``CascadeConfig``)."""
        cfg = self.config["cascade"]
        if isinstance(cfg, BaseModel):
            return cfg.model_dump()
        return cfg

    def _execute_cascade(
        self, input_data: list[dict]
    ) -> tuple[list[dict], float]:
        """Run the filter as a guarantee-bearing proxy/oracle cascade.

        Builds two thin adapters over the operation's prompt -- a cheap proxy
        (single-token logprob classification) and the existing full-quality
        oracle call -- and hands them to :class:`CategoricalCascade`. Records
        the engine labels positive are kept, in input order.
        """
        # Imported lazily so the cascade engine (numpy) is only required when a
        # cascade is actually configured.
        from docetl.operations.utils.cascade import CascadeSpec, CategoricalCascade

        if not input_data:
            return [], 0.0

        cfg = self._cascade_config()
        spec = CascadeSpec(
            proxy_model=cfg["proxy_model"],
            guarantee=cfg.get("guarantee", "recall"),
            target=cfg["target"],
            delta=cfg.get("delta", 0.05),
            label_budget=cfg.get("label_budget", 400),
            positive_label=True,
            negative_label=False,
        )

        oracle_model = self.config.get("model", self.default_model)
        proxy_model = cfg["proxy_model"]
        schema = self.config["output"]["schema"]
        structured_mode = (
            self.config.get("output", {}).get("mode")
            == OutputMode.STRUCTURED_OUTPUT.value
        )
        # Mutable accumulator; the engine drives the adapters sequentially.
        cost = {"total": 0.0}

        def _messages(item: dict) -> list[dict[str, str]]:
            rendered = strict_render(self.config["prompt"], {"input": item})
            return [{"role": "user", "content": rendered}]

        def proxy_predict(item: dict) -> tuple[bool, float]:
            label, prob, c = self.runner.api._classify_with_logprob_with_cost(
                proxy_model, _messages(item), [True, False]
            )
            cost["total"] += c
            return bool(label), prob

        def oracle_predict(item: dict) -> bool:
            if self.runner.is_cancelled:
                raise asyncio.CancelledError("Operation was cancelled")
            llm_result = self.runner.api.call_llm(
                oracle_model,
                "filter",
                _messages(item),
                schema,
                timeout_seconds=self.config.get("timeout", 120),
                max_retries_per_timeout=self.config.get("max_retries_per_timeout", 2),
                bypass_cache=self.config.get("bypass_cache", self.bypass_cache),
                litellm_completion_kwargs=self.config.get(
                    "litellm_completion_kwargs", {}
                ),
                op_config=self.config,
            )
            cost["total"] += llm_result.total_cost
            response = llm_result.response
            if isinstance(response, ModelResponse):
                parsed = self.runner.api.parse_llm_response(
                    response, schema=schema, use_structured_output=structured_mode
                )[0]
            else:
                parsed = response
            return bool(parsed.get(self._filter_key))

        result = CategoricalCascade(spec, proxy_predict, oracle_predict).run(
            input_data
        )

        kept = [
            item for item, label in zip(input_data, result.labels) if bool(label)
        ]

        stats = result.stats
        self.console.log(
            f"[bold green]Cascade filter '{self.config['name']}'[/bold green]: "
            f"{stats.n_items} items, {stats.oracle_calls} oracle / "
            f"{stats.proxy_calls} proxy calls (escalation "
            f"{stats.escalation_rate:.0%}); guarantee={stats.guarantee} "
            f"target={stats.target}, delta={stats.delta} -> kept {len(kept)}"
        )
        return kept, cost["total"]
