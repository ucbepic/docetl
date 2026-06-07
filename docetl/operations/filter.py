"""The `FilterOperation` class is a subclass of `BaseOperation` that implements a filtering operation on input data using a language model."""

import asyncio
from typing import Any

from litellm.utils import ModelResponse
from pydantic import model_validator

from docetl.operations.map import MapOperation
from docetl.operations.utils import strict_render
from docetl.operations.utils.api import OutputMode
from docetl.progress.tracker import active_tracker

# Re-exported for backwards compatibility; the canonical definition now lives in
# cascade_runner so all operators share one config.
from docetl.operations.utils.cascade_runner import CascadeConfig, CascadeMixin  # noqa: F401


class FilterOperation(MapOperation, CascadeMixin):
    class schema(MapOperation.schema):
        type: str = "filter"
        prompt: str
        output: dict[str, Any]
        cascade: CascadeConfig | None = None

        @model_validator(mode="after")
        def validate_cascade_inputs(self):
            if self.cascade is not None:
                bad = [
                    name
                    for name in ("pdf_url_key", "retriever")
                    if getattr(self, name, None)
                ]
                if bad:
                    raise ValueError(
                        "cascade cannot yet be combined with "
                        + " or ".join(bad)
                        + " (the proxy/oracle would not receive the PDF or "
                        "retrieved context). Remove the cascade block or these "
                        "inputs."
                    )
            return self

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

    def _execute_cascade(
        self, input_data: list[dict]
    ) -> tuple[list[dict], float]:
        """Run the filter as a guarantee-bearing proxy/oracle cascade.

        Builds two thin adapters over the operation's prompt -- a cheap proxy
        (single-token logprob classification) and the existing full-quality
        oracle call -- and hands them to the shared cascade runner. Records the
        engine labels positive (kept) are returned in input order. Default
        guarantee is ``recall`` (don't drop relevant docs).
        """
        if not input_data:
            return [], 0.0

        oracle_model = self.config.get("model", self.default_model)
        schema = self.config["output"]["schema"]
        structured_mode = (
            self.config.get("output", {}).get("mode")
            == OutputMode.STRUCTURED_OUTPUT.value
        )

        def render_messages(item: dict) -> list[dict[str, str]]:
            rendered = strict_render(self.config["prompt"], {"input": item})
            return [{"role": "user", "content": rendered}]

        def oracle_predict(item: dict) -> tuple[bool, float]:
            if self.runner.is_cancelled:
                raise asyncio.CancelledError("Operation was cancelled")
            llm_result = self.runner.api.call_llm(
                oracle_model,
                "filter",
                render_messages(item),
                schema,
                timeout_seconds=self.config.get("timeout", 120),
                max_retries_per_timeout=self.config.get("max_retries_per_timeout", 2),
                bypass_cache=self.config.get("bypass_cache", self.bypass_cache),
                litellm_completion_kwargs=self.config.get(
                    "litellm_completion_kwargs", {}
                ),
                op_config=self.config,
            )
            response = llm_result.response
            if isinstance(response, ModelResponse):
                parsed = self.runner.api.parse_llm_response(
                    response, schema=schema, use_structured_output=structured_mode
                )[0]
            else:
                parsed = response
            return bool(parsed.get(self._filter_key)), llm_result.total_cost

        result, cost = self._run_categorical_cascade(
            items=input_data,
            render_messages=render_messages,
            proxy_labels=[True, False],
            oracle_predict=oracle_predict,
            default_guarantee="recall",
            op_label="filter",
        )

        kept_indices = [i for i, lbl in enumerate(result.labels) if bool(lbl)]
        tracker = active_tracker()
        if tracker is not None:
            tracker.update_cascade_info({"kept_input_indices": kept_indices})

        kept = [input_data[i] for i in kept_indices]
        return kept, cost
