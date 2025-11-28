"""The `FilterOperation` class is a subclass of `BaseOperation` that implements a filtering operation on input data using a language model."""

from typing import Any

from pydantic import model_validator

from docetl.operations.map import MapOperation


class FilterOperation(MapOperation):
    class schema(MapOperation.schema):
        type: str = "filter"
        prompt: str
        output: dict[str, Any]

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
            return super().execute(input_data)
        finally:
            self._filter_is_build = previous_state
