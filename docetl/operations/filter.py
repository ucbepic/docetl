"""The `FilterOperation` class is a subclass of `BaseOperation` that implements a filtering operation on input data using a language model."""

from typing import Any

from pydantic import model_validator

from docetl.operations.map import MapOperation
from docetl.operations.utils.validation import (
    convert_schema_to_dict_format,
    is_pydantic_model,
)


class FilterOperation(MapOperation):
    class schema(MapOperation.schema):
        type: str = "filter"
        prompt: str
        output: dict[str, Any] | Any

        @model_validator(mode="after")
        def validate_filter_output_schema(self):
            # Check that schema exists and has the right structure for filtering
            raw_schema = self.output["schema"]

            # Convert Pydantic schema to dict format for validation
            if is_pydantic_model(raw_schema):
                schema_dict = convert_schema_to_dict_format(raw_schema)
            else:
                schema_dict = raw_schema

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

        This method performs the following steps:
        1. Processes each input item using an LLM model
        2. Validates the output
        3. Filters the results based on the specified filter key
        4. Calculates the total cost of the operation

        The method uses multi-threading to process items in parallel, improving performance
        for large datasets.

        Usage:
        ```python
        from docetl.operations import FilterOperation

        config = {
            "prompt": "Determine if the following item is important: {{input}}",
            "output": {
                "schema": {"is_important": "bool"}
            },
            "model": "gpt-3.5-turbo"
        }
        filter_op = FilterOperation(config)
        input_data = [
            {"id": 1, "text": "Critical update"},
            {"id": 2, "text": "Regular maintenance"}
        ]
        results, cost = filter_op.execute(input_data)
        print(f"Filtered results: {results}")
        print(f"Total cost: {cost}")
        ```
        """
        filter_key = next(
            iter(
                [
                    k
                    for k in self.config["output"]["schema"].keys()
                    if k != "_short_explanation"
                ]
            )
        )

        results, total_cost = super().execute(input_data)

        # Drop records with filter_key values that are False
        if not is_build:
            results = [result for result in results if result[filter_key]]

        # Drop the filter_key from the results
        for result in results:
            result.pop(filter_key, None)

        return results, total_cost
