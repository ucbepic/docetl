"""
The `MapOperation` and `ParallelMapOperation` classes are subclasses of `BaseOperation` that perform mapping operations on input data. They use LLM-based processing to transform input items into output items based on specified prompts and schemas, and can also perform key dropping operations.
"""

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests
from jinja2 import Template
from litellm.utils import ModelResponse
from pydantic import Field, field_validator, model_validator
from tqdm import tqdm

from docetl.base_schemas import Tool, ToolFunction
from docetl.operations.base import BaseOperation
from docetl.operations.utils import RichLoopBar, strict_render, validate_output_types
from docetl.operations.utils.api import OutputMode
from docetl.utils import has_jinja_syntax, prompt_user_for_non_jinja_confirmation


class MapOperation(BaseOperation):
    class schema(BaseOperation.schema):
        type: str = "map"
        output: dict[str, Any] | None = None
        prompt: str | None = None
        model: str | None = None
        optimize: bool | None = None
        recursively_optimize: bool | None = None
        sample_size: int | None = None
        tools: list[dict[str, Any]] | None = (
            None  # FIXME: Why isn't this using the Tool data class so validation works automatically?
        )
        validation_rules: list[str] | None = Field(None, alias="validate")
        num_retries_on_validate_failure: int | None = None
        drop_keys: list[str] | None = None
        timeout: int | None = None
        enable_observability: bool = False
        batch_size: int | None = None
        clustering_method: str | None = None
        batch_prompt: str | None = None
        litellm_completion_kwargs: dict[str, Any] = {}
        pdf_url_key: str | None = None
        flush_partial_result: bool = False
        limit: int | None = Field(None, gt=0)
        # Calibration parameters
        calibrate: bool = False
        num_calibration_docs: int = Field(10, gt=0)

        @field_validator("batch_prompt")
        def validate_batch_prompt(cls, v):
            if v is not None:
                # Check if it has Jinja syntax
                if not has_jinja_syntax(v):
                    # This will be handled during initialization with user confirmation
                    # We'll mark it for later processing
                    return v
                try:
                    template = Template(v)
                    # Test render with a minimal inputs list to validate template
                    template.render(inputs=[{}])
                except Exception as e:
                    raise ValueError(
                        f"Invalid Jinja2 template in 'batch_prompt' or missing required 'inputs' variable: {str(e)}"
                    ) from e
            return v

        @field_validator("prompt")
        def validate_prompt(cls, v):
            if v is not None:
                # Check if it has Jinja syntax
                if not has_jinja_syntax(v):
                    # This will be handled during initialization with user confirmation
                    # We'll mark it for later processing
                    return v
                try:
                    Template(v)
                except Exception as e:
                    raise ValueError(
                        f"Invalid Jinja2 template in 'prompt': {str(e)}"
                    ) from e
            return v

        @field_validator("tools")
        def validate_tools(cls, v):
            if v is not None:
                for tool in v:
                    try:
                        tool_obj = Tool(**tool)
                    except Exception:
                        raise TypeError("Tool must be a dictionary")

                    if not (tool_obj.code and tool_obj.function):
                        raise ValueError(
                            "Tool is missing required 'code' or 'function' key"
                        )

                    if not isinstance(tool_obj.function, ToolFunction):
                        raise TypeError("'function' in tool must be a dictionary")

                    for key in ["name", "description", "parameters"]:
                        if not getattr(tool_obj.function, key):
                            raise ValueError(
                                f"Tool is missing required '{key}' in 'function'"
                            )
            return v

        @model_validator(mode="after")
        def validate_prompt_and_output_requirements(self):
            # If drop_keys is not specified, both prompt and output must be present
            if not self.drop_keys:
                if not self.prompt or not self.output:
                    raise ValueError(
                        "If 'drop_keys' is not specified, both 'prompt' and 'output' must be present in the configuration"
                    )

                if self.output and not self.output.get("schema"):
                    raise ValueError("Missing 'schema' in 'output' configuration")

            return self

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_batch_size: int = self.config.get(
            "max_batch_size", kwargs.get("max_batch_size", None)
        )
        self.clustering_method = "random"
        # Check for non-Jinja prompts and prompt user for confirmation
        if "prompt" in self.config and not has_jinja_syntax(self.config["prompt"]):
            if not prompt_user_for_non_jinja_confirmation(
                self.config["prompt"], self.config["name"], "prompt"
            ):
                raise ValueError(
                    f"Operation '{self.config['name']}' cancelled by user. Please add Jinja2 template syntax to your prompt."
                )
            # Mark that we need to append document statement
            self.config["_append_document_to_prompt"] = True
        if "batch_prompt" in self.config and not has_jinja_syntax(
            self.config["batch_prompt"]
        ):
            if not prompt_user_for_non_jinja_confirmation(
                self.config["batch_prompt"], self.config["name"], "batch_prompt"
            ):
                raise ValueError(
                    f"Operation '{self.config['name']}' cancelled by user. Please add Jinja2 template syntax to your batch_prompt."
                )
            # Mark that we need to append document statement
            self.config["_append_document_to_batch_prompt"] = True

    def _limit_applies_to_inputs(self) -> bool:
        return True

    def _handle_result(self, result: dict[str, Any]) -> tuple[dict | None, bool]:
        return result, True

    def _generate_calibration_context(self, input_data: list[dict]) -> str:
        """
        Generate calibration context by running the operation on a sample of documents
        and using an LLM to suggest prompt improvements for consistency.

        Returns:
            str: Additional context to add to the original prompt
        """
        import random

        # Set seed for reproducibility
        random.seed(42)

        # Sample documents for calibration
        num_calibration_docs = min(
            self.config.get("num_calibration_docs", 10), len(input_data)
        )
        if num_calibration_docs == len(input_data):
            calibration_sample = input_data
        else:
            calibration_sample = random.sample(input_data, num_calibration_docs)

        self.console.log(
            f"[bold blue]Running calibration on {num_calibration_docs} documents...[/bold blue]"
        )

        # Temporarily disable calibration to avoid infinite recursion
        original_calibrate = self.config.get("calibrate", False)
        self.config["calibrate"] = False

        try:
            # Run the map operation on the calibration sample
            calibration_results, _ = self.execute(calibration_sample)

            # Prepare the calibration analysis prompt
            calibration_prompt = f"""
The following prompt was applied to sample documents to generate these input-output pairs:

"{self.config["prompt"]}"

Sample inputs and their outputs:
"""

            for i, (input_doc, output_doc) in enumerate(
                zip(calibration_sample, calibration_results)
            ):
                calibration_prompt += f"\n--- Example {i+1} ---\n"
                calibration_prompt += f"Input: {input_doc}\n"
                calibration_prompt += f"Output: {output_doc}\n"

            calibration_prompt += """
Based on these examples, provide reference anchors that will be appended to the prompt to help maintain consistency when processing all documents.

DO NOT provide generic advice. Instead, use specific examples from above as calibration points.
Note that the outputs might be incorrect, because the user's prompt was not calibrated or rich in the first place.
You can ignore the outputs if they are incorrect, and focus on the diversity of the inputs.

Format as concrete reference points:
- "For reference, consider '[specific input text]' → [output] as a baseline for [category/level]"
- "Documents similar to '[specific input text]' should be classified as [output]"

Reference anchors:"""

            # Call LLM to get calibration suggestions
            messages = [{"role": "user", "content": calibration_prompt}]
            # Use a copy of the user-provided completion kwargs so we don't mutate the original
            # and avoid hard-coding temperature to a value that may not be supported by certain models.
            completion_kwargs = dict(self.config.get("litellm_completion_kwargs", {}))
            # If the user did not explicitly specify a temperature, let the model default handle it
            # to prevent incompatibility errors with providers that don't support 0.0.
            # If a temperature is already provided, respect the user's choice.

            llm_result = self.runner.api.call_llm(
                self.config.get("model", self.default_model),
                "calibration",
                messages,
                {"calibration_context": "string"},
                timeout_seconds=self.config.get("timeout", 120),
                max_retries_per_timeout=self.config.get("max_retries_per_timeout", 2),
                bypass_cache=self.config.get("bypass_cache", self.bypass_cache),
                litellm_completion_kwargs=completion_kwargs,
                op_config=self.config,
            )

            # Parse the response
            if hasattr(llm_result, "response"):
                calibration_context = self.runner.api.parse_llm_response(
                    llm_result.response,
                    schema={"calibration_context": "string"},
                    manually_fix_errors=self.manually_fix_errors,
                )[0].get("calibration_context", "")
            else:
                calibration_context = ""

            return calibration_context

        finally:
            # Restore original calibration setting
            self.config["calibrate"] = original_calibrate

    def execute(self, input_data: list[dict]) -> tuple[list[dict], float]:
        """
        Executes the map operation on the provided input data.

        Args:
            input_data (list[dict]): The input data to process.

        Returns:
            tuple[list[dict], float]: A tuple containing the processed results and the total cost of the operation.

        This method performs the following steps:
        1. If calibration is enabled, runs calibration to improve prompt consistency
        2. If a prompt is specified, it processes each input item using the specified prompt and LLM model
        3. Applies gleaning if configured
        4. Validates the output
        5. If drop_keys is specified, it drops the specified keys from each document
        6. Aggregates results and calculates total cost

        The method uses parallel processing to improve performance.
        """
        limit_value = self.config.get("limit")

        # Check if there's no prompt and only drop_keys
        if "prompt" not in self.config and "drop_keys" in self.config:
            data_to_process = input_data
            if limit_value is not None and self._limit_applies_to_inputs():
                data_to_process = input_data[:limit_value]
            # If only drop_keys is specified, simply drop the keys and return
            dropped_results = []
            for item in data_to_process:
                new_item = {
                    k: v for k, v in item.items() if k not in self.config["drop_keys"]
                }
                dropped_results.append(new_item)
                if limit_value is not None and len(dropped_results) >= limit_value:
                    break
            return dropped_results, 0.0  # Return the modified data with no cost

        if limit_value is not None and self._limit_applies_to_inputs():
            input_data = input_data[:limit_value]

        # Generate calibration context if enabled
        calibration_context = ""
        if self.config.get("calibrate", False) and "prompt" in self.config:
            calibration_context = self._generate_calibration_context(input_data)
            if calibration_context:
                # Store original prompt for potential restoration
                self._original_prompt = self.config["prompt"]
                # Augment the prompt with calibration context
                self.config["prompt"] = (
                    f"{self.config['prompt']}\n\n{calibration_context}"
                )
                self.console.log(
                    f"[bold green]New map ({self.config['name']}) prompt augmented with context on how to improve consistency:[/bold green] {self.config['prompt']}"
                )
            else:
                self.console.log(
                    f"[bold yellow]Extra context on how to improve consistency failed to generate for map ({self.config['name']}); continuing with prompt as is.[/bold yellow]"
                )

        if self.status:
            self.status.stop()

        def _process_map_item(
            item: dict, initial_result: dict | None = None
        ) -> tuple[dict | None, float]:

            # Build retrieval context (if configured)
            retrieval_context = self._maybe_build_retrieval_context({"input": item})
            ctx = {"input": item, "retrieval_context": retrieval_context}
            rendered = strict_render(self.config["prompt"], ctx)
            # If template didn't use retrieval_context, prepend a standard header
            prompt = (
                f"Here is some extra context:\n{retrieval_context}\n\n{rendered}"
                if retrieval_context
                and "retrieval_context" not in self.config["prompt"]
                else rendered
            )
            messages = [{"role": "user", "content": prompt}]
            if self.config.get("pdf_url_key", None):
                # Append the pdf to the prompt
                try:
                    pdf_url = item[self.config["pdf_url_key"]]
                except KeyError:
                    raise ValueError(
                        f"PDF URL key '{self.config['pdf_url_key']}' not found in input data"
                    )

                # Download content
                if pdf_url.startswith("http"):
                    file_data = requests.get(pdf_url).content
                else:
                    with open(pdf_url, "rb") as f:
                        file_data = f.read()
                encoded_file = base64.b64encode(file_data).decode("utf-8")
                base64_url = f"data:application/pdf;base64,{encoded_file}"

                messages[0]["content"] = [
                    {"type": "image_url", "image_url": {"url": base64_url}},
                    {"type": "text", "text": prompt},
                ]

            def validation_fn(response: dict[str, Any] | ModelResponse):
                structured_mode = (
                    self.config.get("output", {}).get("mode")
                    == OutputMode.STRUCTURED_OUTPUT.value
                )
                output = (
                    self.runner.api.parse_llm_response(
                        response,
                        schema=self.config["output"]["schema"],
                        tools=self.config.get("tools", None),
                        manually_fix_errors=self.manually_fix_errors,
                        use_structured_output=structured_mode,
                    )[0]
                    if isinstance(response, ModelResponse)
                    else response
                )
                # Type-check output values against schema declarations
                is_types_valid, _errors = validate_output_types(
                    output,
                    self.config["output"]["schema"],
                )
                if not is_types_valid:
                    return output, False

                for key, value in item.items():
                    if key not in self.config["output"]["schema"]:
                        output[key] = value
                if self.runner.api.validate_output(self.config, output, self.console):
                    return output, True
                return output, False

            if self.runner.is_cancelled:
                raise asyncio.CancelledError("Operation was cancelled")
            llm_result = self.runner.api.call_llm(
                self.config.get("model", self.default_model),
                "map",
                messages,
                self.config["output"]["schema"],
                tools=self.config.get("tools", None),
                scratchpad=None,
                timeout_seconds=self.config.get("timeout", 120),
                max_retries_per_timeout=self.config.get("max_retries_per_timeout", 2),
                validation_config=(
                    {
                        "num_retries": self.num_retries_on_validate_failure,
                        "val_rule": self.config.get("validate", []),
                        "validation_fn": validation_fn,
                    }
                ),
                gleaning_config=self.config.get("gleaning", None),
                verbose=self.config.get("verbose", False),
                bypass_cache=self.config.get("bypass_cache", self.bypass_cache),
                initial_result=initial_result,
                litellm_completion_kwargs=self.config.get(
                    "litellm_completion_kwargs", {}
                ),
                op_config=self.config,
            )

            if llm_result.validated:
                # Parse the response
                if isinstance(llm_result.response, ModelResponse):
                    structured_mode = (
                        self.config.get("output", {}).get("mode")
                        == OutputMode.STRUCTURED_OUTPUT.value
                    )
                    outputs = self.runner.api.parse_llm_response(
                        llm_result.response,
                        schema=self.config["output"]["schema"],
                        tools=self.config.get("tools", None),
                        manually_fix_errors=self.manually_fix_errors,
                        use_structured_output=structured_mode,
                    )
                else:
                    outputs = [llm_result.response]

                # Augment the output with the original item
                outputs = [{**item, **output} for output in outputs]
                if self.config.get("enable_observability", False):
                    for output in outputs:
                        output[f"_observability_{self.config['name']}"] = {
                            "prompt": prompt
                        }
                # Add retrieved context if save_retriever_output is enabled
                if self.config.get("save_retriever_output", False):
                    for output in outputs:
                        output[f"_{self.config['name']}_retrieved_context"] = (
                            retrieval_context if retrieval_context else ""
                        )
                return outputs, llm_result.total_cost

            return None, llm_result.total_cost

        # If there's a batch prompt, let's use that
        def _process_map_batch(items: list[dict]) -> tuple[list[dict], float]:
            total_cost = 0
            if len(items) > 1 and self.config.get("batch_prompt", None):
                # Raise error if pdf_url_key is set
                if self.config.get("pdf_url_key", None):
                    raise ValueError("Batch prompts do not support PDF URLs")

                batch_prompt = strict_render(
                    self.config["batch_prompt"], {"inputs": items}
                )

                # Issue the batch call
                llm_result = self.runner.api.call_llm_batch(
                    self.config.get("model", self.default_model),
                    "batch map",
                    [{"role": "user", "content": batch_prompt}],
                    self.config["output"]["schema"],
                    verbose=self.config.get("verbose", False),
                    timeout_seconds=self.config.get("timeout", 120),
                    max_retries_per_timeout=self.config.get(
                        "max_retries_per_timeout", 2
                    ),
                    bypass_cache=self.config.get("bypass_cache", self.bypass_cache),
                    litellm_completion_kwargs=self.config.get(
                        "litellm_completion_kwargs", {}
                    ),
                )
                total_cost += llm_result.total_cost

                # Parse the LLM response
                structured_mode = (
                    self.config.get("output", {}).get("mode")
                    == OutputMode.STRUCTURED_OUTPUT.value
                )
                parsed_output = self.runner.api.parse_llm_response(
                    llm_result.response,
                    self.config["output"]["schema"],
                    use_structured_output=structured_mode,
                )[0].get("results", [])
                items_and_outputs = [
                    (item, parsed_output[idx] if idx < len(parsed_output) else None)
                    for idx, item in enumerate(items)
                ]
            else:
                items_and_outputs = [(item, None) for item in items]

            # Run _process_map_item for each item
            all_results = []
            if len(items_and_outputs) > 1:
                with ThreadPoolExecutor(max_workers=self.max_batch_size) as executor:
                    futures = [
                        executor.submit(
                            _process_map_item,
                            items_and_outputs[i][0],
                            items_and_outputs[i][1],
                        )
                        for i in range(len(items_and_outputs))
                    ]
                    for i in range(len(futures)):
                        try:
                            results, item_cost = futures[i].result()
                            if results is not None:
                                all_results.extend(results)
                            total_cost += item_cost
                        except Exception as e:
                            if self.config.get("skip_on_error", False):
                                self.console.log(
                                    f"[bold red]Error in map operation {self.config['name']}, skipping item:[/bold red] {e}"
                                )
                                continue
                            else:
                                raise e
            else:
                try:
                    results, item_cost = _process_map_item(
                        items_and_outputs[0][0], items_and_outputs[0][1]
                    )
                    if results is not None:
                        all_results.extend(results)
                    total_cost += item_cost
                except Exception as e:
                    if self.config.get("skip_on_error", False):
                        self.console.log(
                            f"[bold red]Error in map operation {self.config['name']}, skipping item:[/bold red] {e}"
                        )
                    else:
                        raise e

            return all_results, total_cost

        limit_counter = 0
        batch_size = self.max_batch_size if self.max_batch_size is not None else 1
        total_batches = (len(input_data) + batch_size - 1) // batch_size
        if total_batches == 0:
            if self.status:
                self.status.start()
            return [], 0.0

        worker_limit = self.max_batch_size or self.max_threads or 1
        window_size = (
            total_batches
            if limit_value is None
            else max(1, (limit_value + batch_size - 1) // batch_size)
        )

        results: list[dict] = []
        total_cost = 0.0
        limit_reached = False
        op_name = self.config["name"]

        if limit_value is not None and not self._limit_applies_to_inputs():
            self.console.log(
                f"[yellow]Note: Operation will terminate early once {limit_value} items pass the filter condition.[/yellow]"
            )

        with ThreadPoolExecutor(max_workers=worker_limit) as executor:
            with RichLoopBar(
                total=total_batches,
                desc=f"Processing {op_name} (map) on all documents",
                console=self.console,
            ) as pbar:
                chunk_start = 0
                while chunk_start < total_batches and not limit_reached:
                    chunk_end = min(total_batches, chunk_start + window_size)
                    chunk_ordinals = list(range(chunk_start, chunk_end))
                    futures = []
                    for ordinal in chunk_ordinals:
                        start_idx = ordinal * batch_size
                        batch = input_data[start_idx : start_idx + batch_size]
                        futures.append(executor.submit(_process_map_batch, batch))

                    for relative_idx, future in enumerate(futures):
                        if limit_value is not None and limit_counter >= limit_value:
                            limit_reached = True
                            break

                        result_list, item_cost = future.result()
                        total_cost += item_cost

                        if result_list:
                            if "drop_keys" in self.config:
                                result_list = [
                                    {
                                        k: v
                                        for k, v in result.items()
                                        if k not in self.config["drop_keys"]
                                    }
                                    for result in result_list
                                ]

                            if self.config.get("flush_partial_results", False):
                                self.runner._flush_partial_results(
                                    op_name, chunk_ordinals[relative_idx], result_list
                                )

                            for result in result_list:
                                processed_result, counts_towards_limit = (
                                    self._handle_result(result)
                                )
                                if processed_result is not None:
                                    results.append(processed_result)

                                if limit_value is not None and counts_towards_limit:
                                    limit_counter += 1
                                    if limit_counter >= limit_value:
                                        limit_reached = True
                                        break

                        pbar.update()

                    chunk_start = chunk_end

        if self.status:
            self.status.start()

        return results, total_cost


class ParallelMapOperation(BaseOperation):
    class schema(BaseOperation.schema):
        type: str = "parallel_map"
        prompts: list[dict[str, Any]] | None = None
        output: dict[str, Any] | None = None
        drop_keys: list[str] | None = None
        enable_observability: bool = False
        pdf_url_key: str | None = None

        @field_validator("prompts")
        def validate_prompts(cls, v):
            if v is not None:
                if not v:
                    raise ValueError("The 'prompts' list cannot be empty")

                for i, prompt_config in enumerate(v):
                    # Validate required keys exist
                    if "prompt" not in prompt_config:
                        raise ValueError(
                            f"Missing required key 'prompt' in prompt configuration {i}"
                        )
                    if "output_keys" not in prompt_config:
                        raise ValueError(
                            f"Missing required key 'output_keys' in prompt configuration {i}"
                        )

                    # Validate output_keys is not empty
                    if not prompt_config["output_keys"]:
                        raise ValueError(
                            f"'output_keys' list in prompt configuration {i} cannot be empty"
                        )

                    # Check if the prompt is a valid Jinja2 template
                    try:
                        Template(prompt_config["prompt"])
                    except Exception as e:
                        raise ValueError(
                            f"Invalid Jinja2 template in prompt configuration {i}: {str(e)}"
                        ) from e
            return v

        @model_validator(mode="after")
        def validate_prompt_requirements(self):
            # If drop_keys is not specified, prompts must be present
            if not self.drop_keys and not self.prompts:
                raise ValueError(
                    "If 'drop_keys' is not specified, 'prompts' must be present in the configuration"
                )

            # Check if all output schema keys are covered by the prompts
            if self.prompts and self.output and "schema" in self.output:
                output_schema = self.output["schema"]
                output_keys_covered = set()
                for prompt_config in self.prompts:
                    output_keys_covered.update(prompt_config["output_keys"])

                missing_keys = set(output_schema.keys()) - output_keys_covered
                if missing_keys:
                    raise ValueError(
                        f"The following output schema keys are not covered by any prompt: {missing_keys}"
                    )

            return self

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def execute(self, input_data: list[dict]) -> tuple[list[dict], float]:
        """
        Executes the parallel map operation on the provided input data.

        Args:
            input_data (list[dict]): The input data to process.

        Returns:
            tuple[list[dict], float]: A tuple containing the processed results and the total cost of the operation.

        This method performs the following steps:
        1. If prompts are specified, it processes each input item using multiple prompts in parallel
        2. Aggregates results from different prompts for each input item
        3. Validates the combined output for each item
        4. If drop_keys is specified, it drops the specified keys from each document
        5. Calculates total cost of the operation
        """
        results = {}
        total_cost = 0
        output_schema = self.config.get("output", {}).get("schema", {})

        # Check if there's no prompt and only drop_keys
        if "prompts" not in self.config and "drop_keys" in self.config:
            # If only drop_keys is specified, simply drop the keys and return
            dropped_results = []
            for item in input_data:
                new_item = {
                    k: v for k, v in item.items() if k not in self.config["drop_keys"]
                }
                dropped_results.append(new_item)
            return dropped_results, 0.0  # Return the modified data with no cost

        if self.status:
            self.status.stop()

        def process_prompt(item, prompt_config):
            prompt = strict_render(prompt_config["prompt"], {"input": item})
            messages = [{"role": "user", "content": prompt}]
            if self.config.get("pdf_url_key", None):
                try:
                    pdf_url = item[self.config["pdf_url_key"]]
                except KeyError:
                    raise ValueError(
                        f"PDF URL key '{self.config['pdf_url_key']}' not found in input data"
                    )
                # Download content
                if pdf_url.startswith("http"):
                    file_data = requests.get(pdf_url).content
                else:
                    with open(pdf_url, "rb") as f:
                        file_data = f.read()
                encoded_file = base64.b64encode(file_data).decode("utf-8")
                base64_url = f"data:application/pdf;base64,{encoded_file}"

                messages[0]["content"] = [
                    {"type": "image_url", "image_url": {"url": base64_url}},
                    {"type": "text", "text": prompt},
                ]

            local_output_schema = {
                key: output_schema.get(key, "string")
                for key in prompt_config["output_keys"]
            }
            model = prompt_config.get("model", self.default_model)
            if not model:
                model = self.default_model

            # Start of Selection
            # If there are tools, we need to pass in the tools
            response = self.runner.api.call_llm(
                model,
                "parallel_map",
                messages,
                local_output_schema,
                tools=prompt_config.get("tools", None),
                timeout_seconds=self.config.get("timeout", 120),
                max_retries_per_timeout=self.config.get("max_retries_per_timeout", 2),
                gleaning_config=prompt_config.get("gleaning", None),
                bypass_cache=self.config.get("bypass_cache", self.bypass_cache),
                litellm_completion_kwargs=self.config.get(
                    "litellm_completion_kwargs", {}
                ),
                op_config=self.config,
            )
            structured_mode = (
                self.config.get("output", {}).get("mode")
                == OutputMode.STRUCTURED_OUTPUT.value
            )
            output = self.runner.api.parse_llm_response(
                response.response,
                schema=local_output_schema,
                tools=prompt_config.get("tools", None),
                manually_fix_errors=self.manually_fix_errors,
                use_structured_output=structured_mode,
            )[0]
            return output, prompt, response.total_cost

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            if "prompts" in self.config:
                # Create all futures at once
                all_futures = [
                    executor.submit(process_prompt, item, prompt_config)
                    for item in input_data
                    for prompt_config in self.config["prompts"]
                ]

                # Process results in order
                for i in tqdm(
                    range(len(all_futures)),
                    desc="Processing parallel map items",
                ):
                    future = all_futures[i]
                    output, prompt, cost = future.result()
                    total_cost += cost

                    # Determine which item this future corresponds to
                    item_index = i // len(self.config["prompts"])
                    prompt_index = i % len(self.config["prompts"])

                    # Initialize or update the item_result
                    if prompt_index == 0:
                        item_result = input_data[item_index].copy()
                        results[item_index] = item_result

                    # Fetch the item_result
                    item_result = results[item_index]

                    if self.config.get("enable_observability", False):
                        if f"_observability_{self.config['name']}" not in item_result:
                            item_result[f"_observability_{self.config['name']}"] = {}
                        item_result[f"_observability_{self.config['name']}"].update(
                            {f"prompt_{prompt_index}": prompt}
                        )

                    # Update the item_result with the output
                    item_result.update(output)

            else:
                results = {i: item.copy() for i, item in enumerate(input_data)}

        # Apply drop_keys if specified
        if "drop_keys" in self.config:
            drop_keys = self.config["drop_keys"]
            for item in results.values():
                for key in drop_keys:
                    item.pop(key, None)

        if self.status:
            self.status.start()

        # Return the results in order
        return [results[i] for i in range(len(input_data)) if i in results], total_cost
