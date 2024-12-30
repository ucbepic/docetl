"""
The `MapOperation` and `ParallelMapOperation` classes are subclasses of `BaseOperation` that perform mapping operations on input data. They use LLM-based processing to transform input items into output items based on specified prompts and schemas, and can also perform key dropping operations.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

from docetl.operations.utils import strict_render
from jinja2 import Environment, Template
from tqdm import tqdm

from docetl.operations.base import BaseOperation
from docetl.operations.utils import RichLoopBar
from docetl.base_schemas import Tool, ToolFunction
from docetl.utils import completion_cost
from pydantic import Field, field_validator
from litellm.utils import ModelResponse




class MapOperation(BaseOperation):
    class schema(BaseOperation.schema):
        type: str = "map"
        output: Optional[Dict[str, Any]] = None
        prompt: Optional[str] = None
        model: Optional[str] = None
        optimize: Optional[bool] = None
        recursively_optimize: Optional[bool] = None
        sample_size: Optional[int] = None
        tools: Optional[List[Dict[str, Any]]] = (
            None  # FIXME: Why isn't this using the Tool data class so validation works automatically?
        )
        validation_rules: Optional[List[str]] = Field(None, alias="validate")
        num_retries_on_validate_failure: Optional[int] = None
        gleaning: Optional[Dict[str, Any]] = None
        drop_keys: Optional[List[str]] = None
        timeout: Optional[int] = None
        enable_observability: bool = False
        batch_size: Optional[int] = None
        clustering_method: Optional[str] = None
        batch_prompt: Optional[str] = None
        litellm_completion_kwargs: Dict[str, Any] = Field(default_factory=dict)
        @field_validator("drop_keys")
        def validate_drop_keys(cls, v):
            if isinstance(v, str):
                return [v]
            return v

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

    def syntax_check(self) -> None:
        """
            Checks the configuration of the MapOperation for required keys and valid structure.

        Raises:
            ValueError: If required keys are missing or invalid in the configuration.
            TypeError: If configuration values have incorrect types.
        """
        config = self.schema(**self.config)

        if config.drop_keys:
            if any(not isinstance(key, str) for key in config.drop_keys):
                raise TypeError("All items in 'drop_keys' must be strings")
        elif not (config.prompt and config.output):
            raise ValueError(
                "If 'drop_keys' is not specified, both 'prompt' and 'output' must be present in the configuration"
            )
        
        if config.batch_prompt:
            try:
                template = Template(config.batch_prompt)
                # Test render with a minimal inputs list to validate template
                template.render(inputs=[{}])
            except Exception as e:
                raise ValueError(
                    f"Invalid Jinja2 template in 'batch_prompt' or missing required 'inputs' variable: {str(e)}"
                ) from e

        if config.prompt or config.output:
            for key in ["prompt", "output"]:
                if not getattr(config, key):
                    raise ValueError(
                        f"Missing required key '{key}' in MapOperation configuration"
                    )

            if config.output and not config.output["schema"]:
                raise ValueError("Missing 'schema' in 'output' configuration")

            if config.prompt:
                try:
                    Template(config.prompt)
                except Exception as e:
                    raise ValueError(
                        f"Invalid Jinja2 template in 'prompt': {str(e)}"
                    ) from e

            if config.model and not isinstance(config.model, str):
                raise TypeError("'model' in configuration must be a string")

            if config.tools:
                for tool in config.tools:
                    try:
                        tool_obj = Tool(**tool)
                    except Exception as e:
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

            self.gleaning_check()
        

    def execute(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        """
        Executes the map operation on the provided input data.

        Args:
            input_data (List[Dict]): The input data to process.

        Returns:
            Tuple[List[Dict], float]: A tuple containing the processed results and the total cost of the operation.

        This method performs the following steps:
        1. If a prompt is specified, it processes each input item using the specified prompt and LLM model
        2. Applies gleaning if configured
        3. Validates the output
        4. If drop_keys is specified, it drops the specified keys from each document
        5. Aggregates results and calculates total cost

        The method uses parallel processing to improve performance.
        """
        # Check if there's no prompt and only drop_keys
        if "prompt" not in self.config and "drop_keys" in self.config:
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

        def _process_map_item(item: Dict, initial_result: Optional[Dict] = None) -> Tuple[Optional[Dict], float]:

            prompt = strict_render(self.config["prompt"], {"input": item})

            def validation_fn(response: Union[Dict[str, Any], ModelResponse]):
                output = self.runner.api.parse_llm_response(
                    response,
                    schema=self.config["output"]["schema"],
                    tools=self.config.get("tools", None),
                    manually_fix_errors=self.manually_fix_errors,
                )[0] if isinstance(response, ModelResponse) else response
                for key, value in item.items():
                    if key not in self.config["output"]["schema"]:
                        output[key] = value
                if self.runner.api.validate_output(self.config, output, self.console):
                    return output, True
                return output, False

            self.runner.rate_limiter.try_acquire("call", weight=1)
            llm_result = self.runner.api.call_llm(
                self.config.get("model", self.default_model),
                "map",
                [{"role": "user", "content": prompt}],
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
                    if self.config.get("validate", None)
                    else None
                ),
                gleaning_config=self.config.get("gleaning", None),
                verbose=self.config.get("verbose", False),
                bypass_cache=self.config.get("bypass_cache", False),
                initial_result=initial_result,
                litellm_completion_kwargs=self.config.get("litellm_completion_kwargs", {}),
            )

            if llm_result.validated:
                # Parse the response
                if isinstance(llm_result.response, ModelResponse):
                    output = self.runner.api.parse_llm_response(
                        llm_result.response,
                        schema=self.config["output"]["schema"],
                        tools=self.config.get("tools", None),
                        manually_fix_errors=self.manually_fix_errors,
                    )[0]
                else:
                    output = llm_result.response

                
                # Augment the output with the original item
                output = {**item, **output}
                if self.config.get("enable_observability", False):
                    output[f"_observability_{self.config['name']}"] = {"prompt": prompt}
                return output, llm_result.total_cost

            return None, llm_result.total_cost
        
         # If there's a batch prompt, let's use that
        def _process_map_batch(items: List[Dict]) -> Tuple[List[Dict], float]:
            total_cost = 0
            if len(items) > 1 and self.config.get("batch_prompt", None):
                batch_prompt = strict_render(self.config["batch_prompt"], {"inputs": items})

                # Issue the batch call
                llm_result = self.runner.api.call_llm_batch(
                    self.config.get("model", self.default_model),
                    "batch map",
                    [{"role": "user", "content": batch_prompt}],
                    self.config["output"]["schema"],
                    verbose=self.config.get("verbose", False),
                    timeout_seconds=self.config.get("timeout", 120),
                    max_retries_per_timeout=self.config.get("max_retries_per_timeout", 2),
                    bypass_cache=self.config.get("bypass_cache", False),
                    litellm_completion_kwargs=self.config.get("litellm_completion_kwargs", {}),
                )
                total_cost += llm_result.total_cost

                # Parse the LLM response
                parsed_output = self.runner.api.parse_llm_response(llm_result.response, self.config["output"]["schema"])[0].get("results", [])
                items_and_outputs = [(item, parsed_output[idx] if idx < len(parsed_output) else None) for idx, item in enumerate(items)]
            else:
                items_and_outputs = [(item, None) for item in items]

            # Run _process_map_item for each item 
            all_results = []
            if len(items_and_outputs) > 1:
                with ThreadPoolExecutor(max_workers=self.max_batch_size) as executor:
                    futures = [executor.submit(_process_map_item, items_and_outputs[i][0], items_and_outputs[i][1]) for i in range(len(items_and_outputs))]
                    for i in range(len(futures)):
                        try:
                            result, item_cost = futures[i].result()
                            if result is not None:
                                all_results.append(result)
                            total_cost += item_cost
                        except Exception as e:
                            if self.config.get("skip_on_error", False):
                               self.console.log(f"[bold red]Error in map operation {self.config['name']}, skipping item:[/bold red] {e}")
                               continue
                            else:
                                raise e
            else:
                try:
                    result, item_cost = _process_map_item(items_and_outputs[0][0], items_and_outputs[0][1])
                    if result is not None:
                        all_results.append(result)
                    total_cost += item_cost
                except Exception as e:
                    if self.config.get("skip_on_error", False):
                        self.console.log(f"[bold red]Error in map operation {self.config['name']}, skipping item:[/bold red] {e}")
                    else:
                        raise e

            # Return items and cost
            return all_results, total_cost

        with ThreadPoolExecutor(max_workers=self.max_batch_size) as executor:
            batch_size = self.max_batch_size if self.max_batch_size is not None else 1
            futures = []
            for i in range(0, len(input_data), batch_size):
                batch = input_data[i:i + batch_size]
                futures.append(executor.submit(_process_map_batch, batch))
            results = []
            total_cost = 0
            pbar = RichLoopBar(
                range(len(futures)),
                desc=f"Processing {self.config['name']} (map) on all documents",
                console=self.console,
            )
            for i in pbar:
                result_list, item_cost = futures[i].result()
                if result_list:
                    if "drop_keys" in self.config:
                        result_list = [{
                            k: v
                            for k, v in result.items()
                            if k not in self.config["drop_keys"]
                        } for result in result_list]
                    results.extend(result_list)
                total_cost += item_cost

        if self.status:
            self.status.start()

        return results, total_cost


class ParallelMapOperation(BaseOperation):
    class schema(BaseOperation.schema):
        type: str = "parallel_map"
        prompts: List[Dict[str, Any]]
        output: Dict[str, Any]
        enable_observability: bool = False

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def syntax_check(self) -> None:
        """
        Checks the configuration of the ParallelMapOperation for required keys and valid structure.

        Raises:
            ValueError: If required keys are missing or if the configuration structure is invalid.
            TypeError: If the configuration values have incorrect types.
        """
        if "drop_keys" in self.config:
            if not isinstance(self.config["drop_keys"], list):
                raise TypeError(
                    "'drop_keys' in configuration must be a list of strings"
                )
            for key in self.config["drop_keys"]:
                if not isinstance(key, str):
                    raise TypeError("All items in 'drop_keys' must be strings")
        elif "prompts" not in self.config:
            raise ValueError(
                "If 'drop_keys' is not specified, 'prompts' must be present in the configuration"
            )

        if "prompts" in self.config:
            if not isinstance(self.config["prompts"], list):
                raise ValueError(
                    "ParallelMapOperation requires a 'prompts' list in the configuration"
                )

            if not self.config["prompts"]:
                raise ValueError("The 'prompts' list cannot be empty")

            for i, prompt_config in enumerate(self.config["prompts"]):
                if not isinstance(prompt_config, dict):
                    raise TypeError(f"Prompt configuration {i} must be a dictionary")

                required_keys = ["prompt", "output_keys"]
                for key in required_keys:
                    if key not in prompt_config:
                        raise ValueError(
                            f"Missing required key '{key}' in prompt configuration {i}"
                        )
                if not isinstance(prompt_config["prompt"], str):
                    raise TypeError(
                        f"'prompt' in prompt configuration {i} must be a string"
                    )

                if not isinstance(prompt_config["output_keys"], list):
                    raise TypeError(
                        f"'output_keys' in prompt configuration {i} must be a list"
                    )

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

                # Check if the model is specified (optional)
                if "model" in prompt_config and not isinstance(
                    prompt_config["model"], str
                ):
                    raise TypeError(
                        f"'model' in prompt configuration {i} must be a string"
                    )

            # Check if all output schema keys are covered by the prompts
            output_schema = self.config["output"]["schema"]
            output_keys_covered = set()
            for prompt_config in self.config["prompts"]:
                output_keys_covered.update(prompt_config["output_keys"])

            missing_keys = set(output_schema.keys()) - output_keys_covered
            if missing_keys:
                raise ValueError(
                    f"The following output schema keys are not covered by any prompt: {missing_keys}"
                )

    def execute(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        """
        Executes the parallel map operation on the provided input data.

        Args:
            input_data (List[Dict]): The input data to process.

        Returns:
            Tuple[List[Dict], float]: A tuple containing the processed results and the total cost of the operation.

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
            local_output_schema = {
                key: output_schema[key] for key in prompt_config["output_keys"]
            }
            model = prompt_config.get("model", self.default_model)
            if not model:
                model = self.default_model

            # Start of Selection
            # If there are tools, we need to pass in the tools
            response = self.runner.api.call_llm(
                model,
                "parallel_map",
                [{"role": "user", "content": prompt}],
                local_output_schema,
                tools=prompt_config.get("tools", None),
                timeout_seconds=self.config.get("timeout", 120),
                max_retries_per_timeout=self.config.get("max_retries_per_timeout", 2),
                bypass_cache=self.config.get("bypass_cache", False),
                litellm_completion_kwargs=self.config.get("litellm_completion_kwargs", {}),
            )
            output = self.runner.api.parse_llm_response(
                response.response,
                schema=local_output_schema,
                tools=prompt_config.get("tools", None),
                manually_fix_errors=self.manually_fix_errors,
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
                        item_result[f"_observability_{self.config['name']}"].update({f"prompt_{prompt_index}": prompt})

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
