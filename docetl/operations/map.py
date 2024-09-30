"""
The `MapOperation` and `ParallelMapOperation` classes are subclasses of `BaseOperation` that perform mapping operations on input data. They use LLM-based processing to transform input items into output items based on specified prompts and schemas, and can also perform key dropping operations.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from jinja2 import Template
from tqdm import tqdm

from docetl.operations.base import BaseOperation
from docetl.operations.clustering_utils import cluster_documents_for_map
from docetl.operations.utils import (
    RichLoopBar,
    call_llm,
    call_llm_with_gleaning,
    call_llm_with_validation,
    parse_llm_response,
    validate_output,
)
from docetl.schemas import MapOperationConfig, Tool, ToolFunction
from docetl.utils import completion_cost, render_jinja_template


class MapOperation(BaseOperation):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_batch_size: int = self.config.get(
            "max_batch_size", kwargs.get("max_batch_size", float("inf"))
        )
        self.clustering_method = "random"

    def syntax_check(self) -> None:
        """
            Checks the configuration of the MapOperation for required keys and valid structure.

        Raises:
            ValueError: If required keys are missing or invalid in the configuration.
            TypeError: If configuration values have incorrect types.
        """
        config = MapOperationConfig(**self.config)

        if config.drop_keys:
            if any(not isinstance(key, str) for key in config.drop_keys):
                raise TypeError("All items in 'drop_keys' must be strings")
        elif not (config.prompt and config.output):
            raise ValueError(
                "If 'drop_keys' is not specified, both 'prompt' and 'output' must be present in the configuration"
            )

        if config.prompt or config.output:
            for key in ["prompt", "output"]:
                if not getattr(config, key):
                    raise ValueError(
                        f"Missing required key '{key}' in MapOperation configuration"
                    )

            if config.output and not config.output.schema:
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
                    if not isinstance(tool, Tool):
                        raise TypeError("Tool must be a dictionary")

                    if not (tool.code and tool.function):
                        raise ValueError(
                            "Tool is missing required 'code' or 'function' key"
                        )

                    if not isinstance(tool.function, ToolFunction):
                        raise TypeError("'function' in tool must be a dictionary")

                    for key in ["name", "description", "parameters"]:
                        if not getattr(tool.function, key):
                            raise ValueError(
                                f"Tool is missing required '{key}' in 'function'"
                            )

            self.gleaning_check()

    def execute(
        self, input_data: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], float]:
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

        The method uses parallel processing and batching to improve performance.
        """

        def cluster_documents(documents: List[Dict]) -> List[List[Dict]]:
            if self.clustering_method == "random":
                random.shuffle(documents)
            elif self.clustering_method == "sem_cluster":
                model = SentenceTransformer("all-MiniLM-L6-v2")
                embeddings = model.encode([str(doc) for doc in documents])
                num_clusters = max(1, len(documents) // self.batch_size)
                kmeans = KMeans(n_clusters=num_clusters)
                kmeans.fit(embeddings)
                clusters = {i: [] for i in range(num_clusters)}
                for idx, label in enumerate(kmeans.labels_):
                    clusters[label].append(documents[idx])
                return list(clusters.values())
            return [
                documents[i : i + self.batch_size]
                for i in range(0, len(documents), self.batch_size)
            ]

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

        def _process_map_batch(batch: List[Dict]) -> Tuple[List[Optional[Dict]], float]:
            prompts = []
            for item in batch:
                prompt_template = Template(self.config["prompt"])
                prompts.append(prompt_template.render(input=item))

            def validation_fn(responses: List[Dict[str, Any]]):
                outputs = []
                for response, item in zip(responses, batch):
                    output = parse_llm_response(
                        response,
                        schema=self.config["output"]["schema"],
                        tools=self.config.get("tools", None),
                        manually_fix_errors=self.manually_fix_errors,
                    )[0]
                    for key, value in item.items():
                        if key not in self.config["output"]["schema"]:
                            output[key] = value
                    if validate_output(self.config, output, self.console):
                        outputs.append((output, True))
                    else:
                        outputs.append((output, False))
                return outputs

            messages = [{"role": "user", "content": prompt} for prompt in prompts]

            if "gleaning" in self.config:
                responses, cost, successes = call_llm_with_validation(
                    messages,
                    llm_call_fn=lambda msgs: call_llm_with_gleaning(
                        self.config.get("model", self.default_model),
                        "map",
                        msgs,
                        self.config["output"]["schema"],
                        self.config["gleaning"]["validation_prompt"],
                        self.config["gleaning"]["num_rounds"],
                        self.console,
                        timeout_seconds=self.config.get("timeout", 120),
                        max_retries_per_timeout=self.config.get(
                            "max_retries_per_timeout", 2
                        ),
                    ),
                    validation_fn=validation_fn,
                    val_rule=self.config.get("validate", []),
                    num_retries=self.num_retries_on_validate_failure,
                    console=self.console,
                )
            else:
                responses, cost, successes = call_llm_with_validation(
                    messages,
                    llm_call_fn=lambda msgs: call_llm(
                        self.config.get("model", self.default_model),
                        "map",
                        msgs,
                        self.config["output"]["schema"],
                        tools=self.config.get("tools", None),
                        console=self.console,
                        timeout_seconds=self.config.get("timeout", 120),
                        max_retries_per_timeout=self.config.get(
                            "max_retries_per_timeout", 2
                        ),
                    ),
                    validation_fn=validation_fn,
                    val_rule=self.config.get("validate", []),
                    num_retries=self.num_retries_on_validate_failure,
                    console=self.console,
                )

            outputs = []
            for response, success in zip(responses, successes):
                if success:
                    outputs.append(response)
                else:
                    outputs.append(None)

            return outputs, cost

        batched_data = cluster_documents(input_data)

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [
                executor.submit(_process_map_batch, batch) for batch in batched_data
            ]
            results = []
            total_cost = 0
            pbar = RichLoopBar(
                range(len(futures)),
                desc=f"Processing {self.config['name']} (map) on all documents",
                console=self.console,
            )

            for future in pbar:
                batch_results, batch_cost = future.result()
                for result in batch_results:
                    if result is not None:
                        if "drop_keys" in self.config:
                            result = {
                                k: v
                                for k, v in result.items()
                                if k not in self.config["drop_keys"]
                            }
                        results.append(result)
                total_cost += batch_cost

        if self.status:
            self.status.start()

        return results, total_cost

    def _validate_output(self, response: Dict[str, Any]) -> bool:
        """
        Validates the output of a single map operation against the specified schema.
        """
        schema = self.config["output"]["schema"]
        for key in schema:
            if key not in response:
                self.console.log(f"[red]Error: Missing key '{key}' in output[/red]")
                return False
        return True

    def _process_map_batch(
        self, batch: List[Dict]
    ) -> Tuple[List[Optional[Dict]], float]:
        """
        Processes a single batch of documents with gleaning in parallel.
        """
        results = []
        total_cost = 0.0

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [
                executor.submit(self._process_single_item, item) for item in batch
            ]

            pbar = RichLoopBar(
                futures, desc="Processing batch items", console=self.console
            )

            for future in pbar:
                result, cost = future.result()
                results.append(result)
                total_cost += cost

        return results, total_cost

    def _process_single_item(self, item: Dict) -> Tuple[Optional[Dict], float]:
        """
        Processes a single item from the batch.
        """
        prompt = render_jinja_template(self.config["prompt"], item)

        def validation_fn(response: Dict[str, Any]):
            output = parse_llm_response(response, tools=self.config.get("tools", None))[
                0
            ]
            for key, value in item.items():
                if key not in self.config["output"]["schema"]:
                    output[key] = value
            if validate_output(self.config, output, self.console):
                return output, True
            return output, False

        if "gleaning" in self.config:
            output, cost, success = call_llm_with_validation(
                [{"role": "user", "content": prompt}],
                llm_call_fn=lambda messages: call_llm_with_gleaning(
                    self.config.get("model", self.default_model),
                    "map",
                    messages,
                    self.config["output"]["schema"],
                    self.config["gleaning"]["validation_prompt"],
                    self.config["gleaning"]["num_rounds"],
                    self.console,
                    timeout_seconds=self.config.get("timeout", 120),
                    max_retries_per_timeout=self.config.get(
                        "max_retries_per_timeout", 2
                    ),
                ),
                validation_fn=validation_fn,
                val_rule=self.config.get("validate", []),
                num_retries=self.num_retries_on_validate_failure,
                console=self.console,
            )
        else:
            output, cost, success = call_llm_with_validation(
                [{"role": "user", "content": prompt}],
                llm_call_fn=lambda messages: call_llm(
                    self.config.get("model", self.default_model),
                    "map",
                    messages,
                    self.config["output"]["schema"],
                    tools=self.config.get("tools", None),
                    console=self.console,
                    timeout_seconds=self.config.get("timeout", 120),
                    max_retries_per_timeout=self.config.get(
                        "max_retries_per_timeout", 2
                    ),
                ),
                validation_fn=validation_fn,
                val_rule=self.config.get("validate", []),
                num_retries=self.num_retries_on_validate_failure,
                console=self.console,
            )

        return (output if success else None), cost


class ParallelMapOperation(BaseOperation):
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
            prompt = render_jinja_template(prompt_config["prompt"], item)
            local_output_schema = {
                key: output_schema[key] for key in prompt_config["output_keys"]
            }

            # Start of Selection
            # If there are tools, we need to pass in the tools
            response = call_llm(
                prompt_config.get("model", self.default_model),
                "parallel_map",
                [{"role": "user", "content": prompt}],
                local_output_schema,
                tools=prompt_config.get("tools", None),
                console=self.console,
                timeout_seconds=self.config.get("timeout", 120),
                max_retries_per_timeout=self.config.get("max_retries_per_timeout", 2),
            )
            output = parse_llm_response(
                response,
                schema=local_output_schema,
                tools=prompt_config.get("tools", None),
                manually_fix_errors=self.manually_fix_errors,
            )[0]
            return output, completion_cost(response)

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
                    output, cost = future.result()
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
