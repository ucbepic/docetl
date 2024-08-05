import math
import time
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from jinja2 import Template
from itertools import groupby
from operator import itemgetter
from motion.operations.base import BaseOperation
from motion.operations.utils import call_llm, parse_llm_response
from motion.operations.utils import validate_output
from litellm import completion_cost
import jinja2
from threading import Lock
from collections import deque


class ReduceOperation(BaseOperation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_samples = 5
        self.max_samples = 1000
        self.fold_times = deque(maxlen=self.max_samples)
        self.merge_times = deque(maxlen=self.max_samples)
        self.lock = Lock()

    def syntax_check(self) -> None:
        required_keys = ["reduce_key", "prompt", "output"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(
                    f"Missing required key '{key}' in ReduceOperation configuration"
                )

        if "schema" not in self.config["output"]:
            raise ValueError("Missing 'schema' in 'output' configuration")

        if not isinstance(self.config["output"]["schema"], dict):
            raise TypeError("'schema' in 'output' configuration must be a dictionary")

        if not self.config["output"]["schema"]:
            raise ValueError("'schema' in 'output' configuration cannot be empty")

        # Check if the prompt is a valid Jinja2 template
        try:
            template = Template(self.config["prompt"])
            template_vars = template.environment.parse(self.config["prompt"]).find_all(
                jinja2.nodes.Name
            )
            template_var_names = {var.name for var in template_vars}
            if "values" not in template_var_names:
                raise ValueError("Template must include the 'values' variable")
        except Exception as e:
            raise ValueError(f"Invalid Jinja2 template in 'prompt': {str(e)}")

        # Check if fold_prompt is a valid Jinja2 template (now required if merge exists)
        if "merge_prompt" in self.config:
            if "fold_prompt" not in self.config:
                raise ValueError(
                    "'fold_prompt' is required when 'merge_prompt' is specified"
                )

        if "fold_prompt" in self.config:
            if "fold_batch_size" not in self.config:
                raise ValueError(
                    "'fold_batch_size' is required when 'fold_prompt' is specified"
                )

            try:
                fold_template = Template(self.config["fold_prompt"])
                fold_template_vars = fold_template.environment.parse(
                    self.config["fold_prompt"]
                ).find_all(jinja2.nodes.Name)
                fold_template_var_names = {var.name for var in fold_template_vars}
                required_vars = {"values", "output"}
                if not required_vars.issubset(fold_template_var_names):
                    raise ValueError(
                        f"Fold template must include variables: {required_vars}"
                    )
            except Exception as e:
                raise ValueError(f"Invalid Jinja2 template in 'fold_prompt': {str(e)}")

        # Check merge_prompt and merge_batch_size
        if "merge_prompt" in self.config:
            if "merge_batch_size" not in self.config:
                raise ValueError(
                    "'merge_batch_size' is required when 'merge_prompt' is specified"
                )

            try:
                merge_template = Template(self.config["merge_prompt"])
                merge_template_vars = merge_template.environment.parse(
                    self.config["merge_prompt"]
                ).find_all(jinja2.nodes.Name)
                merge_template_var_names = {var.name for var in merge_template_vars}
                if "outputs" not in merge_template_var_names:
                    raise ValueError(
                        "Merge template must include the 'outputs' variable"
                    )
            except Exception as e:
                raise ValueError(f"Invalid Jinja2 template in 'merge_prompt': {str(e)}")

        # Check if the model is specified (optional)
        if "model" in self.config and not isinstance(self.config["model"], str):
            raise TypeError("'model' in configuration must be a string")

        # Check if reduce_key is a string
        if not isinstance(self.config["reduce_key"], str):
            raise TypeError("'reduce_key' must be a string")

        # Check if input schema is provided and valid (optional)
        if "input" in self.config:
            if "schema" not in self.config["input"]:
                raise ValueError("Missing 'schema' in 'input' configuration")
            if not isinstance(self.config["input"]["schema"], dict):
                raise TypeError(
                    "'schema' in 'input' configuration must be a dictionary"
                )

        # Check if fold_batch_size and merge_batch_size are positive integers
        for key in ["fold_batch_size", "merge_batch_size"]:
            if key in self.config:
                if not isinstance(self.config[key], int) or self.config[key] <= 0:
                    raise ValueError(f"'{key}' must be a positive integer")

    def execute(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        reduce_key = self.config["reduce_key"]
        input_schema = self.config.get("input", {}).get("schema", {})

        # Sort the input data by the reduce key
        sorted_data = sorted(input_data, key=itemgetter(reduce_key))

        grouped_data = groupby(sorted_data, key=itemgetter(reduce_key))
        grouped_data = [(key, list(group)) for key, group in grouped_data]

        def process_group(
            key: Any, group_list: List[Dict]
        ) -> Tuple[Optional[Dict], float]:
            if input_schema:
                group_list = [
                    {k: item[k] for k in input_schema.keys() if k in item}
                    for item in group_list
                ]

            if "merge_prompt" in self.config:
                result, cost = self._parallel_fold_and_merge(key, group_list)
            elif "fold_prompt" in self.config:
                result, cost = self._incremental_reduce(key, group_list)
            else:
                result, cost = self._batch_reduce(key, group_list)

            # Apply pass-through at the group level
            if (
                result is not None
                and self.config.get("pass_through", False)
                and group_list
            ):
                for k, v in group_list[0].items():
                    if k not in self.config["output"]["schema"] and k not in result:
                        result[k] = v

            return result, cost

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [
                executor.submit(process_group, key, group)
                for key, group in grouped_data
            ]
            results = []
            total_cost = 0
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing reduce items",
                leave=True,
            ):
                output, item_cost = future.result()
                total_cost += item_cost
                if output is not None:
                    results.append(output)

        return results, total_cost

    def _parallel_fold_and_merge(
        self, key: Any, group_list: List[Dict]
    ) -> Tuple[Optional[Dict], float]:
        fold_batch_size = self.config["fold_batch_size"]
        merge_batch_size = self.config["merge_batch_size"]
        total_cost = 0

        def calculate_num_parallel_folds():
            fold_time, fold_default = self.get_fold_time()
            merge_time, merge_default = self.get_merge_time()
            num_group_items = len(group_list)
            return (
                max(
                    1,
                    int(
                        (fold_time * num_group_items * math.log(merge_batch_size))
                        / (fold_batch_size * merge_time)
                    ),
                ),
                fold_default or merge_default,
            )

        num_parallel_folds, used_default_times = calculate_num_parallel_folds()
        fold_results = []
        remaining_items = group_list

        # Parallel folding and merging
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            while remaining_items:
                # Folding phase
                fold_futures = []
                for i in range(min(num_parallel_folds, len(remaining_items))):
                    batch = remaining_items[:fold_batch_size]
                    remaining_items = remaining_items[fold_batch_size:]
                    current_output = fold_results[i] if i < len(fold_results) else None
                    fold_futures.append(
                        executor.submit(
                            self._increment_fold, key, batch, current_output
                        )
                    )

                new_fold_results = []
                for future in as_completed(fold_futures):
                    result, cost = future.result()
                    total_cost += cost
                    if result is not None:
                        new_fold_results.append(result)

                # Update fold_results with new results
                fold_results = new_fold_results + fold_results[len(new_fold_results) :]

                # Single pass merging phase
                if (
                    len(self.merge_times) < self.min_samples
                    and len(fold_results) >= merge_batch_size
                ):
                    merge_futures = []
                    for i in range(0, len(fold_results), merge_batch_size):
                        batch = fold_results[i : i + merge_batch_size]
                        merge_futures.append(
                            executor.submit(self._merge_results, key, batch)
                        )

                    new_results = []
                    for future in as_completed(merge_futures):
                        result, cost = future.result()
                        total_cost += cost
                        if result is not None:
                            new_results.append(result)

                    fold_results = new_results

                # Recalculate num_parallel_folds if we used default times
                if used_default_times:
                    new_num_parallel_folds, used_default_times = (
                        calculate_num_parallel_folds()
                    )
                    if not used_default_times:
                        self.console.print(
                            f"Recalculated num_parallel_folds from {num_parallel_folds} to {new_num_parallel_folds}"
                        )
                        num_parallel_folds = new_num_parallel_folds

        # Final merging if needed
        self.console.print(f"Finished folding! Merging {len(fold_results)} items.")
        while len(fold_results) > 1:
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                merge_futures = []
                for i in range(0, len(fold_results), merge_batch_size):
                    batch = fold_results[i : i + merge_batch_size]
                    merge_futures.append(
                        executor.submit(self._merge_results, key, batch)
                    )

                new_results = []
                for future in as_completed(merge_futures):
                    result, cost = future.result()
                    total_cost += cost
                    if result is not None:
                        new_results.append(result)

                fold_results = new_results

        return (fold_results[0], total_cost) if fold_results else (None, total_cost)

    def _incremental_reduce(
        self, key: Any, group_list: List[Dict]
    ) -> Tuple[Optional[Dict], float]:
        fold_batch_size = self.config["fold_batch_size"]
        total_cost = 0
        current_output = None

        for i in range(0, len(group_list), fold_batch_size):
            batch = group_list[i : i + fold_batch_size]

            folded_output, fold_cost = self._increment_fold(key, batch, current_output)
            total_cost += fold_cost

            if folded_output is None:
                return None, total_cost

            current_output = folded_output

        return current_output, total_cost

    def _increment_fold(
        self, key: Any, batch: List[Dict], current_output: Optional[Dict]
    ) -> Tuple[Optional[Dict], float]:
        if current_output is None:
            return self._batch_reduce(key, batch)

        start_time = time.time()
        fold_prompt_template = Template(self.config["fold_prompt"])
        fold_prompt = fold_prompt_template.render(values=batch, output=current_output)
        response = call_llm(
            self.config.get("model", self.default_model),
            "reduce",
            fold_prompt,
            self.config["output"]["schema"],
        )
        folded_output = parse_llm_response(response)[0]
        folded_output[self.config["reduce_key"]] = key
        fold_cost = completion_cost(response)
        end_time = time.time()
        self._update_fold_time(end_time - start_time)

        if validate_output(self.config, folded_output, self.console):
            return folded_output, fold_cost
        return None, fold_cost

    def _merge_results(
        self, key: Any, outputs: List[Dict]
    ) -> Tuple[Optional[Dict], float]:
        start_time = time.time()
        merge_prompt_template = Template(self.config["merge_prompt"])
        merge_prompt = merge_prompt_template.render(outputs=outputs, reduce_key=key)
        response = call_llm(
            self.config.get("model", self.default_model),
            "merge",
            merge_prompt,
            self.config["output"]["schema"],
        )
        merged_output = parse_llm_response(response)[0]
        merged_output[self.config["reduce_key"]] = key
        merge_cost = completion_cost(response)
        end_time = time.time()
        self._update_merge_time(end_time - start_time)

        if validate_output(self.config, merged_output, self.console):
            return merged_output, merge_cost
        return None, merge_cost

    def get_fold_time(self) -> Tuple[float, bool]:
        if "fold_time" in self.config:
            return self.config["fold_time"], False
        with self.lock:
            if len(self.fold_times) >= self.min_samples:
                return sum(self.fold_times) / len(self.fold_times), False
        return 1.0, True  # Default to 1 second if no data is available

    def get_merge_time(self) -> Tuple[float, bool]:
        if "merge_time" in self.config:
            return self.config["merge_time"], False
        with self.lock:
            if len(self.merge_times) >= self.min_samples:
                return sum(self.merge_times) / len(self.merge_times), False
        return 1.0, True  # Default to 1 second if no data is available

    def _update_fold_time(self, time: float) -> None:
        with self.lock:
            self.fold_times.append(time)

    def _update_merge_time(self, time: float) -> None:
        with self.lock:
            self.merge_times.append(time)

    def _batch_reduce(
        self, key: Any, group_list: List[Dict]
    ) -> Tuple[Optional[Dict], float]:
        prompt_template = Template(self.config["prompt"])
        prompt = prompt_template.render(reduce_key=key, values=group_list)
        response = call_llm(
            self.config.get("model", self.default_model),
            "reduce",
            prompt,
            self.config["output"]["schema"],
        )
        output = parse_llm_response(response)[0]
        output[self.config["reduce_key"]] = key
        item_cost = completion_cost(response)

        if validate_output(self.config, output, self.console):
            return output, item_cost
        return None, item_cost
