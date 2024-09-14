"""
Implements a reduce operation on input data using language models.

Extends BaseOperation to reduce grouped data using batch, incremental, and parallel strategies.

Manages performance metrics and dynamically adjusts processing (i.e., number of parallel folds) based on these metrics.
"""

import math
import random
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, List, Optional, Tuple

import jinja2
import numpy as np
from jinja2 import Template
from docetl.utils import completion_cost
from litellm import embedding
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from docetl.operations.base import BaseOperation
from docetl.operations.utils import (
    call_llm,
    call_llm_with_gleaning,
    parse_llm_response,
    rich_as_completed,
    validate_output,
    gen_embedding,
)


class ReduceOperation(BaseOperation):
    """
    A class that implements a reduce operation on input data using language models.

    This class extends BaseOperation to provide functionality for reducing grouped data
    using various strategies including batch reduce, incremental reduce, and parallel fold and merge.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the ReduceOperation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.min_samples = 5
        self.max_samples = 1000
        self.fold_times = deque(maxlen=self.max_samples)
        self.merge_times = deque(maxlen=self.max_samples)
        self.lock = Lock()
        self.config["reduce_key"] = (
            [self.config["reduce_key"]]
            if isinstance(self.config["reduce_key"], str)
            else self.config["reduce_key"]
        )

    def syntax_check(self) -> None:
        """
        Perform comprehensive syntax checks on the configuration of the ReduceOperation.

        This method validates the presence and correctness of all required configuration keys, Jinja2 templates, and ensures the correct
        structure and types of the entire configuration.

        The method performs the following checks:
        1. Verifies the presence of all required keys in the configuration.
        2. Validates the structure and content of the 'output' configuration, including its 'schema'.
        3. Checks if the main 'prompt' is a valid Jinja2 template and contains the required 'inputs' variable.
        4. If 'merge_prompt' is specified, ensures that 'fold_prompt' is also present.
        5. If 'fold_prompt' is present, verifies the existence of 'fold_batch_size'.
        6. Validates the 'fold_prompt' as a Jinja2 template with required variables 'inputs' and 'output'.
        7. If present, checks 'merge_prompt' as a valid Jinja2 template with required 'outputs' variable.
        8. Verifies types of various configuration inputs (e.g., 'fold_batch_size' as int).
        9. Checks for the presence and validity of optional configurations like 'model'.

        Raises:
            ValueError: If any required configuration is missing, if templates are invalid or missing required
                        variables, or if any other configuration aspect is incorrect or inconsistent.
            TypeError: If any configuration value has an incorrect type, such as 'schema' not being a dict
                       or 'fold_batch_size' not being an integer.
        """
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
            if "inputs" not in template_var_names:
                raise ValueError("Template must include the 'inputs' variable")
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
                required_vars = {"inputs", "output"}
                if not required_vars.issubset(fold_template_var_names):
                    raise ValueError(
                        f"Fold template must include variables: {required_vars}. Current template includes: {fold_template_var_names}"
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

        # Check if reduce_key is a string or a list of strings
        if not isinstance(self.config["reduce_key"], (str, list)):
            raise TypeError("'reduce_key' must be a string or a list of strings")
        if isinstance(self.config["reduce_key"], list):
            if not all(isinstance(key, str) for key in self.config["reduce_key"]):
                raise TypeError("All elements in 'reduce_key' list must be strings")

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

        if "value_sampling" in self.config:
            sampling = self.config["value_sampling"]
            if not isinstance(sampling, dict):
                raise TypeError("'value_sampling' must be a dictionary")

            if "enabled" not in sampling:
                raise ValueError(
                    "'enabled' is required in 'value_sampling' configuration"
                )
            if not isinstance(sampling["enabled"], bool):
                raise TypeError("'enabled' in 'value_sampling' must be a boolean")

            if sampling["enabled"]:
                if "sample_size" not in sampling:
                    raise ValueError(
                        "'sample_size' is required when value_sampling is enabled"
                    )
                if (
                    not isinstance(sampling["sample_size"], int)
                    or sampling["sample_size"] <= 0
                ):
                    raise ValueError("'sample_size' must be a positive integer")

                if "method" not in sampling:
                    raise ValueError(
                        "'method' is required when value_sampling is enabled"
                    )
                if sampling["method"] not in [
                    "random",
                    "first_n",
                    "cluster",
                    "sem_sim",
                ]:
                    raise ValueError(
                        "Invalid 'method'. Must be 'random', 'first_n', or 'embedding'"
                    )

                if sampling["method"] == "embedding":
                    if "embedding_model" not in sampling:
                        raise ValueError(
                            "'embedding_model' is required when using embedding-based sampling"
                        )
                    if "embedding_keys" not in sampling:
                        raise ValueError(
                            "'embedding_keys' is required when using embedding-based sampling"
                        )

        self.gleaning_check()

    def execute(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        """
        Execute the reduce operation on the provided input data.

        This method sorts and groups the input data by the reduce key(s), then processes each group
        using either parallel fold and merge, incremental reduce, or batch reduce strategies.

        Args:
            input_data (List[Dict]): The input data to process.

        Returns:
            Tuple[List[Dict], float]: A tuple containing the processed results and the total cost of the operation.
        """
        reduce_keys = self.config["reduce_key"]
        if isinstance(reduce_keys, str):
            reduce_keys = [reduce_keys]
        input_schema = self.config.get("input", {}).get("schema", {})

        # Check if we need to group everything into one group
        if reduce_keys == ["_all"] or reduce_keys == "_all":
            grouped_data = [("_all", input_data)]
        else:
            # Group the input data by the reduce key(s) while maintaining original order
            def get_group_key(item):
                return tuple(item[key] for key in reduce_keys)

            grouped_data = {}
            for item in input_data:
                key = get_group_key(item)
                if key not in grouped_data:
                    grouped_data[key] = []
                grouped_data[key].append(item)

            # Convert the grouped data to a list of tuples
            grouped_data = list(grouped_data.items())

        def process_group(
            key: Tuple, group_elems: List[Dict]
        ) -> Tuple[Optional[Dict], float]:
            if input_schema:
                group_list = [
                    {k: item[k] for k in input_schema.keys() if k in item}
                    for item in group_elems
                ]
            else:
                group_list = group_elems

            total_cost = 0.0

            # Apply value sampling if enabled
            value_sampling = self.config.get("value_sampling", {})
            if value_sampling.get("enabled", False):
                sample_size = min(value_sampling["sample_size"], len(group_list))
                method = value_sampling["method"]

                if method == "random":
                    group_sample = random.sample(group_list, sample_size)
                    group_sample.sort(key=lambda x: group_list.index(x))
                elif method == "first_n":
                    group_sample = group_list[:sample_size]
                elif method == "cluster":
                    group_sample, embedding_cost = self._cluster_based_sampling(
                        group_list, value_sampling, sample_size
                    )
                    group_sample.sort(key=lambda x: group_list.index(x))
                    total_cost += embedding_cost
                elif method == "sem_sim":
                    group_sample, embedding_cost = self._semantic_similarity_sampling(
                        key, group_list, value_sampling, sample_size
                    )
                    group_sample.sort(key=lambda x: group_list.index(x))
                    total_cost += embedding_cost

                group_list = group_sample

            # Only execute merge-based plans if associative = True
            if "merge_prompt" in self.config and self.config.get("associative", True):
                result, cost = self._parallel_fold_and_merge(key, group_list)
            elif "fold_prompt" in self.config:
                result, cost = self._incremental_reduce(key, group_list)
            else:
                result, cost = self._batch_reduce(key, group_list)

            total_cost += cost

            # Apply pass-through at the group level
            if (
                result is not None
                and self.config.get("pass_through", False)
                and group_elems
            ):
                for k, v in group_elems[0].items():
                    if k not in self.config["output"]["schema"] and k not in result:
                        result[k] = v

            return result, total_cost

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [
                executor.submit(process_group, key, group)
                for key, group in grouped_data
            ]
            results = []
            total_cost = 0
            for future in rich_as_completed(
                futures,
                total=len(futures),
                desc="Processing reduce items",
                leave=True,
                console=self.console,
            ):
                output, item_cost = future.result()
                total_cost += item_cost
                if output is not None:
                    results.append(output)

        return results, total_cost

    def _get_embeddings(
        self, items: List[Dict], value_sampling: Dict
    ) -> Tuple[List[List[float]], float]:
        embedding_model = value_sampling["embedding_model"]
        embedding_keys = value_sampling["embedding_keys"]
        if not embedding_keys:
            embedding_keys = list(items[0].keys())
        embeddings = []
        cost = 0
        batch_size = 1000

        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            texts = [
                " ".join(str(item[key]) for key in embedding_keys if key in item)[
                    :10000
                ]
                for item in batch
            ]
            response = gen_embedding(embedding_model, texts)
            embeddings.extend([data["embedding"] for data in response["data"]])
            cost += completion_cost(response)

        return embeddings, cost

    def _cluster_based_sampling(
        self, group_list: List[Dict], value_sampling: Dict, sample_size: int
    ) -> Tuple[List[Dict], float]:
        embeddings, cost = self._get_embeddings(group_list, value_sampling)

        kmeans = KMeans(n_clusters=sample_size, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)

        sampled_items = []
        for i in range(sample_size):
            cluster_items = [
                item for item, label in zip(group_list, cluster_labels) if label == i
            ]
            if cluster_items:
                sampled_items.append(random.choice(cluster_items))

        return sampled_items, cost

    def _semantic_similarity_sampling(
        self, key: Tuple, group_list: List[Dict], value_sampling: Dict, sample_size: int
    ) -> Tuple[List[Dict], float]:
        embedding_model = value_sampling["embedding_model"]
        query_text_template = Template(value_sampling["query_text"])
        query_text = query_text_template.render(
            reduce_key=dict(zip(self.config["reduce_key"], key))
        )

        embeddings, cost = self._get_embeddings(group_list, value_sampling)

        query_response = gen_embedding(embedding_model, [query_text])
        query_embedding = query_response["data"][0]["embedding"]
        cost += completion_cost(query_response)

        similarities = cosine_similarity([query_embedding], embeddings)[0]

        top_k_indices = np.argsort(similarities)[-sample_size:]

        return [group_list[i] for i in top_k_indices], cost

    def _parallel_fold_and_merge(
        self, key: Tuple, group_list: List[Dict]
    ) -> Tuple[Optional[Dict], float]:
        """
        Perform parallel folding and merging on a group of items.

        This method implements a strategy that combines parallel folding of input items
        and merging of intermediate results to efficiently process large groups. It works as follows:
        1. The input group is initially divided into smaller batches for efficient processing.
        2. The method performs an initial round of folding operations on these batches.
        3. After the first round of folds, a few merges are performed to estimate the merge runtime.
        4. Based on the estimated merge runtime and observed fold runtime, it calculates the optimal number of parallel folds. Subsequent rounds of folding are then performed concurrently, with the number of parallel folds determined by the runtime estimates.
        5. The folding process repeats in rounds, progressively reducing the number of items to be processed.
        6. Once all folding operations are complete, the method recursively performs final merges on the fold results to combine them into a final result.
        7. Throughout this process, the method may adjust the number of parallel folds based on updated performance metrics (i.e., fold and merge runtimes) to maintain efficiency.

        Args:
            key (Tuple): The reduce key tuple for the group.
            group_list (List[Dict]): The list of items in the group to be processed.

        Returns:
            Tuple[Optional[Dict], float]: A tuple containing the final merged result (or None if processing failed)
            and the total cost of the operation.
        """
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
                        self.console.log(
                            f"Recalculated num_parallel_folds from {num_parallel_folds} to {new_num_parallel_folds}"
                        )
                        num_parallel_folds = new_num_parallel_folds

        # Final merging if needed
        while len(fold_results) > 1:
            self.console.log(f"Finished folding! Merging {len(fold_results)} items.")
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
        self, key: Tuple, group_list: List[Dict]
    ) -> Tuple[Optional[Dict], float]:
        """
        Perform an incremental reduce operation on a group of items.

        This method processes the group in batches, incrementally folding the results.

        Args:
            key (Tuple): The reduce key tuple for the group.
            group_list (List[Dict]): The list of items in the group to be processed.

        Returns:
            Tuple[Optional[Dict], float]: A tuple containing the final reduced result (or None if processing failed)
            and the total cost of the operation.
        """
        fold_batch_size = self.config["fold_batch_size"]
        total_cost = 0
        current_output = None

        # Calculate and log the number of folds to be performed
        num_folds = (len(group_list) + fold_batch_size - 1) // fold_batch_size

        scratchpad = ""
        for i in range(0, len(group_list), fold_batch_size):
            # Log the current iteration and total number of folds
            current_fold = i // fold_batch_size + 1
            self.console.log(
                f"Processing fold {current_fold} of {num_folds} for group with key {key}"
            )
            batch = group_list[i : i + fold_batch_size]

            folded_output, fold_cost = self._increment_fold(
                key, batch, current_output, scratchpad
            )
            total_cost += fold_cost

            if folded_output is None:
                continue

            # Pop off updated_scratchpad
            if "updated_scratchpad" in folded_output:
                scratchpad = folded_output["updated_scratchpad"]
                self.console.log(f"Updated notes: {scratchpad}")
                del folded_output["updated_scratchpad"]

            current_output = folded_output

        return current_output, total_cost

    def _increment_fold(
        self,
        key: Tuple,
        batch: List[Dict],
        current_output: Optional[Dict],
        scratchpad: Optional[str] = None,
    ) -> Tuple[Optional[Dict], float]:
        """
        Perform an incremental fold operation on a batch of items.

        This method folds a batch of items into the current output using the fold prompt.

        Args:
            key (Tuple): The reduce key tuple for the group.
            batch (List[Dict]): The batch of items to be folded.
            current_output (Optional[Dict]): The current accumulated output, if any.
            scratchpad (Optional[str]): The scratchpad to use for the fold operation.
        Returns:
            Tuple[Optional[Dict], float]: A tuple containing the folded output (or None if processing failed)
            and the cost of the fold operation.
        """
        if current_output is None:
            return self._batch_reduce(key, batch, scratchpad)

        start_time = time.time()
        fold_prompt_template = Template(self.config["fold_prompt"])
        fold_prompt = fold_prompt_template.render(
            inputs=batch,
            output=current_output,
            reduce_key=dict(zip(self.config["reduce_key"], key)),
        )
        response = call_llm(
            self.config.get("model", self.default_model),
            "reduce",
            [{"role": "user", "content": fold_prompt}],
            self.config["output"]["schema"],
            scratchpad=scratchpad,
            console=self.console,
        )
        folded_output = parse_llm_response(response)[0]

        folded_output.update(dict(zip(self.config["reduce_key"], key)))
        fold_cost = completion_cost(response)
        end_time = time.time()
        self._update_fold_time(end_time - start_time)

        if validate_output(self.config, folded_output, self.console):
            return folded_output, fold_cost
        return None, fold_cost

    def _merge_results(
        self, key: Tuple, outputs: List[Dict]
    ) -> Tuple[Optional[Dict], float]:
        """
        Merge multiple outputs into a single result.

        This method merges a list of outputs using the merge prompt.

        Args:
            key (Tuple): The reduce key tuple for the group.
            outputs (List[Dict]): The list of outputs to be merged.

        Returns:
            Tuple[Optional[Dict], float]: A tuple containing the merged output (or None if processing failed)
            and the cost of the merge operation.
        """
        start_time = time.time()
        merge_prompt_template = Template(self.config["merge_prompt"])
        merge_prompt = merge_prompt_template.render(
            outputs=outputs, reduce_key=dict(zip(self.config["reduce_key"], key))
        )
        response = call_llm(
            self.config.get("model", self.default_model),
            "merge",
            [{"role": "user", "content": merge_prompt}],
            self.config["output"]["schema"],
            console=self.console,
        )
        merged_output = parse_llm_response(response)[0]
        merged_output.update(dict(zip(self.config["reduce_key"], key)))
        merge_cost = completion_cost(response)
        end_time = time.time()
        self._update_merge_time(end_time - start_time)

        if validate_output(self.config, merged_output, self.console):
            return merged_output, merge_cost
        return None, merge_cost

    def get_fold_time(self) -> Tuple[float, bool]:
        """
        Get the average fold time or a default value.

        Returns:
            Tuple[float, bool]: A tuple containing the average fold time (or default) and a boolean
            indicating whether the default value was used.
        """
        if "fold_time" in self.config:
            return self.config["fold_time"], False
        with self.lock:
            if len(self.fold_times) >= self.min_samples:
                return sum(self.fold_times) / len(self.fold_times), False
        return 1.0, True  # Default to 1 second if no data is available

    def get_merge_time(self) -> Tuple[float, bool]:
        """
        Get the average merge time or a default value.

        Returns:
            Tuple[float, bool]: A tuple containing the average merge time (or default) and a boolean
            indicating whether the default value was used.
        """
        if "merge_time" in self.config:
            return self.config["merge_time"], False
        with self.lock:
            if len(self.merge_times) >= self.min_samples:
                return sum(self.merge_times) / len(self.merge_times), False
        return 1.0, True  # Default to 1 second if no data is available

    def _update_fold_time(self, time: float) -> None:
        """
        Update the fold time statistics.

        Args:
            time (float): The time taken for a fold operation.
        """
        with self.lock:
            self.fold_times.append(time)

    def _update_merge_time(self, time: float) -> None:
        """
        Update the merge time statistics.

        Args:
            time (float): The time taken for a merge operation.
        """
        with self.lock:
            self.merge_times.append(time)

    def _batch_reduce(
        self, key: Tuple, group_list: List[Dict], scratchpad: Optional[str] = None
    ) -> Tuple[Optional[Dict], float]:
        """
        Perform a batch reduce operation on a group of items.

        This method reduces a group of items into a single output using the reduce prompt.

        Args:
            key (Tuple): The reduce key tuple for the group.
            group_list (List[Dict]): The list of items to be reduced.
            scratchpad (Optional[str]): The scratchpad to use for the reduce operation.
        Returns:
            Tuple[Optional[Dict], float]: A tuple containing the reduced output (or None if processing failed)
            and the cost of the reduce operation.
        """
        prompt_template = Template(self.config["prompt"])
        prompt = prompt_template.render(
            reduce_key=dict(zip(self.config["reduce_key"], key)), inputs=group_list
        )
        item_cost = 0

        if "gleaning" in self.config:
            response, gleaning_cost = call_llm_with_gleaning(
                self.config.get("model", self.default_model),
                "reduce",
                [{"role": "user", "content": prompt}],
                self.config["output"]["schema"],
                self.config["gleaning"]["validation_prompt"],
                self.config["gleaning"]["num_rounds"],
                console=self.console,
            )
            item_cost += gleaning_cost
        else:
            response = call_llm(
                self.config.get("model", self.default_model),
                "reduce",
                [{"role": "user", "content": prompt}],
                self.config["output"]["schema"],
                console=self.console,
                scratchpad=scratchpad,
            )

        item_cost += completion_cost(response)

        output = parse_llm_response(response)[0]
        output.update(dict(zip(self.config["reduce_key"], key)))

        if validate_output(self.config, output, self.console):
            return output, item_cost
        return None, item_cost
