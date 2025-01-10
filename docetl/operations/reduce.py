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
from typing import Any, Dict, List, Optional, Tuple, Union

import jinja2
import numpy as np
from jinja2 import Template
from pydantic import Field

from docetl.operations.base import BaseOperation
from docetl.operations.clustering_utils import (
    cluster_documents,
    get_embeddings_for_clustering,
)
from docetl.operations.utils import rich_as_completed, strict_render
from docetl.utils import completion_cost


class ReduceOperation(BaseOperation):
    """
    A class that implements a reduce operation on input data using language models.

    This class extends BaseOperation to provide functionality for reducing grouped data
    using various strategies including batch reduce, incremental reduce, and parallel fold and merge.
    """

    class schema(BaseOperation.schema):
        type: str = "reduce"
        reduce_key: Union[str, List[str]]
        output: Optional[Dict[str, Any]] = None
        prompt: Optional[str] = None
        optimize: Optional[bool] = None
        synthesize_resolve: Optional[bool] = None
        model: Optional[str] = None
        input: Optional[Dict[str, Any]] = None
        pass_through: Optional[bool] = None
        associative: Optional[bool] = None
        fold_prompt: Optional[str] = None
        fold_batch_size: Optional[int] = None
        value_sampling: Optional[Dict[str, Any]] = None
        verbose: Optional[bool] = None
        timeout: Optional[int] = None
        litellm_completion_kwargs: Dict[str, Any] = Field(default_factory=dict)
        enable_observability: bool = False

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
        self.intermediates = {}
        self.lineage_keys = self.config.get("output", {}).get("lineage", [])

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
                    f"Missing required key '{key}' in {self.config['name']} configuration"
                )

        if "schema" not in self.config["output"]:
            raise ValueError(
                f"Missing 'schema' in {self.config['name']} 'output' configuration"
            )

        if not isinstance(self.config["output"]["schema"], dict):
            raise TypeError(
                f"'schema' in {self.config['name']} 'output' configuration must be a dictionary"
            )

        if not self.config["output"]["schema"]:
            raise ValueError(
                f"'schema' in {self.config['name']} 'output' configuration cannot be empty"
            )

        # Check if the prompt is a valid Jinja2 template
        try:
            template = Template(self.config["prompt"])
            template_vars = template.environment.parse(self.config["prompt"]).find_all(
                jinja2.nodes.Name
            )
            template_var_names = {var.name for var in template_vars}
            if "inputs" not in template_var_names:
                raise ValueError(
                    f"Prompt template for {self.config['name']} must include the 'inputs' variable"
                )
        except Exception as e:
            raise ValueError(
                f"Invalid Jinja2 template in {self.config['name']} 'prompt': {str(e)}"
            )

        # Check if fold_prompt is a valid Jinja2 template (now required if merge exists)
        if "merge_prompt" in self.config:
            if "fold_prompt" not in self.config:
                raise ValueError(
                    f"'fold_prompt' is required when 'merge_prompt' is specified in {self.config['name']}"
                )

        if "fold_prompt" in self.config:
            if "fold_batch_size" not in self.config:
                raise ValueError(
                    f"'fold_batch_size' is required when 'fold_prompt' is specified in {self.config['name']}"
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
                        f"Fold template in {self.config['name']} must include variables: {required_vars}. Current template includes: {fold_template_var_names}"
                    )
            except Exception as e:
                raise ValueError(
                    f"Invalid Jinja2 template in {self.config['name']} 'fold_prompt': {str(e)}"
                )

        # Check merge_prompt and merge_batch_size
        if "merge_prompt" in self.config:
            if "merge_batch_size" not in self.config:
                raise ValueError(
                    f"'merge_batch_size' is required when 'merge_prompt' is specified in {self.config['name']}"
                )

            try:
                merge_template = Template(self.config["merge_prompt"])
                merge_template_vars = merge_template.environment.parse(
                    self.config["merge_prompt"]
                ).find_all(jinja2.nodes.Name)
                merge_template_var_names = {var.name for var in merge_template_vars}
                if "outputs" not in merge_template_var_names:
                    raise ValueError(
                        f"Merge template in {self.config['name']} must include the 'outputs' variable"
                    )
            except Exception as e:
                raise ValueError(
                    f"Invalid Jinja2 template in {self.config['name']} 'merge_prompt': {str(e)}"
                )

        # Check if the model is specified (optional)
        if "model" in self.config and not isinstance(self.config["model"], str):
            raise TypeError(
                f"'model' in {self.config['name']} configuration must be a string"
            )

        # Check if reduce_key is a string or a list of strings
        if not isinstance(self.config["reduce_key"], (str, list)):
            raise TypeError(
                f"'reduce_key' in {self.config['name']} configuration must be a string or a list of strings"
            )
        if isinstance(self.config["reduce_key"], list):
            if not all(isinstance(key, str) for key in self.config["reduce_key"]):
                raise TypeError(
                    f"All elements in 'reduce_key' list in {self.config['name']} configuration must be strings"
                )

        # Check if input schema is provided and valid (optional)
        if "input" in self.config:
            if "schema" not in self.config["input"]:
                raise ValueError(
                    f"Missing 'schema' in {self.config['name']} 'input' configuration"
                )
            if not isinstance(self.config["input"]["schema"], dict):
                raise TypeError(
                    f"'schema' in {self.config['name']} 'input' configuration must be a dictionary"
                )

        # Check if fold_batch_size and merge_batch_size are positive integers
        for key in ["fold_batch_size", "merge_batch_size"]:
            if key in self.config:
                if not isinstance(self.config[key], int) or self.config[key] <= 0:
                    raise ValueError(
                        f"'{key}' in {self.config['name']} configuration must be a positive integer"
                    )

        if "value_sampling" in self.config:
            sampling = self.config["value_sampling"]
            if not isinstance(sampling, dict):
                raise TypeError(
                    f"'value_sampling' in {self.config['name']} configuration must be a dictionary"
                )

            if "enabled" not in sampling:
                raise ValueError(
                    f"'enabled' is required in {self.config['name']} 'value_sampling' configuration"
                )
            if not isinstance(sampling["enabled"], bool):
                raise TypeError(
                    f"'enabled' in {self.config['name']} 'value_sampling' configuration must be a boolean"
                )

            if sampling["enabled"]:
                if "sample_size" not in sampling:
                    raise ValueError(
                        f"'sample_size' is required when value_sampling is enabled in {self.config['name']}"
                    )
                if (
                    not isinstance(sampling["sample_size"], int)
                    or sampling["sample_size"] <= 0
                ):
                    raise ValueError(
                        f"'sample_size' in {self.config['name']} configuration must be a positive integer"
                    )

                if "method" not in sampling:
                    raise ValueError(
                        f"'method' is required when value_sampling is enabled in {self.config['name']}"
                    )
                if sampling["method"] not in [
                    "random",
                    "first_n",
                    "cluster",
                    "sem_sim",
                ]:
                    raise ValueError(
                        f"Invalid 'method'. Must be 'random', 'first_n', or 'embedding' in {self.config['name']}"
                    )

                if sampling["method"] == "embedding":
                    if "embedding_model" not in sampling:
                        raise ValueError(
                            f"'embedding_model' is required when using embedding-based sampling in {self.config['name']}"
                        )
                    if "embedding_keys" not in sampling:
                        raise ValueError(
                            f"'embedding_keys' is required when using embedding-based sampling in {self.config['name']}"
                        )

        # Check if lineage is a list of strings
        if "lineage" in self.config.get("output", {}):
            if not isinstance(self.config["output"]["lineage"], list):
                raise TypeError(
                    f"'lineage' in {self.config['name']} 'output' configuration must be a list"
                )
            if not all(
                isinstance(key, str) for key in self.config["output"]["lineage"]
            ):
                raise TypeError(
                    f"All elements in 'lineage' list in {self.config['name']} 'output' configuration must be strings"
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
        if self.config.get("gleaning", {}).get("validation_prompt", None):
            self.console.log(
                f"Using gleaning with validation prompt: {self.config.get('gleaning', {}).get('validation_prompt', '')}"
            )

        reduce_keys = self.config["reduce_key"]
        if isinstance(reduce_keys, str):
            reduce_keys = [reduce_keys]
        input_schema = self.config.get("input", {}).get("schema", {})

        if self.status:
            self.status.stop()

        # Check if we need to group everything into one group
        if reduce_keys == ["_all"] or reduce_keys == "_all":
            grouped_data = [("_all", input_data)]
        else:
            # Group the input data by the reduce key(s) while maintaining original order
            def get_group_key(item):
                key_values = []
                for key in reduce_keys:
                    value = item[key]
                    # Special handling for list-type values
                    if isinstance(value, list):
                        key_values.append(
                            tuple(sorted(value))
                        )  # Convert list to sorted tuple
                    else:
                        key_values.append(value)
                return tuple(key_values)

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
                result, prompts, cost = self._parallel_fold_and_merge(key, group_list)
            elif self.config.get("fold_batch_size", None) and self.config.get(
                "fold_batch_size"
            ) >= len(group_list):
                # If the fold batch size is greater than or equal to the number of items in the group,
                # we can just run a single fold operation
                result, prompt, cost = self._batch_reduce(key, group_list)
                prompts = [prompt]
            elif "fold_prompt" in self.config:
                result, prompts, cost = self._incremental_reduce(key, group_list)
            else:
                result, prompt, cost = self._batch_reduce(key, group_list)
                prompts = [prompt]

            total_cost += cost

            # Add the counts of items in the group to the result
            result[f"_counts_prereduce_{self.config['name']}"] = len(group_elems)

            if self.config.get("enable_observability", False):
                # Add the _observability_{self.config['name']} key to the result
                result[f"_observability_{self.config['name']}"] = {"prompts": prompts}

            # Apply pass-through at the group level
            if (
                result is not None
                and self.config.get("pass_through", False)
                and group_elems
            ):
                for k, v in group_elems[0].items():
                    if k not in self.config["output"]["schema"] and k not in result:
                        result[k] = v

            # Add lineage information
            if result is not None and self.lineage_keys:
                lineage = []
                for item in group_elems:
                    lineage_item = {
                        k: item.get(k) for k in self.lineage_keys if k in item
                    }
                    if lineage_item:
                        lineage.append(lineage_item)
                result[f"{self.config['name']}_lineage"] = lineage

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
                desc=f"Processing {self.config['name']} (reduce) on all documents",
                leave=True,
                console=self.console,
            ):
                output, item_cost = future.result()
                total_cost += item_cost
                if output is not None:
                    results.append(output)

        if self.config.get("persist_intermediates", False):
            for result in results:
                key = tuple(result[k] for k in self.config["reduce_key"])
                if key in self.intermediates:
                    result[f"_{self.config['name']}_intermediates"] = (
                        self.intermediates[key]
                    )

        if self.status:
            self.status.start()

        return results, total_cost

    def _cluster_based_sampling(
        self, group_list: List[Dict], value_sampling: Dict, sample_size: int
    ) -> Tuple[List[Dict], float]:
        if sample_size >= len(group_list):
            return group_list, 0

        clusters, cost = cluster_documents(
            group_list, value_sampling, sample_size, self.runner.api
        )

        sampled_items = []
        idx_added_already = set()
        num_clusters = len(clusters)
        for i in range(sample_size):
            # Add a random item from the cluster
            idx = i % num_clusters

            # Skip if there are no items in the cluster
            if len(clusters[idx]) == 0:
                continue

            if len(clusters[idx]) == 1:
                # If there's only one item in the cluster, add it directly if we haven't already
                if idx not in idx_added_already:
                    sampled_items.append(clusters[idx][0])
                continue

            random_choice_idx = random.randint(0, len(clusters[idx]) - 1)
            max_attempts = 10
            while random_choice_idx in idx_added_already and max_attempts > 0:
                random_choice_idx = random.randint(0, len(clusters[idx]) - 1)
                max_attempts -= 1
            idx_added_already.add(random_choice_idx)
            sampled_items.append(clusters[idx][random_choice_idx])

        return sampled_items, cost

    def _semantic_similarity_sampling(
        self, key: Tuple, group_list: List[Dict], value_sampling: Dict, sample_size: int
    ) -> Tuple[List[Dict], float]:
        embedding_model = value_sampling["embedding_model"]
        query_text = strict_render(
            value_sampling["query_text"],
            {"reduce_key": dict(zip(self.config["reduce_key"], key))},
        )

        embeddings, cost = get_embeddings_for_clustering(
            group_list, value_sampling, self.runner.api
        )

        query_response = self.runner.api.gen_embedding(embedding_model, [query_text])
        query_embedding = query_response["data"][0]["embedding"]
        cost += completion_cost(query_response)

        from sklearn.metrics.pairwise import cosine_similarity

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
        prompts = []

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

        if self.config.get("persist_intermediates", False):
            self.intermediates[key] = []
            iter_count = 0

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
                    result, prompt, cost = future.result()
                    total_cost += cost
                    prompts.append(prompt)
                    if result is not None:
                        new_fold_results.append(result)
                        if self.config.get("persist_intermediates", False):
                            self.intermediates[key].append(
                                {
                                    "iter": iter_count,
                                    "intermediate": result,
                                    "scratchpad": result["updated_scratchpad"],
                                }
                            )
                            iter_count += 1

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
                        result, prompt, cost = future.result()
                        total_cost += cost
                        prompts.append(prompt)
                        if result is not None:
                            new_results.append(result)
                            if self.config.get("persist_intermediates", False):
                                self.intermediates[key].append(
                                    {
                                        "iter": iter_count,
                                        "intermediate": result,
                                        "scratchpad": None,
                                    }
                                )
                                iter_count += 1

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
                    result, prompt, cost = future.result()
                    total_cost += cost
                    prompts.append(prompt)
                    if result is not None:
                        new_results.append(result)
                        if self.config.get("persist_intermediates", False):
                            self.intermediates[key].append(
                                {
                                    "iter": iter_count,
                                    "intermediate": result,
                                    "scratchpad": None,
                                }
                            )
                            iter_count += 1

                fold_results = new_results

        return (
            (fold_results[0], prompts, total_cost)
            if fold_results
            else (None, prompts, total_cost)
        )

    def _incremental_reduce(
        self, key: Tuple, group_list: List[Dict]
    ) -> Tuple[Optional[Dict], List[str], float]:
        """
        Perform an incremental reduce operation on a group of items.

        This method processes the group in batches, incrementally folding the results.

        Args:
            key (Tuple): The reduce key tuple for the group.
            group_list (List[Dict]): The list of items in the group to be processed.

        Returns:
            Tuple[Optional[Dict], List[str], float]: A tuple containing the final reduced result (or None if processing failed),
            the list of prompts used, and the total cost of the operation.
        """
        fold_batch_size = self.config["fold_batch_size"]
        total_cost = 0
        current_output = None
        prompts = []

        # Calculate and log the number of folds to be performed
        num_folds = (len(group_list) + fold_batch_size - 1) // fold_batch_size

        scratchpad = ""
        if self.config.get("persist_intermediates", False):
            self.intermediates[key] = []
            iter_count = 0

        for i in range(0, len(group_list), fold_batch_size):
            # Log the current iteration and total number of folds
            current_fold = i // fold_batch_size + 1
            if self.config.get("verbose", False):
                self.console.log(
                    f"Processing fold {current_fold} of {num_folds} for group with key {key}"
                )
            batch = group_list[i : i + fold_batch_size]

            folded_output, prompt, fold_cost = self._increment_fold(
                key, batch, current_output, scratchpad
            )
            total_cost += fold_cost
            prompts.append(prompt)

            if folded_output is None:
                continue

            if self.config.get("persist_intermediates", False):
                self.intermediates[key].append(
                    {
                        "iter": iter_count,
                        "intermediate": folded_output,
                        "scratchpad": folded_output["updated_scratchpad"],
                    }
                )
                iter_count += 1

            # Pop off updated_scratchpad
            if "updated_scratchpad" in folded_output:
                scratchpad = folded_output["updated_scratchpad"]
                if self.config.get("verbose", False):
                    self.console.log(
                        f"Updated scratchpad for fold {current_fold}: {scratchpad}"
                    )
                del folded_output["updated_scratchpad"]

            current_output = folded_output

        return current_output, prompts, total_cost

    def validation_fn(self, response: Dict[str, Any]):
        output = self.runner.api.parse_llm_response(
            response,
            schema=self.config["output"]["schema"],
        )[0]
        if self.runner.api.validate_output(self.config, output, self.console):
            return output, True
        return output, False

    def _increment_fold(
        self,
        key: Tuple,
        batch: List[Dict],
        current_output: Optional[Dict],
        scratchpad: Optional[str] = None,
    ) -> Tuple[Optional[Dict], str, float]:
        """
        Perform an incremental fold operation on a batch of items.

        This method folds a batch of items into the current output using the fold prompt.

        Args:
            key (Tuple): The reduce key tuple for the group.
            batch (List[Dict]): The batch of items to be folded.
            current_output (Optional[Dict]): The current accumulated output, if any.
            scratchpad (Optional[str]): The scratchpad to use for the fold operation.
        Returns:
            Tuple[Optional[Dict], str, float]: A tuple containing the folded output (or None if processing failed),
            the prompt used, and the cost of the fold operation.
        """
        if current_output is None:
            return self._batch_reduce(key, batch, scratchpad)

        start_time = time.time()
        fold_prompt = strict_render(
            self.config["fold_prompt"],
            {
                "inputs": batch,
                "output": current_output,
                "reduce_key": dict(zip(self.config["reduce_key"], key)),
            },
        )

        response = self.runner.api.call_llm(
            self.config.get("model", self.default_model),
            "reduce",
            [{"role": "user", "content": fold_prompt}],
            self.config["output"]["schema"],
            scratchpad=scratchpad,
            timeout_seconds=self.config.get("timeout", 120),
            max_retries_per_timeout=self.config.get("max_retries_per_timeout", 2),
            validation_config=(
                {
                    "num_retries": self.num_retries_on_validate_failure,
                    "val_rule": self.config.get("validate", []),
                    "validation_fn": self.validation_fn,
                }
                if self.config.get("validate", None)
                else None
            ),
            bypass_cache=self.config.get("bypass_cache", False),
            verbose=self.config.get("verbose", False),
            litellm_completion_kwargs=self.config.get("litellm_completion_kwargs", {}),
        )

        end_time = time.time()
        self._update_fold_time(end_time - start_time)

        if response.validated:
            folded_output = self.runner.api.parse_llm_response(
                response.response,
                schema=self.config["output"]["schema"],
                manually_fix_errors=self.manually_fix_errors,
            )[0]

            folded_output.update(dict(zip(self.config["reduce_key"], key)))
            fold_cost = response.total_cost

            return folded_output, fold_prompt, fold_cost

        return None, fold_prompt, fold_cost

    def _merge_results(
        self, key: Tuple, outputs: List[Dict]
    ) -> Tuple[Optional[Dict], str, float]:
        """
        Merge multiple outputs into a single result.

        This method merges a list of outputs using the merge prompt.

        Args:
            key (Tuple): The reduce key tuple for the group.
            outputs (List[Dict]): The list of outputs to be merged.

        Returns:
            Tuple[Optional[Dict], str, float]: A tuple containing the merged output (or None if processing failed),
            the prompt used, and the cost of the merge operation.
        """
        start_time = time.time()
        merge_prompt = strict_render(
            self.config["merge_prompt"],
            {
                "outputs": outputs,
                "reduce_key": dict(zip(self.config["reduce_key"], key)),
            },
        )
        response = self.runner.api.call_llm(
            self.config.get("model", self.default_model),
            "merge",
            [{"role": "user", "content": merge_prompt}],
            self.config["output"]["schema"],
            timeout_seconds=self.config.get("timeout", 120),
            max_retries_per_timeout=self.config.get("max_retries_per_timeout", 2),
            validation_config=(
                {
                    "num_retries": self.num_retries_on_validate_failure,
                    "val_rule": self.config.get("validate", []),
                    "validation_fn": self.validation_fn,
                }
                if self.config.get("validate", None)
                else None
            ),
            bypass_cache=self.config.get("bypass_cache", False),
            verbose=self.config.get("verbose", False),
            litellm_completion_kwargs=self.config.get("litellm_completion_kwargs", {}),
        )

        end_time = time.time()
        self._update_merge_time(end_time - start_time)

        if response.validated:
            merged_output = self.runner.api.parse_llm_response(
                response.response,
                schema=self.config["output"]["schema"],
                manually_fix_errors=self.manually_fix_errors,
            )[0]
            merged_output.update(dict(zip(self.config["reduce_key"], key)))
            merge_cost = response.total_cost
            return merged_output, merge_prompt, merge_cost

        return None, merge_prompt, merge_cost

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
    ) -> Tuple[Optional[Dict], str, float]:
        """
        Perform a batch reduce operation on a group of items.

        This method reduces a group of items into a single output using the reduce prompt.

        Args:
            key (Tuple): The reduce key tuple for the group.
            group_list (List[Dict]): The list of items to be reduced.
            scratchpad (Optional[str]): The scratchpad to use for the reduce operation.
        Returns:
            Tuple[Optional[Dict], str, float]: A tuple containing the reduced output (or None if processing failed),
            the prompt used, and the cost of the reduce operation.
        """
        prompt = strict_render(
            self.config["prompt"],
            {
                "reduce_key": dict(zip(self.config["reduce_key"], key)),
                "inputs": group_list,
            },
        )
        item_cost = 0

        response = self.runner.api.call_llm(
            self.config.get("model", self.default_model),
            "reduce",
            [{"role": "user", "content": prompt}],
            self.config["output"]["schema"],
            scratchpad=scratchpad,
            timeout_seconds=self.config.get("timeout", 120),
            max_retries_per_timeout=self.config.get("max_retries_per_timeout", 2),
            bypass_cache=self.config.get("bypass_cache", False),
            validation_config=(
                {
                    "num_retries": self.num_retries_on_validate_failure,
                    "val_rule": self.config.get("validate", []),
                    "validation_fn": self.validation_fn,
                }
                if self.config.get("validate", None)
                else None
            ),
            gleaning_config=self.config.get("gleaning", None),
            verbose=self.config.get("verbose", False),
            litellm_completion_kwargs=self.config.get("litellm_completion_kwargs", {}),
        )

        item_cost += response.total_cost

        if response.validated:
            output = self.runner.api.parse_llm_response(
                response.response,
                schema=self.config["output"]["schema"],
                manually_fix_errors=self.manually_fix_errors,
            )[0]
            output.update(dict(zip(self.config["reduce_key"], key)))

            return output, prompt, item_cost
        return None, prompt, item_cost
