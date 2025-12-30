"""
The `ResolveOperation` class is a subclass of `BaseOperation` that performs a resolution operation on a dataset. It uses a combination of blocking techniques and LLM-based comparisons to efficiently identify and resolve duplicate or related entries within the dataset.
"""

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import jinja2
from jinja2 import Template
from litellm import model_cost
from pydantic import Field, ValidationInfo, field_validator, model_validator

from docetl.operations.base import BaseOperation
from docetl.operations.utils import RichLoopBar, rich_as_completed, strict_render
from docetl.operations.utils.blocking import RuntimeBlockingOptimizer
from docetl.utils import (
    completion_cost,
    extract_jinja_variables,
    has_jinja_syntax,
    prompt_user_for_non_jinja_confirmation,
)


def find_cluster(item, cluster_map):
    while item != cluster_map[item]:
        cluster_map[item] = cluster_map[cluster_map[item]]
        item = cluster_map[item]
    return item


class ResolveOperation(BaseOperation):
    class schema(BaseOperation.schema):
        type: str = "resolve"
        comparison_prompt: str
        resolution_prompt: str | None = None
        output: dict[str, Any] | None = None
        embedding_model: str | None = None
        resolution_model: str | None = None
        comparison_model: str | None = None
        blocking_keys: list[str] | None = None
        blocking_threshold: float | None = Field(None, ge=0, le=1)
        blocking_target_recall: float | None = Field(None, ge=0, le=1)
        blocking_conditions: list[str] | None = None
        input: dict[str, Any] | None = None
        embedding_batch_size: int | None = Field(None, gt=0)
        compare_batch_size: int | None = Field(None, gt=0)
        limit_comparisons: int | None = Field(None, gt=0)
        optimize: bool | None = None
        timeout: int | None = Field(None, gt=0)
        litellm_completion_kwargs: dict[str, Any] = Field(default_factory=dict)
        enable_observability: bool = False

        @field_validator("comparison_prompt")
        def validate_comparison_prompt(cls, v):
            if v is not None:
                # Check if it has Jinja syntax
                if not has_jinja_syntax(v):
                    # This will be handled during initialization with user confirmation
                    return v
                try:
                    comparison_template = Template(v)
                    comparison_vars = comparison_template.environment.parse(v).find_all(
                        jinja2.nodes.Name
                    )
                    comparison_var_names = {var.name for var in comparison_vars}
                    if (
                        "input1" not in comparison_var_names
                        or "input2" not in comparison_var_names
                    ):
                        raise ValueError(
                            f"'comparison_prompt' must contain both 'input1' and 'input2' variables. {v}"
                        )
                except Exception as e:
                    raise ValueError(
                        f"Invalid Jinja2 template in 'comparison_prompt': {str(e)}"
                    )
            return v

        @field_validator("resolution_prompt")
        def validate_resolution_prompt(cls, v):
            if v is not None:
                # Check if it has Jinja syntax
                if not has_jinja_syntax(v):
                    # This will be handled during initialization with user confirmation
                    return v
                try:
                    reduction_template = Template(v)
                    reduction_vars = reduction_template.environment.parse(v).find_all(
                        jinja2.nodes.Name
                    )
                    reduction_var_names = {var.name for var in reduction_vars}
                    if "inputs" not in reduction_var_names:
                        raise ValueError(
                            "'resolution_prompt' must contain 'inputs' variable"
                        )
                except Exception as e:
                    raise ValueError(
                        f"Invalid Jinja2 template in 'resolution_prompt': {str(e)}"
                    )
            return v

        @field_validator("input")
        def validate_input_schema(cls, v):
            if v is not None:
                if "schema" not in v:
                    raise ValueError("Missing 'schema' in 'input' configuration")
                if not isinstance(v["schema"], dict):
                    raise TypeError(
                        "'schema' in 'input' configuration must be a dictionary"
                    )
            return v

        @model_validator(mode="after")
        def validate_output_schema(self, info: ValidationInfo):
            # Skip validation if we're using from dataframe accessors
            if isinstance(info.context, dict) and info.context.get(
                "_from_df_accessors"
            ):
                return self

            if self.output is None:
                raise ValueError(
                    "Missing required key 'output' in ResolveOperation configuration"
                )

            if "schema" not in self.output:
                raise ValueError("Missing 'schema' in 'output' configuration")

            if not isinstance(self.output["schema"], dict):
                raise TypeError(
                    "'schema' in 'output' configuration must be a dictionary"
                )

            if not self.output["schema"]:
                raise ValueError("'schema' in 'output' configuration cannot be empty")

            return self

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Check for non-Jinja prompts and prompt user for confirmation
        if "comparison_prompt" in self.config and not has_jinja_syntax(
            self.config["comparison_prompt"]
        ):
            if not prompt_user_for_non_jinja_confirmation(
                self.config["comparison_prompt"],
                self.config["name"],
                "comparison_prompt",
            ):
                raise ValueError(
                    f"Operation '{self.config['name']}' cancelled by user. Please add Jinja2 template syntax to your comparison_prompt."
                )
            # Mark that we need to append document statement
            # Note: comparison_prompt uses input1 and input2, so we'll handle it specially in strict_render
            self.config["_append_document_to_comparison_prompt"] = True
        if "resolution_prompt" in self.config and not has_jinja_syntax(
            self.config["resolution_prompt"]
        ):
            if not prompt_user_for_non_jinja_confirmation(
                self.config["resolution_prompt"],
                self.config["name"],
                "resolution_prompt",
            ):
                raise ValueError(
                    f"Operation '{self.config['name']}' cancelled by user. Please add Jinja2 template syntax to your resolution_prompt."
                )
            # Mark that we need to append document statement (resolution uses inputs)
            self.config["_append_document_to_resolution_prompt"] = True
            self.config["_is_reduce_operation"] = True

    def compare_pair(
        self,
        comparison_prompt: str,
        model: str,
        item1: dict,
        item2: dict,
        blocking_keys: list[str] = [],
        timeout_seconds: int = 120,
        max_retries_per_timeout: int = 2,
    ) -> tuple[bool, float, str]:
        """
        Compares two items using an LLM model to determine if they match.

        Args:
            comparison_prompt (str): The prompt template for comparison.
            model (str): The LLM model to use for comparison.
            item1 (dict): The first item to compare.
            item2 (dict): The second item to compare.

        Returns:
            tuple[bool, float, str]: A tuple containing a boolean indicating whether the items match, the cost of the comparison, and the prompt.
        """
        if blocking_keys:
            if all(
                key in item1
                and key in item2
                and str(item1[key]).lower() == str(item2[key]).lower()
                for key in blocking_keys
            ):
                return True, 0, ""

        prompt = strict_render(comparison_prompt, {"input1": item1, "input2": item2})
        response = self.runner.api.call_llm(
            model,
            "compare",
            [{"role": "user", "content": prompt}],
            {"is_match": "bool"},
            timeout_seconds=timeout_seconds,
            max_retries_per_timeout=max_retries_per_timeout,
            bypass_cache=self.config.get("bypass_cache", self.bypass_cache),
            litellm_completion_kwargs=self.config.get("litellm_completion_kwargs", {}),
            op_config=self.config,
        )
        output = self.runner.api.parse_llm_response(
            response.response,
            {"is_match": "bool"},
        )[0]

        return output["is_match"], response.total_cost, prompt

    def syntax_check(self) -> None:
        context = {"_from_df_accessors": self.runner._from_df_accessors}
        super().syntax_check(context)

    def validation_fn(self, response: dict[str, Any]):
        output = self.runner.api.parse_llm_response(
            response,
            schema=self.config["output"]["schema"],
        )[0]
        if self.runner.api.validate_output(self.config, output, self.console):
            return output, True
        return output, False

    def execute(self, input_data: list[dict]) -> tuple[list[dict], float]:
        """
        Executes the resolve operation on the provided dataset.

        Args:
            input_data (list[dict]): The dataset to resolve.

        Returns:
            tuple[list[dict], float]: A tuple containing the resolved results and the total cost of the operation.

        This method performs the following steps:
        1. Initial blocking based on specified conditions and/or embedding similarity
        2. Pairwise comparison of potentially matching entries using LLM
        3. Clustering of matched entries
        4. Resolution of each cluster into a single entry (if applicable)
        5. Result aggregation and validation

        The method also calculates and logs statistics such as comparisons saved by blocking and self-join selectivity.
        """
        if len(input_data) == 0:
            return [], 0

        # Initialize observability data for all items at the start
        if self.config.get("enable_observability", False):
            observability_key = f"_observability_{self.config['name']}"
            for item in input_data:
                if observability_key not in item:
                    item[observability_key] = {
                        "comparison_prompts": [],
                        "resolution_prompt": None,
                    }

        blocking_keys = self.config.get("blocking_keys", [])
        blocking_threshold = self.config.get("blocking_threshold")
        blocking_conditions = self.config.get("blocking_conditions", [])
        limit_comparisons = self.config.get("limit_comparisons")
        total_cost = 0
        if self.status:
            self.status.stop()

        # Track pre-computed embeddings from auto-optimization
        precomputed_embeddings = None

        # Auto-compute blocking threshold if no blocking configuration is provided
        if not blocking_threshold and not blocking_conditions and not limit_comparisons:
            # Get target recall from operation config (default 0.95)
            target_recall = self.config.get("blocking_target_recall", 0.95)
            self.console.log(
                f"[yellow]No blocking configuration. Auto-computing threshold (target recall: {target_recall:.0%})...[/yellow]"
            )
            # Determine blocking keys if not set
            auto_blocking_keys = blocking_keys if blocking_keys else None
            if not auto_blocking_keys:
                prompt_template = self.config.get("comparison_prompt", "")
                prompt_vars = extract_jinja_variables(prompt_template)
                prompt_vars = [
                    var
                    for var in prompt_vars
                    if var not in ["input", "input1", "input2"]
                ]
                auto_blocking_keys = list(
                    set([var.split(".")[-1] for var in prompt_vars])
                )
            if not auto_blocking_keys:
                auto_blocking_keys = list(input_data[0].keys())
            blocking_keys = auto_blocking_keys

            # Create comparison function for threshold optimization
            def compare_fn_for_optimization(item1, item2):
                return self.compare_pair(
                    self.config["comparison_prompt"],
                    self.config.get("comparison_model", self.default_model),
                    item1,
                    item2,
                    blocking_keys=[],  # Don't use key-based shortcut during optimization
                    timeout_seconds=self.config.get("timeout", 120),
                    max_retries_per_timeout=self.config.get(
                        "max_retries_per_timeout", 2
                    ),
                )

            # Run threshold optimization
            optimizer = RuntimeBlockingOptimizer(
                runner=self.runner,
                config=self.config,
                default_model=self.default_model,
                max_threads=self.max_threads,
                console=self.console,
                target_recall=target_recall,
                sample_size=min(100, len(input_data) * (len(input_data) - 1) // 4),
            )
            blocking_threshold, precomputed_embeddings, optimization_cost = (
                optimizer.optimize_resolve(
                    input_data,
                    compare_fn_for_optimization,
                    blocking_keys=blocking_keys,
                )
            )
            total_cost += optimization_cost

        input_schema = self.config.get("input", {}).get("schema", {})
        if not blocking_keys:
            # Set them to all keys in the input data
            blocking_keys = list(input_data[0].keys())

        def is_match(item1: dict[str, Any], item2: dict[str, Any]) -> bool:
            return any(
                eval(condition, {"input1": item1, "input2": item2})
                for condition in blocking_conditions
            )

        # Calculate embeddings if blocking_threshold is set
        embeddings = None
        if blocking_threshold is not None:
            # Use precomputed embeddings if available from auto-optimization
            if precomputed_embeddings is not None:
                embeddings = precomputed_embeddings
            else:
                self.console.log(
                    f"[cyan]Creating embeddings for {len(input_data)} items...[/cyan]"
                )
                embedding_model = self.config.get(
                    "embedding_model", "text-embedding-3-small"
                )
                model_input_context_length = model_cost.get(embedding_model, {}).get(
                    "max_input_tokens", 8192
                )
                batch_size = self.config.get("embedding_batch_size", 1000)
                embeddings = []
                embedding_cost = 0.0
                num_batches = (len(input_data) + batch_size - 1) // batch_size

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(input_data))
                    batch = input_data[start_idx:end_idx]

                    if num_batches > 1:
                        self.console.log(
                            f"[dim]Creating embeddings: batch {batch_idx + 1}/{num_batches} "
                            f"({end_idx}/{len(input_data)} items)[/dim]"
                        )

                    texts = [
                        " ".join(
                            str(item[key]) for key in blocking_keys if key in item
                        )[: model_input_context_length * 3]
                        for item in batch
                    ]
                    response = self.runner.api.gen_embedding(
                        model=embedding_model, input=texts
                    )
                    embeddings.extend([data["embedding"] for data in response["data"]])
                    embedding_cost += completion_cost(response)

                total_cost += embedding_cost

        # Build a mapping of blocking key values to indices
        # This is used later for cluster merging (when two items match, merge all items sharing their key values)
        value_to_indices: dict[tuple[str, ...], list[int]] = {}
        for i, item in enumerate(input_data):
            key = tuple(str(item.get(k, "")) for k in blocking_keys)
            if key not in value_to_indices:
                value_to_indices[key] = []
            value_to_indices[key].append(i)

        # Total number of pairs to potentially compare
        n = len(input_data)
        total_pairs = n * (n - 1) // 2

        # Apply code-based blocking conditions (check all pairs)
        code_blocked_pairs = []
        if blocking_conditions:
            for i in range(n):
                for j in range(i + 1, n):
                    if is_match(input_data[i], input_data[j]):
                        code_blocked_pairs.append((i, j))

        # Apply cosine similarity blocking if threshold is specified
        embedding_blocked_pairs = []
        if blocking_threshold is not None and embeddings is not None:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity

            similarity_matrix = cosine_similarity(embeddings)
            code_blocked_set = set(code_blocked_pairs)

            # Use numpy to efficiently find all pairs above threshold
            i_indices, j_indices = np.triu_indices(n, k=1)
            similarities = similarity_matrix[i_indices, j_indices]
            above_threshold_mask = similarities >= blocking_threshold

            # Get pairs above threshold
            above_threshold_i = i_indices[above_threshold_mask]
            above_threshold_j = j_indices[above_threshold_mask]

            # Filter out pairs already in code_blocked_set
            embedding_blocked_pairs = [
                (int(i), int(j))
                for i, j in zip(above_threshold_i, above_threshold_j)
                if (i, j) not in code_blocked_set
            ]

        # Combine pairs from both blocking methods
        all_blocked_pairs = code_blocked_pairs + embedding_blocked_pairs

        # If no blocking was applied, compare all pairs
        if not blocking_conditions and blocking_threshold is None:
            all_blocked_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        # Apply limit_comparisons with prioritization
        if limit_comparisons is not None and len(all_blocked_pairs) > limit_comparisons:
            # Prioritize code-based pairs, then sample from embedding pairs if needed
            if len(code_blocked_pairs) >= limit_comparisons:
                # If we have enough code-based pairs, just sample from those
                blocked_pairs = random.sample(code_blocked_pairs, limit_comparisons)
                self.console.log(
                    f"Using {limit_comparisons} code-based pairs (had {len(code_blocked_pairs)} available)"
                )
            else:
                # Take all code-based pairs + sample from embedding pairs
                remaining_slots = limit_comparisons - len(code_blocked_pairs)
                sampled_embedding_pairs = random.sample(
                    embedding_blocked_pairs,
                    min(remaining_slots, len(embedding_blocked_pairs)),
                )
                blocked_pairs = code_blocked_pairs + sampled_embedding_pairs
                self.console.log(
                    f"Using {len(code_blocked_pairs)} code-based + {len(sampled_embedding_pairs)} embedding-based pairs "
                    f"(total: {len(blocked_pairs)})"
                )
        else:
            blocked_pairs = all_blocked_pairs
            if len(code_blocked_pairs) > 0 and len(embedding_blocked_pairs) > 0:
                self.console.log(
                    f"Using all {len(code_blocked_pairs)} code-based + {len(embedding_blocked_pairs)} embedding-based pairs"
                )

        # Initialize clusters with all indices
        clusters = [{i} for i in range(len(input_data))]
        cluster_map = {i: i for i in range(len(input_data))}

        # Modified merge_clusters to handle all indices with the same value

        def merge_clusters(item1: int, item2: int) -> None:
            root1, root2 = find_cluster(item1, cluster_map), find_cluster(
                item2, cluster_map
            )
            if root1 != root2:
                if len(clusters[root1]) < len(clusters[root2]):
                    root1, root2 = root2, root1
                clusters[root1] |= clusters[root2]
                cluster_map[root2] = root1
                clusters[root2] = set()

                # Also merge all other indices that share the same values
                key1 = tuple(str(input_data[item1].get(k, "")) for k in blocking_keys)
                key2 = tuple(str(input_data[item2].get(k, "")) for k in blocking_keys)

                # Merge all indices with the same values
                for idx in value_to_indices.get(key1, []):
                    if idx != item1:
                        root_idx = find_cluster(idx, cluster_map)
                        if root_idx != root1:
                            clusters[root1] |= clusters[root_idx]
                            cluster_map[root_idx] = root1
                            clusters[root_idx] = set()

                for idx in value_to_indices.get(key2, []):
                    if idx != item2:
                        root_idx = find_cluster(idx, cluster_map)
                        if root_idx != root1:
                            clusters[root1] |= clusters[root_idx]
                            cluster_map[root_idx] = root1
                            clusters[root_idx] = set()

        # Compute an auto-batch size based on the number of comparisons
        def auto_batch() -> int:
            # Maximum batch size limit for 4o-mini model
            M = 500

            n = len(input_data)
            m = len(blocked_pairs)

            # https://www.wolframalpha.com/input?i=k%28k-1%29%2F2+%2B+%28n-k%29%28k-1%29+%3D+m%2C+solve+for+k
            # Two possible solutions for k:
            # k = -1/2 sqrt((1 - 2n)^2 - 8m) + n + 1/2
            # k = 1/2 (sqrt((1 - 2n)^2 - 8m) + 2n + 1)

            discriminant = (1 - 2 * n) ** 2 - 8 * m
            sqrt_discriminant = discriminant**0.5

            k1 = -0.5 * sqrt_discriminant + n + 0.5
            k2 = 0.5 * (sqrt_discriminant + 2 * n + 1)

            # Take the maximum viable solution
            k = max(k1, k2)
            return M if k < 0 else min(int(k), M)

        # Compare pairs and update clusters in real-time
        batch_size = self.config.get("compare_batch_size", auto_batch())

        # Log blocking summary
        total_possible_comparisons = len(input_data) * (len(input_data) - 1) // 2
        self.console.log(
            f"Comparing {len(blocked_pairs):,} pairs "
            f"({len(blocked_pairs)/total_possible_comparisons*100:.1f}% of {total_possible_comparisons:,} total, "
            f"batch size: {batch_size})"
        )
        pair_costs = 0

        pbar = RichLoopBar(
            range(0, len(blocked_pairs), batch_size),
            desc=f"Processing batches of {batch_size} LLM comparisons",
            console=self.console,
        )
        last_processed = 0
        for i in pbar:
            batch_end = last_processed + batch_size
            batch = blocked_pairs[last_processed:batch_end]
            # Filter pairs for the initial batch
            better_batch = [
                pair
                for pair in batch
                if find_cluster(pair[0], cluster_map) == pair[0]
                and find_cluster(pair[1], cluster_map) == pair[1]
            ]

            # Expand better_batch if it doesn’t reach batch_size
            while len(better_batch) < batch_size and batch_end < len(blocked_pairs):
                # Move batch_end forward by batch_size to get more pairs
                next_end = batch_end + batch_size
                next_batch = blocked_pairs[batch_end:next_end]

                better_batch.extend(
                    pair
                    for pair in next_batch
                    if find_cluster(pair[0], cluster_map) == pair[0]
                    and find_cluster(pair[1], cluster_map) == pair[1]
                )

                # Update batch_end to prevent overlapping in the next loop
                batch_end = next_end
            better_batch = better_batch[:batch_size]
            last_processed = batch_end
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                future_to_pair = {
                    executor.submit(
                        self.compare_pair,
                        self.config["comparison_prompt"],
                        self.config.get("comparison_model", self.default_model),
                        input_data[pair[0]],
                        input_data[pair[1]],
                        blocking_keys,
                        timeout_seconds=self.config.get("timeout", 120),
                        max_retries_per_timeout=self.config.get(
                            "max_retries_per_timeout", 2
                        ),
                    ): pair
                    for pair in better_batch
                }

                for future in as_completed(future_to_pair):
                    pair = future_to_pair[future]
                    is_match_result, cost, prompt = future.result()
                    pair_costs += cost
                    if is_match_result:
                        merge_clusters(pair[0], pair[1])

                    if self.config.get("enable_observability", False):
                        observability_key = f"_observability_{self.config['name']}"
                        for idx in (pair[0], pair[1]):
                            if observability_key not in input_data[idx]:
                                input_data[idx][observability_key] = {
                                    "comparison_prompts": [],
                                    "resolution_prompt": None,
                                }
                            input_data[idx][observability_key][
                                "comparison_prompts"
                            ].append(prompt)

        total_cost += pair_costs

        # Collect final clusters
        final_clusters = [cluster for cluster in clusters if cluster]

        # Process each cluster
        results = []

        def process_cluster(cluster):
            if len(cluster) > 1:
                cluster_items = [input_data[i] for i in cluster]
                if input_schema:
                    cluster_items = [
                        {k: item[k] for k in input_schema.keys() if k in item}
                        for item in cluster_items
                    ]

                resolution_prompt = strict_render(
                    self.config["resolution_prompt"], {"inputs": cluster_items}
                )
                reduction_response = self.runner.api.call_llm(
                    self.config.get("resolution_model", self.default_model),
                    "reduce",
                    [{"role": "user", "content": resolution_prompt}],
                    self.config["output"]["schema"],
                    timeout_seconds=self.config.get("timeout", 120),
                    max_retries_per_timeout=self.config.get(
                        "max_retries_per_timeout", 2
                    ),
                    bypass_cache=self.config.get("bypass_cache", self.bypass_cache),
                    validation_config=(
                        {
                            "val_rule": self.config.get("validate", []),
                            "validation_fn": self.validation_fn,
                        }
                        if self.config.get("validate", None)
                        else None
                    ),
                    litellm_completion_kwargs=self.config.get(
                        "litellm_completion_kwargs", {}
                    ),
                    op_config=self.config,
                )
                reduction_cost = reduction_response.total_cost

                if self.config.get("enable_observability", False):
                    for item in [input_data[i] for i in cluster]:
                        observability_key = f"_observability_{self.config['name']}"
                        if observability_key not in item:
                            item[observability_key] = {
                                "comparison_prompts": [],
                                "resolution_prompt": None,
                            }
                        item[observability_key]["resolution_prompt"] = resolution_prompt

                if reduction_response.validated:
                    reduction_output = self.runner.api.parse_llm_response(
                        reduction_response.response,
                        self.config["output"]["schema"],
                        manually_fix_errors=self.manually_fix_errors,
                    )[0]

                    # If the output is overwriting an existing key, we want to save the kv pairs
                    keys_in_output = [
                        k
                        for k in set(reduction_output.keys())
                        if k in cluster_items[0].keys()
                    ]

                    return (
                        [
                            {
                                **item,
                                f"_kv_pairs_preresolve_{self.config['name']}": {
                                    k: item[k] for k in keys_in_output
                                },
                                **{
                                    k: reduction_output[k]
                                    for k in self.config["output"]["schema"]
                                },
                            }
                            for item in [input_data[i] for i in cluster]
                        ],
                        reduction_cost,
                    )
                return [], reduction_cost
            else:
                # Set the output schema to be the keys found in the compare_prompt
                compare_prompt_keys = extract_jinja_variables(
                    self.config["comparison_prompt"]
                )
                # Get the set of keys in the compare_prompt
                compare_prompt_keys = set(
                    [
                        k.replace("input1.", "")
                        for k in compare_prompt_keys
                        if "input1" in k
                    ]
                )

                # For each key in the output schema, find the most similar key in the compare_prompt
                output_keys = set(self.config["output"]["schema"].keys())
                key_mapping = {}
                for output_key in output_keys:
                    best_match = None
                    best_score = 0
                    for compare_key in compare_prompt_keys:
                        score = sum(
                            c1 == c2 for c1, c2 in zip(output_key, compare_key)
                        ) / max(len(output_key), len(compare_key))
                        if score > best_score:
                            best_score = score
                            best_match = compare_key
                    key_mapping[output_key] = best_match

                # Create the result dictionary using the key mapping
                result = input_data[list(cluster)[0]].copy()
                result[f"_kv_pairs_preresolve_{self.config['name']}"] = {
                    ok: result[ck] for ok, ck in key_mapping.items() if ck in result
                }
                for output_key, compare_key in key_mapping.items():
                    if compare_key in input_data[list(cluster)[0]]:
                        result[output_key] = input_data[list(cluster)[0]][compare_key]
                    elif output_key in input_data[list(cluster)[0]]:
                        result[output_key] = input_data[list(cluster)[0]][output_key]
                    else:
                        result[output_key] = None  # or some default value

                return [result], 0

        # Calculate the number of records before and clusters after
        num_records_before = len(input_data)
        num_clusters_after = len(final_clusters)
        self.console.log(f"Number of keys before resolution: {num_records_before}")
        self.console.log(
            f"Number of distinct keys after resolution: {num_clusters_after}"
        )

        # If no resolution prompt is provided, we can skip the resolution phase
        # And simply select the most common value for each key
        if not self.config.get("resolution_prompt", None):
            for cluster in final_clusters:
                if len(cluster) > 1:
                    for key in self.config["output"]["keys"]:
                        most_common_value = max(
                            set(input_data[i][key] for i in cluster),
                            key=lambda x: sum(
                                1 for i in cluster if input_data[i][key] == x
                            ),
                        )
                        for i in cluster:
                            input_data[i][key] = most_common_value
            results = input_data
        else:
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                futures = [
                    executor.submit(process_cluster, cluster)
                    for cluster in final_clusters
                ]
                for future in rich_as_completed(
                    futures,
                    total=len(futures),
                    desc="Determining resolved key for each group of equivalent keys",
                    console=self.console,
                ):
                    cluster_results, cluster_cost = future.result()
                    results.extend(cluster_results)
                    total_cost += cluster_cost

        total_pairs = len(input_data) * (len(input_data) - 1) // 2
        true_match_count = sum(
            len(cluster) * (len(cluster) - 1) // 2
            for cluster in final_clusters
            if len(cluster) > 1
        )
        true_match_selectivity = (
            true_match_count / total_pairs if total_pairs > 0 else 0
        )
        self.console.log(f"Self-join selectivity: {true_match_selectivity:.4f}")

        if self.status:
            self.status.start()

        return results, total_cost
