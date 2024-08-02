from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from jinja2 import Template
from collections import defaultdict
import json
from motion.operations.base import BaseOperation
from motion.operations.utils import call_llm, parse_llm_response, embedding
from motion.operations.utils import validate_output
from litellm import completion_cost
from sklearn.metrics.pairwise import cosine_similarity
import jinja2


def compare_pair(
    comparison_prompt: str, model: str, item1: Dict, item2: Dict
) -> Tuple[bool, float]:
    prompt_template = Template(comparison_prompt)
    prompt = prompt_template.render(input1=item1, input2=item2)
    response = call_llm(
        model,
        "compare",
        prompt,
        {"is_match": "bool"},
    )
    output = parse_llm_response(response)[0]
    return output["is_match"], completion_cost(response)


class ResolveOperation(BaseOperation):
    def syntax_check(self) -> None:
        required_keys = ["comparison_prompt", "output"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(
                    f"Missing required key '{key}' in ResolveOperation configuration"
                )

        if "schema" not in self.config["output"]:
            raise ValueError("Missing 'schema' in 'output' configuration")

        if not isinstance(self.config["output"]["schema"], dict):
            raise TypeError("'schema' in 'output' configuration must be a dictionary")

        if not self.config["output"]["schema"]:
            raise ValueError("'schema' in 'output' configuration cannot be empty")

        # Check if the comparison_prompt is a valid Jinja2 template
        try:
            comparison_template = Template(self.config["comparison_prompt"])
            comparison_vars = comparison_template.environment.parse(
                self.config["comparison_prompt"]
            ).find_all(jinja2.nodes.Name)
            comparison_var_names = {var.name for var in comparison_vars}
            if (
                "input1" not in comparison_var_names
                or "input2" not in comparison_var_names
            ):
                raise ValueError(
                    "'comparison_prompt' must contain both 'input1' and 'input2' variables"
                )

            if "resolution_prompt" in self.config:
                reduction_template = Template(self.config["resolution_prompt"])
                reduction_vars = reduction_template.environment.parse(
                    self.config["resolution_prompt"]
                ).find_all(jinja2.nodes.Name)
                reduction_var_names = {var.name for var in reduction_vars}
                if "matched_entries" not in reduction_var_names:
                    raise ValueError(
                        "'resolution_prompt' must contain 'matched_entries' variable"
                    )
        except Exception as e:
            raise ValueError(f"Invalid Jinja2 template: {str(e)}")

        # Check if the model is specified (optional)
        if "model" in self.config and not isinstance(self.config["model"], str):
            raise TypeError("'model' in configuration must be a string")

        # Check blocking_keys (optional)
        if "blocking_keys" in self.config:
            if not isinstance(self.config["blocking_keys"], list):
                raise TypeError("'blocking_keys' must be a list")
            if not all(isinstance(key, str) for key in self.config["blocking_keys"]):
                raise TypeError("All items in 'blocking_keys' must be strings")

        # Check blocking_threshold (optional)
        if "blocking_threshold" in self.config:
            if not isinstance(self.config["blocking_threshold"], (int, float)):
                raise TypeError("'blocking_threshold' must be a number")
            if not 0 <= self.config["blocking_threshold"] <= 1:
                raise ValueError("'blocking_threshold' must be between 0 and 1")

        # Check blocking_conditions (optional)
        if "blocking_conditions" in self.config:
            if not isinstance(self.config["blocking_conditions"], list):
                raise TypeError("'blocking_conditions' must be a list")
            if not all(
                isinstance(cond, str) for cond in self.config["blocking_conditions"]
            ):
                raise TypeError("All items in 'blocking_conditions' must be strings")

        # Check if input schema is provided and valid (optional)
        if "input" in self.config:
            if "schema" not in self.config["input"]:
                raise ValueError("Missing 'schema' in 'input' configuration")
            if not isinstance(self.config["input"]["schema"], dict):
                raise TypeError(
                    "'schema' in 'input' configuration must be a dictionary"
                )

    def execute(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        blocking_keys = self.config.get("blocking_keys", [])
        blocking_threshold = self.config.get("blocking_threshold")
        blocking_conditions = self.config.get("blocking_conditions", [])
        input_schema = self.config.get("input", {}).get("schema", {})
        total_cost = 0

        if len(input_data) == 0:
            return [], 0

        def is_match(item1: Dict[str, Any], item2: Dict[str, Any]) -> bool:
            return any(
                eval(condition, {"input1": item1, "input2": item2})
                for condition in blocking_conditions
            )

        # Calculate embeddings if blocking_threshold is set
        embeddings = None
        if blocking_threshold is not None:
            embedding_model = self.config.get("embedding_model", self.default_model)

            def get_embeddings_batch(
                items: List[Dict[str, Any]]
            ) -> List[Tuple[List[float], float]]:
                texts = [
                    " ".join(str(item[key]) for key in blocking_keys if key in item)
                    for item in items
                ]
                response = embedding(model=embedding_model, input=texts)
                return [
                    (data["embedding"], completion_cost(response))
                    for data in response["data"]
                ]

            embeddings = []
            costs = []
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                for i in range(
                    0, len(input_data), self.config.get("embedding_batch_size", 1000)
                ):
                    batch = input_data[
                        i : i + self.config.get("embedding_batch_size", 1000)
                    ]
                    batch_results = list(executor.map(get_embeddings_batch, [batch]))
                    for result in batch_results:
                        embeddings.extend([r[0] for r in result])
                        costs.extend([r[1] for r in result])

                total_cost += sum(costs)

        # Initialize clusters
        clusters = [{i} for i in range(len(input_data))]
        cluster_map = {i: i for i in range(len(input_data))}

        def find_cluster(item):
            while item != cluster_map[item]:
                cluster_map[item] = cluster_map[cluster_map[item]]
                item = cluster_map[item]
            return item

        def merge_clusters(item1, item2):
            root1, root2 = find_cluster(item1), find_cluster(item2)
            if root1 != root2:
                if len(clusters[root1]) < len(clusters[root2]):
                    root1, root2 = root2, root1
                clusters[root1] |= clusters[root2]
                cluster_map[root2] = root1
                clusters[root2] = set()

        # Generate all pairs to compare
        # TODO: virtualize this if possible
        all_pairs = [
            (i, j)
            for i in range(len(input_data))
            for j in range(i + 1, len(input_data))
        ]

        # Filter pairs based on blocking rules
        def should_compare(pair):
            i, j = pair
            if find_cluster(i) == find_cluster(j):
                return False
            if blocking_threshold is not None:
                if (
                    cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                    < blocking_threshold
                ):
                    return False
            if blocking_conditions:
                if not is_match(input_data[i], input_data[j]):
                    return False
            return True

        filtered_pairs = list(filter(should_compare, all_pairs))

        # Calculate and print statistics
        total_possible_comparisons = len(input_data) * (len(input_data) - 1) // 2
        comparisons_made = len(filtered_pairs)
        comparisons_saved = total_possible_comparisons - comparisons_made
        self.console.print(
            f"[green]Comparisons saved by blocking: {comparisons_saved} "
            f"({(comparisons_saved / total_possible_comparisons) * 100:.2f}%)[/green]"
        )

        # Compare pairs and update clusters in real-time
        batch_size = self.config.get("compare_batch_size", 100)
        pair_costs = 0

        for i in tqdm(
            range(0, len(filtered_pairs), batch_size),
            desc=f"Processing batches of {batch_size} LLM comparisons",
        ):
            batch = filtered_pairs[i : i + batch_size]

            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                future_to_pair = {
                    executor.submit(
                        compare_pair,
                        self.config["comparison_prompt"],
                        self.config.get("comparison_model", self.default_model),
                        input_data[pair[0]],
                        input_data[pair[1]],
                    ): pair
                    for pair in batch
                }

                for future in as_completed(future_to_pair):
                    pair = future_to_pair[future]
                    is_match, cost = future.result()
                    pair_costs += cost
                    if is_match:
                        merge_clusters(pair[0], pair[1])

        total_cost += pair_costs

        # Collect final clusters
        final_clusters = [cluster for cluster in clusters if cluster]

        # Process each cluster
        results = []

        def process_cluster(cluster):
            if len(cluster) > 1:
                cluster_items = [input_data[i] for i in cluster]
                reduction_template = Template(self.config["resolution_prompt"])
                if input_schema:
                    cluster_items = [
                        {k: item[k] for k in input_schema.keys() if k in item}
                        for item in cluster_items
                    ]

                resolution_prompt = reduction_template.render(
                    matched_entries=cluster_items
                )
                reduction_response = call_llm(
                    self.config.get("resolution_model", self.default_model),
                    "reduce",
                    resolution_prompt,
                    self.config["output"]["schema"],
                )
                reduction_output = parse_llm_response(reduction_response)[0]
                reduction_cost = completion_cost(reduction_response)

                if validate_output(self.config, reduction_output, self.console):
                    return (
                        [
                            {
                                **item,
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
                return [input_data[list(cluster)[0]]], 0

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [
                executor.submit(process_cluster, cluster) for cluster in final_clusters
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Determining resolved key for each group of equivalent keys",
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
        self.console.print(f"Self-join selectivity: {true_match_selectivity:.4f}")

        return results, total_cost
