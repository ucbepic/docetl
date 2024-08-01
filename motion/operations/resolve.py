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

        # If there's no blocking, put all elements in one cluster
        if not blocking_keys and blocking_threshold is None and not blocking_conditions:
            clusters = {0: list(range(len(input_data)))}
        else:

            def is_match(item1: Dict[str, Any], item2: Dict[str, Any]) -> bool:
                return any(
                    eval(condition, {"input1": item1, "input2": item2})
                    for condition in blocking_conditions
                )

            # Initial clustering
            clusters = {}
            for i, item in enumerate(input_data):
                matched = False
                for rep, cluster in clusters.items():
                    if is_match(item, input_data[rep]):
                        cluster.append(i)
                        matched = True
                        break
                if not matched:
                    clusters[i] = [i]

            if blocking_threshold is not None:
                embedding_model = self.config.get("embedding_model", self.default_model)

                def get_embedding(item: Dict[str, Any]) -> Tuple[List[float], float]:
                    text = " ".join(
                        str(item[key]) for key in blocking_keys if key in item
                    )
                    response = embedding(model=embedding_model, input=[text])
                    return response["data"][0]["embedding"], completion_cost(response)

                with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                    embeddings = list(executor.map(get_embedding, input_data))
                    embeddings, costs = zip(*embeddings)
                    total_cost += sum(costs)

                # Additional clustering based on embeddings
                for i, item in enumerate(input_data):
                    for rep, cluster in list(clusters.items()):
                        if (
                            i not in cluster
                            and cosine_similarity([embeddings[i]], [embeddings[rep]])[
                                0
                            ][0]
                            >= blocking_threshold
                        ):
                            clusters[rep].append(i)

        # Pairwise comparisons within clusters
        true_matches = {}
        pair_costs = 0

        def compare_pair(item1: Dict, item2: Dict) -> Tuple[bool, float]:
            prompt_template = Template(self.config["comparison_prompt"])
            prompt = prompt_template.render(input1=item1, input2=item2)
            response = call_llm(
                self.config.get("comparison_model", self.default_model),
                "compare",
                prompt,
                {"is_match": "bool"},
            )
            output = parse_llm_response(response)[0]
            return output["is_match"], completion_cost(response)

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            future_to_pair = {}
            for cluster in clusters.values():
                cluster_items = [input_data[i] for i in cluster]
                for i, item1 in enumerate(cluster_items):
                    for j, item2 in enumerate(cluster_items):
                        if i < j:
                            future = executor.submit(compare_pair, item1, item2)
                            future_to_pair[future] = (cluster[i], cluster[j])

            total_pairs = len(future_to_pair)
            for future in tqdm(
                as_completed(future_to_pair), total=total_pairs, desc="Comparing pairs"
            ):
                pair = future_to_pair[future]
                is_match, cost = future.result()
                pair_costs += cost
                if is_match:
                    if pair[0] not in true_matches:
                        true_matches[pair[0]] = set()
                    if pair[1] not in true_matches:
                        true_matches[pair[1]] = set()
                    true_matches[pair[0]].add(pair[1])
                    true_matches[pair[1]].add(pair[0])

        # Calculate and print the true match selectivity
        n = len(input_data)
        total_possible_pairs = n * (n - 1) // 2
        true_match_count = sum(len(matches) for matches in true_matches.values()) // 2
        true_match_selectivity = true_match_count / total_possible_pairs
        self.console.print(f"Self-join selectivity: {true_match_selectivity:.4f}")
        total_cost += pair_costs

        # Group true matches into sub-clusters
        sub_clusters = []
        processed = set()
        for i, matches in true_matches.items():
            if i not in processed:
                sub_cluster = {i} | matches
                for j in matches:
                    sub_cluster |= true_matches.get(j, set())
                sub_clusters.append(list(sub_cluster))
                processed |= sub_cluster

        # Add items that didn't match anything as their own clusters
        for i in range(len(input_data)):
            if i not in processed:
                sub_clusters.append([i])

        # Process each sub-cluster
        results = []

        def process_sub_cluster(sub_cluster):
            if len(sub_cluster) > 1:
                true_match_items = [input_data[i] for i in sub_cluster]
                reduction_template = Template(self.config["resolution_prompt"])
                if input_schema:
                    true_match_items = [
                        {k: item[k] for k in input_schema.keys() if k in item}
                        for item in true_match_items
                    ]

                resolution_prompt = reduction_template.render(
                    matched_entries=true_match_items
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
                            for item in [input_data[i] for i in sub_cluster]
                        ],
                        reduction_cost,
                    )
                return [], reduction_cost
            else:
                return [input_data[sub_cluster[0]]], 0

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [
                executor.submit(process_sub_cluster, sub_cluster)
                for sub_cluster in sub_clusters
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing sub-clusters",
                leave=True,
            ):
                sub_results, sub_cost = future.result()
                results.extend(sub_results)
                total_cost += sub_cost

        return results, total_cost
