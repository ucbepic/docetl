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


def compare_pair(
    comparison_prompt: str, model: str, item1: Dict, item2: Dict
) -> Tuple[bool, float]:
    prompt_template = Template(comparison_prompt)
    prompt = prompt_template.render(left=item1, right=item2)
    response = call_llm(
        model,
        "compare",
        prompt,
        {"is_match": "bool"},
    )
    output = parse_llm_response(response)[0]
    return output["is_match"], completion_cost(response)


class EquijoinOperation(BaseOperation):
    def syntax_check(self) -> None:
        required_keys = ["join_key", "comparison_prompt"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(
                    f"Missing required key '{key}' in EquijoinOperation configuration"
                )

        if (
            "left" not in self.config["join_key"]
            or "right" not in self.config["join_key"]
        ):
            raise ValueError("Both 'left' and 'right' must be specified in 'join_key'")

        if (
            "name" not in self.config["join_key"]["left"]
            or "name" not in self.config["join_key"]["right"]
        ):
            raise ValueError(
                "Both left and right join keys must have a 'name' specified"
            )

    def execute(
        self, left_data: List[Dict], right_data: List[Dict]
    ) -> Tuple[List[Dict], float]:
        left_key = self.config["join_key"]["left"]["name"]
        right_key = self.config["join_key"]["right"]["name"]
        left_limit = self.config["join_key"]["left"].get("limit", float("inf"))
        right_limit = self.config["join_key"]["right"].get("limit", float("inf"))
        blocking_threshold = self.config.get("blocking_threshold")
        blocking_conditions = self.config.get("blocking_conditions", [])
        total_cost = 0

        def is_match(left_item: Dict[str, Any], right_item: Dict[str, Any]) -> bool:
            return any(
                eval(condition, {"left": left_item, "right": right_item})
                for condition in blocking_conditions
            )

        if len(left_data) == 0 or len(right_data) == 0:
            return [], 0

        # Initial blocking
        blocked_pairs = []
        for left_item in left_data:
            for right_item in right_data:
                if is_match(left_item, right_item):
                    blocked_pairs.append((left_item, right_item))

        if blocking_threshold is not None:
            embedding_model = self.config.get("embedding_model", self.default_model)

            def get_embedding(
                item: Dict[str, Any], keys: List[str]
            ) -> Tuple[List[float], float]:
                text = " ".join(str(item[key]) for key in keys if key in item)
                response = embedding(model=embedding_model, input=[text])
                return response["data"][0]["embedding"], completion_cost(response)

            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                left_embeddings = list(
                    executor.map(
                        lambda x: get_embedding(x, [left_key]),
                        left_data,
                    )
                )
                right_embeddings = list(
                    executor.map(
                        lambda x: get_embedding(x, [right_key]),
                        right_data,
                    )
                )

            left_embeddings, left_costs = zip(*left_embeddings)
            right_embeddings, right_costs = zip(*right_embeddings)
            total_cost += sum(left_costs) + sum(right_costs)

            # Compute all cosine similarities in one call
            similarities = cosine_similarity(left_embeddings, right_embeddings)

            # Additional blocking based on embeddings
            for i, left_item in enumerate(left_data):
                for j, right_item in enumerate(right_data):
                    if (left_item, right_item) not in blocked_pairs:
                        if similarities[i][j] >= blocking_threshold:
                            blocked_pairs.append((left_item, right_item))

        # If there are no blocking conditions or embedding threshold, use all pairs
        if not blocking_conditions and blocking_threshold is None:
            blocked_pairs = [
                (left_item, right_item)
                for left_item in left_data
                for right_item in right_data
            ]

        # Calculate and print statistics
        total_possible_comparisons = len(left_data) * len(right_data)
        comparisons_made = len(blocked_pairs)
        comparisons_saved = total_possible_comparisons - comparisons_made
        self.console.print(
            f"[green]Comparisons saved by blocking: {comparisons_saved} "
            f"({(comparisons_saved / total_possible_comparisons) * 100:.2f}%)[/green]"
        )

        # LLM-based comparison for blocked pairs
        def get_hashable_key(item: Dict) -> str:
            return json.dumps(item, sort_keys=True)

        left_match_counts = defaultdict(int)
        right_match_counts = defaultdict(int)
        results = []
        comparison_costs = 0

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            future_to_pair = {
                executor.submit(
                    compare_pair,
                    self.config["comparison_prompt"],
                    self.config.get("comparison_model", self.default_model),
                    left,
                    right,
                ): (left, right)
                for left, right in blocked_pairs
            }

            for future in tqdm(
                as_completed(future_to_pair),
                total=len(future_to_pair),
                desc="Comparing pairs",
            ):
                pair = future_to_pair[future]
                is_match, cost = future.result()
                comparison_costs += cost

                if is_match:
                    joined_item = {}
                    left_item, right_item = pair
                    left_key_hash = get_hashable_key(left_item)
                    right_key_hash = get_hashable_key(right_item)
                    if (
                        left_match_counts[left_key_hash] >= left_limit
                        or right_match_counts[right_key_hash] >= right_limit
                    ):
                        continue

                    for key, value in left_item.items():
                        joined_item[f"{key}_left" if key in right_item else key] = value
                    for key, value in right_item.items():
                        joined_item[f"{key}_right" if key in left_item else key] = value
                    if validate_output(self.config, joined_item, self.console):
                        results.append(joined_item)
                        left_match_counts[left_key_hash] += 1
                        right_match_counts[right_key_hash] += 1

        total_cost += comparison_costs

        # Calculate and print the join selectivity
        join_selectivity = (
            len(results) / (len(left_data) * len(right_data))
            if len(left_data) * len(right_data) > 0
            else 0
        )
        self.console.print(f"Equijoin selectivity: {join_selectivity:.4f}")

        return results, total_cost
