from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random
import re
from typing import List, Dict, Any, Tuple
import numpy as np
from litellm import embedding, completion_cost
from uuid import uuid4
from rich.console import Console
from motion.operations.resolve import compare_pair


class ResolveOptimizer:
    def __init__(
        self,
        config: Dict[str, Any],
        op_config: Dict[str, Any],
        console: Console,
        llm_client: Any,
        max_threads: int,
        target_recall: float = 0.95,
        sample_size: int = 300,
        sampling_weight: float = 5,
        agent_max_retries: int = 5,
    ):
        self.config = config
        self.op_config = op_config
        self.llm_client = llm_client
        self.max_threads = max_threads
        self.console = console
        self.target_recall = target_recall
        self.sample_size = sample_size
        self.sampling_weight = sampling_weight
        self.agent_max_retries = agent_max_retries

    def optimize(
        self, input_data: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], float]:
        embeddings, blocking_keys, embedding_cost = self._compute_embeddings(input_data)
        similarities = self._calculate_cosine_similarities(embeddings)

        sampled_pairs = self._sample_pairs(similarities)
        comparison_results, comparison_cost = self._perform_comparisons(
            input_data, sampled_pairs
        )

        self._print_similarity_histogram(similarities, comparison_results)

        threshold = self._find_optimal_threshold(comparison_results, similarities)

        blocking_rules = self._generate_blocking_rules(
            blocking_keys, input_data, comparison_results
        )

        if blocking_rules:
            false_negatives = self._verify_blocking_rule(
                input_data, blocking_rules[0], blocking_keys, comparison_results
            )
            if not false_negatives:
                self.console.print(
                    "[green]Blocking rule verified. No false negatives detected in the sample.[/green]"
                )
            else:
                self.console.print(
                    f"[red]Blocking rule rejected. {len(false_negatives)} false negatives detected in the sample.[/red]"
                )
                for i, j in false_negatives[:5]:  # Show up to 5 examples
                    self.console.print(
                        f"  Filtered pair: {{ {blocking_keys[0]}: {input_data[i][blocking_keys[0]]} }} and {{ {blocking_keys[0]}: {input_data[j][blocking_keys[0]]} }}"
                    )
                if len(false_negatives) > 5:
                    self.console.print(f"  ... and {len(false_negatives) - 5} more.")
                blocking_rules = (
                    []
                )  # Clear the blocking rule if it introduces false negatives

        optimized_config = self._update_config(threshold, blocking_keys, blocking_rules)
        return optimized_config, embedding_cost + comparison_cost

    def _compute_embeddings(
        self, input_data: List[Dict[str, Any]]
    ) -> Tuple[List[List[float]], List[str], float]:
        blocking_keys = self.op_config.get("blocking_keys", [])
        if not blocking_keys:
            prompt_template = self.op_config.get("comparison_prompt", "")
            keys = set(re.findall(r"input[12]\.(\w+)", prompt_template))
            blocking_keys = list(keys)
        if not blocking_keys:
            self.console.print(
                "[yellow]Warning: No blocking keys found. Using all keys for blocking.[/yellow]"
            )
            blocking_keys = list(input_data[0].keys())

        texts = [
            " ".join(str(item[key]) for key in blocking_keys if key in item)
            for item in input_data
        ]
        response = embedding(
            model=self.op_config.get("embedding_model", "text-embedding-3-small"),
            input=texts,
        )
        embeddings = [data["embedding"] for data in response["data"]]
        cost = completion_cost(response)
        self.console.print(
            f"[bold]Cost of creating embeddings on the sample: ${cost:.4f}[/bold]"
        )
        return embeddings, blocking_keys, cost

    def _calculate_cosine_similarities(
        self, embeddings: List[List[float]]
    ) -> List[Tuple[int, int, float]]:
        embeddings_array = np.array(embeddings)
        norms = np.linalg.norm(embeddings_array, axis=1)
        dot_products = np.dot(embeddings_array, embeddings_array.T)
        similarities_matrix = dot_products / np.outer(norms, norms)
        i, j = np.triu_indices(len(embeddings), k=1)
        similarities = list(
            zip(i.tolist(), j.tolist(), similarities_matrix[i, j].tolist())
        )
        return similarities

    def _print_similarity_histogram(
        self,
        similarities: List[Tuple[int, int, float]],
        comparison_results: List[Tuple[int, int, bool]],
    ):
        flat_similarities = [sim[-1] for sim in similarities if sim[-1] != 1]
        hist, bin_edges = np.histogram(flat_similarities, bins=20)
        max_bar_width, max_count = 50, max(hist)
        normalized_hist = [int(count / max_count * max_bar_width) for count in hist]

        # Create a dictionary to store true labels
        true_labels = {(i, j): is_match for i, j, is_match in comparison_results}

        self.console.print("\n[bold]Embedding Cosine Similarity Distribution:[/bold]")
        for i, count in enumerate(normalized_hist):
            bar = "█" * count
            label = f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}"

            # Count true matches and not matches in this bin
            true_matches = 0
            not_matches = 0
            labeled_count = 0
            for sim in similarities:
                if bin_edges[i] <= sim[2] < bin_edges[i + 1]:
                    if (sim[0], sim[1]) in true_labels:
                        labeled_count += 1
                        if true_labels[(sim[0], sim[1])]:
                            true_matches += 1
                        else:
                            not_matches += 1

            # Calculate percentages of labeled pairs
            if labeled_count > 0:
                true_match_percent = (true_matches / labeled_count) * 100
                not_match_percent = (not_matches / labeled_count) * 100
            else:
                true_match_percent = 0
                not_match_percent = 0

            self.console.print(
                f"{label}: {bar} "
                f"(Labeled: {labeled_count}/{hist[i]}, [green]{true_match_percent:.1f}% match[/green], [red]{not_match_percent:.1f}% not match[/red])"
            )
        self.console.print("\n")

    def _sample_pairs(
        self, similarities: List[Tuple[int, int, float]]
    ) -> List[Tuple[int, int]]:
        sorted_similarities = sorted(similarities, key=lambda x: x[2], reverse=True)

        # Calculate weights that favor higher similarities
        similarities_array = np.array([sim[2] for sim in sorted_similarities])
        weights = np.exp(
            self.sampling_weight * similarities_array
        )  # Exponential weighting
        weights /= weights.sum()

        # Sample pairs based on the calculated weights
        sampled_indices = np.random.choice(
            len(sorted_similarities),
            size=min(self.sample_size, len(sorted_similarities)),
            replace=False,
            p=weights,
        )

        sampled_pairs = [
            (sorted_similarities[i][0], sorted_similarities[i][1])
            for i in sampled_indices
        ]
        return sampled_pairs

    def _perform_comparisons(
        self, input_data: List[Dict[str, Any]], pairs: List[Tuple[int, int]]
    ) -> Tuple[List[Tuple[int, int, bool]], float]:
        comparisons, total_cost = [], 0
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [
                executor.submit(
                    compare_pair,
                    self.op_config["comparison_prompt"],
                    self.op_config.get(
                        "comparison_model", self.config.get("model", "gpt-4o-mini")
                    ),
                    input_data[i],
                    input_data[j],
                )
                for i, j in pairs
            ]
            for future, (i, j) in zip(futures, pairs):
                is_match, cost = future.result()
                comparisons.append((i, j, is_match))
                total_cost += cost

        self.console.print(
            f"[bold]Cost of pairwise comparisons on the sample: ${total_cost:.4f}[/bold]"
        )
        return comparisons, total_cost

    def _find_optimal_threshold(
        self,
        comparisons: List[Tuple[int, int, bool]],
        similarities: List[Tuple[int, int, float]],
    ) -> float:
        true_labels = np.array([comp[2] for comp in comparisons])
        sim_dict = {(i, j): sim for i, j, sim in similarities}
        sim_scores = np.array([sim_dict[(i, j)] for i, j, _ in comparisons])

        thresholds = np.linspace(0, 1, 100)
        precisions, recalls = [], []

        for threshold in thresholds:
            predictions = sim_scores >= threshold
            tp = np.sum(predictions & true_labels)
            fp = np.sum(predictions & ~true_labels)
            fn = np.sum(~predictions & true_labels)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            precisions.append(precision)
            recalls.append(recall)

        valid_indices = [i for i, r in enumerate(recalls) if r >= self.target_recall]
        if not valid_indices:
            optimal_threshold = float(thresholds[np.argmax(recalls)])
        else:
            optimal_threshold = float(thresholds[max(valid_indices)])

        # Improved selectivity estimation
        all_similarities = np.array([s[2] for s in similarities])
        sampled_similarities = sim_scores

        # Calculate sampling probabilities
        sampling_probs = np.exp(self.sampling_weight * sampled_similarities)
        sampling_probs /= sampling_probs.sum()

        # Estimate selectivity using importance sampling
        weights = 1 / (len(all_similarities) * sampling_probs)
        estimated_selectivity = np.sum(weights * true_labels) / np.sum(weights)

        # Calculate variance of the estimate
        var_estimate = np.sum(
            weights**2 * (true_labels - estimated_selectivity) ** 2
        ) / (np.sum(weights) ** 2)

        # Standard error
        se_estimated = np.sqrt(var_estimate)

        # Calculate 95% confidence interval
        ci_lower = max(0, estimated_selectivity - 1.96 * se_estimated)
        ci_upper = min(1, estimated_selectivity + 1.96 * se_estimated)

        self.console.print(
            f"[bold cyan]┌─ Estimated Self-Join Selectivity ─────────────────────────┐[/bold cyan]"
        )
        self.console.print(
            f"[bold cyan]│[/bold cyan] [yellow]Target Recall:[/yellow] {self.target_recall:.0%}"
        )
        self.console.print(
            f"[bold cyan]│[/bold cyan] [yellow]Estimate:[/yellow] {estimated_selectivity:.4f}"
        )
        self.console.print(
            f"[bold cyan]│[/bold cyan] [yellow]95% Confidence Interval:[/yellow] [{ci_lower:.4f} - {ci_upper:.4f}]"
        )
        self.console.print(
            f"[bold cyan]└───────────────────────────────────────────────────────────┘[/bold cyan]"
        )
        self.console.print(
            f"[bold]Chosen similarity threshold for blocking: {optimal_threshold:.4f}[/bold]"
        )

        return round(optimal_threshold, 4)

    def _generate_blocking_rules(
        self,
        blocking_keys: List[str],
        input_data: List[Dict[str, Any]],
        comparisons: List[Tuple[int, int, bool]],
    ) -> List[str]:
        # Sample 2 true and 2 false comparisons
        true_comparisons = [comp for comp in comparisons if comp[2]][:2]
        false_comparisons = [comp for comp in comparisons if not comp[2]][:2]
        sample_datas = [
            (
                {key: input_data[i][key] for key in blocking_keys},
                {key: input_data[j][key] for key in blocking_keys},
                is_match,
            )
            for i, j, is_match in true_comparisons + false_comparisons
        ]

        messages = [
            {
                "role": "user",
                "content": f"""Given the following sample comparisons between entities, generate a single-line Python statement that acts as a blocking rule for entity resolution. This rule will be used in the form: `eval(blocking_rule, {{"input1": item1, "input2": item2}})`.

    Sample comparisons (note: these are just a few examples and may not represent all possible cases):
    {json.dumps(sample_datas, indent=2)}

    For context, here is the comparison prompt that will be used for the more expensive, detailed comparison:
    {self.op_config.get('comparison_prompt', 'No comparison prompt provided.')}

    Please generate ONE one-line blocking rule that adheres to the following criteria:
    1. The rule should evaluate to True if the entities are possibly a match and require further comparison.
    2. The rule should evaluate to False ONLY if the entities are definitely not a match.
    3. The rule must be a single Python expression that can be evaluated using the eval() function.
    4. The rule should be much faster to evaluate than the full comparison prompt.
    5. The rule should capture the essence of the comparison prompt but in a simplified manner.
    6. The rule should be general enough to work well on the entire dataset, not just these specific examples.
    7. The rule should handle inconsistent casing by using string methods like .lower() when comparing string values.
    8. The rule should err on the side of inclusivity - it's better to have false positives than false negatives.

    Example structure of a one-line blocking rule:
    "(condition1) or (condition2) or (condition3)"

    Where conditions could be comparisons like:
    "input1['field'].lower() == input2['field'].lower()"
    "abs(len(input1['text']) - len(input2['text'])) <= 5"
    "any(word in input1['description'].lower() for word in input2['description'].lower().split())"

    If there's no clear rule that can be generated based on the given information, return the string "True" to ensure all pairs are compared.

    Remember, the primary goal of the blocking rule is to safely reduce the number of comparisons by quickly identifying pairs that are definitely not matches, while keeping all potential matches for further evaluation.""",
            }
        ]

        for attempt in range(self.agent_max_retries):  # Up to 3 attempts
            # Generate blocking rule using the LLM
            response = self.llm_client.generate(
                messages,
                "You are an expert in entity resolution and Python programming. Your task is to generate one efficient blocking rule based on the given sample comparisons and data structure.",
                {
                    "type": "object",
                    "properties": {
                        "blocking_rule": {
                            "type": "string",
                            "description": "One-line Python statement acting as a blocking rule",
                        }
                    },
                },
            )

            # Extract the blocking rule from the LLM response
            blocking_rule = response.choices[0].message.tool_calls[0].function.arguments
            blocking_rule = json.loads(blocking_rule).get("blocking_rule")

            if blocking_rule:
                self.console.print("")  # Print a newline

                if blocking_rule.strip() == "True":
                    self.console.print(
                        "[yellow]No suitable blocking rule could be found. Proceeding without a blocking rule.[/yellow]"
                    )
                    return []

                self.console.print(
                    f"[bold]Generated blocking rule (Attempt {attempt + 1}):[/bold] {blocking_rule}"
                )

                # Test the blocking rule
                filtered_pairs = self._test_blocking_rule(
                    input_data, blocking_keys, blocking_rule, comparisons
                )

                if not filtered_pairs:
                    self.console.print(
                        "[green]Blocking rule looks good! No known matches were filtered out.[/green]"
                    )
                    return [blocking_rule]
                else:
                    feedback = f"The previous rule incorrectly filtered out {len(filtered_pairs)} known matches. "
                    feedback += (
                        "Here are up to 3 examples of incorrectly filtered pairs:\n"
                    )
                    for i, j in filtered_pairs[:3]:
                        feedback += f"Item 1: {json.dumps({key: input_data[i][key] for key in blocking_keys})}\Item 2: {json.dumps({key: input_data[j][key] for key in blocking_keys})}\n"
                        feedback += "These pairs are known matches but were filtered out by the rule.\n"
                    feedback += "Please generate a new rule that doesn't filter out these matches."

                    messages.append({"role": "assistant", "content": blocking_rule})
                    messages.append({"role": "user", "content": feedback})
            else:
                self.console.print("[yellow]No blocking rule generated.[/yellow]")
                return []

        self.console.print(
            f"[yellow]Failed to generate a suitable blocking rule after {self.agent_max_retries} attempts. Proceeding without a blocking rule.[/yellow]"
        )
        return []

    def _test_blocking_rule(
        self,
        input_data: List[Dict[str, Any]],
        blocking_keys: List[str],
        blocking_rule: str,
        comparisons: List[Tuple[int, int, bool]],
    ) -> List[Tuple[int, int]]:
        def apply_blocking_rule(item1, item2):
            try:
                return eval(blocking_rule, {"input1": item1, "input2": item2})
            except Exception as e:
                self.console.print(f"[red]Error applying blocking rule: {e}[/red]")
                return True  # If there's an error, we default to comparing the pair

        filtered_pairs = []

        for i, j, is_match in comparisons:
            if is_match:
                item1 = {
                    k: input_data[i][k] for k in blocking_keys if k in input_data[i]
                }
                item2 = {
                    k: input_data[j][k] for k in blocking_keys if k in input_data[j]
                }

                if not apply_blocking_rule(item1, item2):
                    filtered_pairs.append((i, j))

        if filtered_pairs:
            self.console.print(
                f"[yellow italic]LLM Correction: The blocking rule incorrectly filtered out {len(filtered_pairs)} known positive matches.[/yellow italic]"
            )
            for i, j in filtered_pairs[:5]:  # Show up to 5 examples
                self.console.print(
                    f"  Incorrectly filtered pair 1: {json.dumps({key: input_data[i][key] for key in blocking_keys})}  and pair 2: {json.dumps({key: input_data[j][key] for key in blocking_keys})}"
                )
            if len(filtered_pairs) > 5:
                self.console.print(
                    f"  ... and {len(filtered_pairs) - 5} more incorrect pairs."
                )

        return filtered_pairs

    def _verify_blocking_rule(
        self,
        input_data: List[Dict[str, Any]],
        blocking_rule: str,
        blocking_keys: List[str],
        comparison_results: List[Tuple[int, int, bool]],
    ) -> List[Tuple[int, int]]:
        def apply_blocking_rule(item1, item2):
            try:
                return eval(blocking_rule, {"input1": item1, "input2": item2})
            except Exception as e:
                self.console.print(f"[red]Error applying blocking rule: {e}[/red]")
                return True  # If there's an error, we default to comparing the pair

        false_negatives = []

        for i, j, is_match in comparison_results:
            if is_match:
                item1 = {
                    k: input_data[i][k] for k in blocking_keys if k in input_data[i]
                }
                item2 = {
                    k: input_data[j][k] for k in blocking_keys if k in input_data[j]
                }

                if not apply_blocking_rule(item1, item2):
                    false_negatives.append((i, j))

        return false_negatives

    def _update_config(
        self, threshold: float, blocking_keys: List[str], blocking_rules: List[str]
    ) -> Dict[str, Any]:
        optimized_config = self.op_config.copy()
        optimized_config["blocking_keys"] = blocking_keys
        optimized_config["blocking_threshold"] = threshold
        if blocking_rules:
            optimized_config["blocking_conditions"] = blocking_rules
        if "embedding_model" not in optimized_config:
            optimized_config["embedding_model"] = "text-embedding-3-small"
        return optimized_config
