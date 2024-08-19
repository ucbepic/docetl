from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random
import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from litellm import embedding, completion_cost
from uuid import uuid4
from rich.console import Console
from motion.operations.resolve import compare_pair as compare_pair_resolve
from motion.operations.equijoin import compare_pair as compare_pair_equijoin
from scipy.optimize import brentq


class JoinOptimizer:
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

    def _analyze_map_prompt_categorization(self, map_prompt: str) -> bool:
        """
        Analyze the map prompt to determine if it's explicitly categorical.

        Args:
            map_prompt (str): The map prompt to analyze.

        Returns:
            bool: True if the prompt is explicitly categorical, False otherwise.
        """
        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant tasked with analyzing prompts for data processing operations.",
            },
            {
                "role": "user",
                "content": f"""Analyze the following map operation prompt and determine if it is explicitly categorical, 
                meaning it details a specific set of possible outputs:

                {map_prompt}

                Respond with 'Yes' if the prompt is explicitly categorical, detailing a finite set of possible outputs.
                Respond with 'No' if the prompt allows for open-ended or non-categorical responses.
                Provide a brief explanation for your decision.""",
            },
        ]

        response = self.llm_client.generate(
            messages,
            "You are an expert in analyzing natural language prompts for data processing tasks.",
            {
                "type": "object",
                "properties": {
                    "is_categorical": {
                        "type": "string",
                        "enum": ["Yes", "No"],
                        "description": "Whether the prompt is explicitly categorical",
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Brief explanation for the decision",
                    },
                },
                "required": ["is_categorical", "explanation"],
            },
        )

        analysis = json.loads(response.choices[0].message.content)

        self.console.log(f"[bold]Map Prompt Analysis:[/bold]")
        self.console.log(f"Is Categorical: {analysis['is_categorical']}")
        self.console.log(f"Explanation: {analysis['explanation']}")

        return analysis["is_categorical"].lower() == "yes"

    def _determine_duplicate_keys(
        self,
        input_data: List[Dict[str, Any]],
        reduce_key: List[str],
        map_prompt: Optional[str] = None,
    ) -> bool:
        # Prepare a sample of the input data for analysis
        sample_size = min(10, len(input_data))
        data_sample = random.sample(
            [{rk: item[rk] for rk in reduce_key} for item in input_data], sample_size
        )

        context_prefix = ""
        if map_prompt:
            context_prefix = f"For context, these values came out of a pipeline with the following prompt:\n\n{map_prompt}\n\n"

        messages = [
            {
                "role": "user",
                "content": f"{context_prefix}I want to do a reduce operation on these values, and I need to determine if there are semantic duplicates in the data, where the strings are different but they technically belong in the same group. Note that exact string duplicates should not be considered here.\n\nHere's a sample of the data (showing the '{reduce_key}' field(s)): {data_sample}\n\nBased on this {'context and ' if map_prompt else ''}sample, are there likely to be such semantic duplicates (not exact string matches) in the dataset? Respond with 'yes' only if you think there are semantic duplicates, or 'no' if you don't see evidence of semantic duplicates or if you only see exact string duplicates.",
            },
        ]
        response = self.llm_client.generate(
            messages,
            "You are an expert data analyst. Analyze the given data sample and determine if there are likely to be semantic duplicate values that belong in the same group, even if the strings are different.",
            {
                "type": "object",
                "properties": {
                    "likely_duplicates": {
                        "type": "string",
                        "enum": ["Yes", "No"],
                        "description": "Whether duplicates are likely to exist in the full dataset",
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Brief explanation for the decision",
                    },
                },
                "required": ["likely_duplicates", "explanation"],
            },
        )

        analysis = json.loads(response.choices[0].message.content)

        self.console.log(f"[bold]Duplicate Analysis for '{reduce_key}':[/bold]")
        self.console.log(f"Likely Duplicates: {analysis['likely_duplicates']}")
        self.console.log(f"Explanation: {analysis['explanation']}")

        if analysis["likely_duplicates"].lower() == "yes":
            self.console.log(
                "[yellow]Duplicates are likely. Consider using a deduplication strategy in the resolution step.[/yellow]"
            )
            return True
        return False

    def _sample_random_pairs(
        self, input_data: List[Dict[str, Any]], n: int
    ) -> List[Tuple[int, int]]:
        """Sample random pairs of indices, excluding exact matches."""
        pairs = set()
        max_attempts = n * 10  # Avoid infinite loop
        attempts = 0

        while len(pairs) < n and attempts < max_attempts:
            i, j = random.sample(range(len(input_data)), 2)
            if i != j and input_data[i] != input_data[j]:
                pairs.add((min(i, j), max(i, j)))  # Ensure ordered pairs
            attempts += 1

        return list(pairs)

    def _check_duplicates_with_llm(
        self,
        input_data: List[Dict[str, Any]],
        pairs: List[Tuple[int, int]],
        reduce_key: List[str],
        map_prompt: Optional[str],
    ) -> bool:
        """Use LLM to check if any pairs are duplicates."""

        content = "Analyze the following pairs of entries and determine if any of them are likely duplicates. Respond with 'Yes' if you find any likely duplicates, or 'No' if none of the pairs seem to be duplicates. Provide a brief explanation for your decision.\n\n"

        if map_prompt:
            content = (
                f"For reference, here is the map prompt used earlier in the pipeline: {map_prompt}\n\n"
                + content
            )

        for i, (idx1, idx2) in enumerate(pairs, 1):
            content += f"Pair {i}:\n"
            content += "Entry 1:\n"
            for key in reduce_key:
                content += f"{key}: {json.dumps(input_data[idx1][key], indent=2)}\n"
            content += "\nEntry 2:\n"
            for key in reduce_key:
                content += f"{key}: {json.dumps(input_data[idx2][key], indent=2)}\n"
            content += "\n"

        messages = [{"role": "user", "content": content}]

        system_prompt = "You are an AI assistant tasked with identifying potential duplicate entries in a dataset."
        response_schema = {
            "type": "object",
            "properties": {
                "duplicates_found": {"type": "string", "enum": ["Yes", "No"]},
                "explanation": {"type": "string"},
            },
            "required": ["duplicates_found", "explanation"],
        }

        response = self.llm_client.generate(messages, system_prompt, response_schema)

        # Print the duplicates_found and explanation
        self.console.log(
            f"[bold]Duplicates in keys found:[/bold] {response['duplicates_found']}\n"
            f"[bold]Explanation:[/bold] {response['explanation']}"
        )

        return response["duplicates_found"].lower() == "yes"

    def synthesize_compare_prompt(
        self, map_prompt: Optional[str], reduce_key: List[str]
    ) -> str:

        system_prompt = f"You are an AI assistant tasked with creating a comparison prompt for LLM-assisted entity resolution. Your task is to create a comparison prompt that will be used to compare two entities, referred to as input1 and input2, to see if they are likely the same entity based on the following reduce key(s): {', '.join(reduce_key)}."
        if map_prompt:
            system_prompt += f"\n\nFor context, here is the prompt used earlier in the pipeline to create the inputs to resolve: {map_prompt}"

        messages = [
            {
                "role": "user",
                "content": f"""
    Create a comparison prompt for entity resolution: The prompt should:
    1. Be tailored to the specific domain and type of data being compared, based on the context provided.
    2. Instruct to compare two entities, referred to as input1 and input2.
    3. Specifically mention comparing each reduce key in input1 and input2 (e.g., input1.{{key}} and input2.{{key}} for each key in {reduce_key}).
    4. Include instructions to consider relevant attributes or characteristics for comparison.
    5. Ask to respond with "True" if the entities are likely the same, or "False" if they are likely different.

    Example structure:
    ```
    Compare the following two [entity type]:

    [Entity 1]:
    {{{{ input1.key1 }}}}

    [Entity 2]:
    {{{{ input2.key1 }}}}

    Are these [entities] likely referring to the same [entity type]? Consider [list relevant attributes or characteristics to compare]. Respond with "True" if they are likely the same [entity type], or "False" if they are likely different [entity types].
    ```

    Please generate the comparison prompt:
    """,
            }
        ]

        response = self.llm_client.generate(
            messages,
            system_prompt,
            {
                "type": "object",
                "properties": {
                    "comparison_prompt": {
                        "type": "string",
                        "description": "Detailed comparison prompt for entity resolution",
                    }
                },
                "required": ["comparison_prompt"],
            },
        )

        comparison_prompt = json.loads(response.choices[0].message.content)[
            "comparison_prompt"
        ]

        # Log the synthesized comparison prompt
        self.console.log("[green]Synthesized comparison prompt:[/green]")
        self.console.log(comparison_prompt)

        if not comparison_prompt:
            raise ValueError(
                "Could not synthesize a comparison prompt. Please provide a comparison prompt in the config."
            )

        return comparison_prompt

    def synthesize_resolution_prompt(
        self,
        map_prompt: Optional[str],
        reduce_key: List[str],
        output_schema: Dict[str, str],
    ) -> str:
        system_prompt = f"""You are an AI assistant tasked with creating a resolution prompt for LLM-assisted entity resolution. 
        Your task is to create a prompt that will be used to merge multiple duplicate keys into a single, consolidated key.
        The key(s) being resolved (known as the reduce_key) are {', '.join(reduce_key)}.
        The duplicate keys will be provided in a list called 'matched_entries' in a Jinja2 template.
        """

        if map_prompt:
            system_prompt += f"\n\nFor context, here is the prompt used earlier in the pipeline to create the inputs to resolve: {map_prompt}"

        messages = [
            {
                "role": "user",
                "content": f"""
    Create a resolution prompt for merging duplicate keys into a single key. The prompt should:
    1. Be tailored to the specific domain and type of data being merged, based on the context provided.
    2. Use a Jinja2 template to iterate over the duplicate keys (accessed as 'matched_entries', where each item is a dictionary containing the reduce_key fields, which you can access as entry.reduce_key for each reduce_key in {reduce_key}).
    3. Instruct to create a single, consolidated key from the duplicate keys.
    4. Include guidelines for resolving conflicts (e.g., choosing the most recent, most complete, or most reliable information).
    5. Specify that the output of the resolution prompt should conform to the given output schema: {json.dumps(output_schema, indent=2)}

    Example structure:
    ```
    Analyze the following duplicate entries:

    {{% for key in matched_entries %}}
    Entry {{{{ loop.index }}}}:
    {{{{ key | tojson }}}}

    {{% endfor %}}

    Create a single, consolidated key that combines the information from all duplicate entries. 
    When merging, follow these guidelines:
    1. [Provide specific merging instructions relevant to the data type]
    2. [Provide conflict resolution guidelines]
    3. [Any other relevant instructions]

    Ensure that the merged key conforms to the following schema:
    {json.dumps(output_schema, indent=2)}

    Return the consolidated key as a single [appropriate data type] value.
    ```

    Please generate the resolution prompt:
    """,
            }
        ]

        response = self.llm_client.generate(
            messages,
            system_prompt,
            {
                "type": "object",
                "properties": {
                    "resolution_prompt": {
                        "type": "string",
                        "description": "Detailed resolution prompt for merging duplicate keys",
                    }
                },
                "required": ["resolution_prompt"],
            },
        )

        resolution_prompt = json.loads(response.choices[0].message.content)[
            "resolution_prompt"
        ]

        # Log the synthesized resolution prompt
        self.console.log("[green]Synthesized resolution prompt:[/green]")
        self.console.log(resolution_prompt)

        if not resolution_prompt:
            raise ValueError(
                "Could not synthesize a resolution prompt. Please provide a resolution prompt in the config."
            )

        return resolution_prompt

    def optimize_resolve(
        self, input_data: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], float]:

        # Check if the operation is marked as empty
        if self.op_config.get("empty", False):
            # Extract the map prompt from the intermediates
            map_prompt = self.op_config["_intermediates"]["map_prompt"]
            reduce_key = self.op_config["_intermediates"]["reduce_key"]

            if reduce_key is None:
                raise ValueError(
                    "[yellow]Warning: No reduce key found in intermediates for synthesized resolve operation.[/yellow]"
                )

            dedup = True

            if map_prompt:
                # Analyze the map prompt
                analysis = self._analyze_map_prompt_categorization(map_prompt)

                if analysis:
                    dedup = False
            else:
                self.console.log(
                    "[yellow]No map prompt found in intermediates for analysis.[/yellow]"
                )

            if dedup is False:
                dedup = self._determine_duplicate_keys(
                    input_data, reduce_key, map_prompt
                )

            # Now do the last attempt of pairwise comparisons
            if dedup is False:
                # Sample up to 20 random pairs of keys for duplicate analysis
                sampled_pairs = self._sample_random_pairs(input_data, 20)

                # Use LLM to check for duplicates
                duplicates_found = self._check_duplicates_with_llm(
                    input_data, sampled_pairs, reduce_key, map_prompt
                )

                if duplicates_found:
                    dedup = True

            if dedup is False:
                # If no deduplication is needed, return the same config with 0 cost
                return self.op_config, 0.0

            # Add the reduce key to the output schema in the config
            self.op_config["output"] = {"schema": {rk: "string" for rk in reduce_key}}
            self.op_config["comparison_prompt"] = self.synthesize_compare_prompt(
                map_prompt, reduce_key
            )
            self.op_config["resolution_prompt"] = self.synthesize_resolution_prompt(
                map_prompt, reduce_key, self.op_config["output"]["schema"]
            )

            # Pop off the empty flag
            self.op_config.pop("empty")

        embeddings, blocking_keys, embedding_cost = self._compute_embeddings(input_data)
        self.console.log(
            f"[bold]Cost of creating embeddings on the sample: ${embedding_cost:.4f}[/bold]"
        )

        similarities = self._calculate_cosine_similarities(embeddings)

        sampled_pairs = self._sample_pairs(similarities)
        comparison_results, comparison_cost = self._perform_comparisons_resolve(
            input_data, sampled_pairs
        )

        self._print_similarity_histogram(similarities, comparison_results)

        threshold, estimated_selectivity = self._find_optimal_threshold(
            comparison_results, similarities
        )

        blocking_rules = self._generate_blocking_rules(
            blocking_keys, input_data, comparison_results
        )

        if blocking_rules:
            false_negatives, rule_selectivity = self._verify_blocking_rule(
                input_data,
                blocking_rules[0],
                blocking_keys,
                comparison_results,
            )
            if not false_negatives and rule_selectivity <= estimated_selectivity:
                self.console.log(
                    "[green]Blocking rule verified. No false negatives detected in the sample and selectivity is within estimated selectivity.[/green]"
                )
            else:
                if false_negatives:
                    self.console.log(
                        f"[red]Blocking rule rejected. {len(false_negatives)} false negatives detected in the sample.[/red]"
                    )
                    for i, j in false_negatives[:5]:  # Show up to 5 examples
                        self.console.log(
                            f"  Filtered pair: {{ {blocking_keys[0]}: {input_data[i][blocking_keys[0]]} }} and {{ {blocking_keys[0]}: {input_data[j][blocking_keys[0]]} }}"
                        )
                    if len(false_negatives) > 5:
                        self.console.log(f"  ... and {len(false_negatives) - 5} more.")
                if rule_selectivity > estimated_selectivity:
                    self.console.log(
                        f"[red]Blocking rule rejected. Rule selectivity ({rule_selectivity:.4f}) is higher than the estimated selectivity ({estimated_selectivity:.4f}).[/red]"
                    )
                blocking_rules = (
                    []
                )  # Clear the blocking rule if it introduces false negatives or is too selective

        optimized_config = self._update_config(threshold, blocking_keys, blocking_rules)
        return optimized_config, embedding_cost + comparison_cost

    def optimize_equijoin(
        self, left_data: List[Dict[str, Any]], right_data: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], float]:
        left_key = self.op_config["join_key"]["left"]["name"]
        right_key = self.op_config["join_key"]["right"]["name"]

        left_embeddings, _, left_embedding_cost = self._compute_embeddings(
            left_data, [left_key]
        )
        right_embeddings, _, right_embedding_cost = self._compute_embeddings(
            right_data, [right_key]
        )
        self.console.log(
            f"[bold]Cost of creating embeddings on the sample: ${left_embedding_cost + right_embedding_cost:.4f}[/bold]"
        )

        similarities = self._calculate_cross_similarities(
            left_embeddings, right_embeddings
        )

        sampled_pairs = self._sample_pairs(similarities)
        comparison_results, comparison_cost = self._perform_comparisons_equijoin(
            left_data, right_data, sampled_pairs
        )

        self._print_similarity_histogram(similarities, comparison_results)

        threshold, estimated_selectivity = self._find_optimal_threshold(
            comparison_results, similarities
        )

        blocking_rules = self._generate_blocking_rules_equijoin(
            left_key, right_key, left_data, right_data, comparison_results
        )

        if blocking_rules:
            false_negatives, rule_selectivity = self._verify_blocking_rule_equijoin(
                left_data,
                right_data,
                blocking_rules[0],
                left_key,
                right_key,
                comparison_results,
            )
            if not false_negatives and rule_selectivity <= estimated_selectivity:
                self.console.log(
                    "[green]Blocking rule verified. No false negatives detected in the sample and selectivity is within bounds.[/green]"
                )
            else:
                if false_negatives:
                    self.console.log(
                        f"[red]Blocking rule rejected. {len(false_negatives)} false negatives detected in the sample.[/red]"
                    )
                    for i, j in false_negatives[:5]:  # Show up to 5 examples
                        self.console.log(
                            f"  Filtered pair: {{ {left_key}: {left_data[i][left_key]} }} and {{ {right_key}: {right_data[j][right_key]} }}"
                        )
                    if len(false_negatives) > 5:
                        self.console.log(f"  ... and {len(false_negatives) - 5} more.")
                if rule_selectivity > estimated_selectivity:
                    self.console.log(
                        f"[red]Blocking rule rejected. Rule selectivity ({rule_selectivity:.4f}) is higher than the estimated selectivity ({estimated_selectivity:.4f}).[/red]"
                    )
                blocking_rules = (
                    []
                )  # Clear the blocking rule if it introduces false negatives or is too selective

        optimized_config = self._update_config_equijoin(
            threshold, left_key, right_key, blocking_rules
        )
        return (
            optimized_config,
            left_embedding_cost + right_embedding_cost + comparison_cost,
        )

    def _compute_embeddings(
        self, input_data: List[Dict[str, Any]], keys: List[str] = None
    ) -> Tuple[List[List[float]], List[str], float]:
        if keys is None:
            keys = self.op_config.get("blocking_keys", [])
            if not keys:
                prompt_template = self.op_config.get("comparison_prompt", "")
                keys = list(set(re.findall(r"input[12]\.(\w+)", prompt_template)))
            if not keys:
                self.console.log(
                    "[yellow]Warning: No blocking keys found. Using all keys for blocking.[/yellow]"
                )
                keys = list(input_data[0].keys())

        texts = [
            " ".join(str(item[key]) for key in keys if key in item)
            for item in input_data
        ]
        response = embedding(
            model=self.op_config.get("embedding_model", "text-embedding-3-small"),
            input=texts,
        )
        embeddings = [data["embedding"] for data in response["data"]]
        cost = completion_cost(response)
        return embeddings, keys, cost

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

        self.console.log("\n[bold]Embedding Cosine Similarity Distribution:[/bold]")
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

            self.console.log(
                f"{label}: {bar} "
                f"(Labeled: {labeled_count}/{hist[i]}, [green]{true_match_percent:.1f}% match[/green], [red]{not_match_percent:.1f}% not match[/red])"
            )
        self.console.log("\n")

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

    def _calculate_cross_similarities(
        self, left_embeddings: List[List[float]], right_embeddings: List[List[float]]
    ) -> List[Tuple[int, int, float]]:
        left_array = np.array(left_embeddings)
        right_array = np.array(right_embeddings)
        dot_product = np.dot(left_array, right_array.T)
        norm_left = np.linalg.norm(left_array, axis=1)
        norm_right = np.linalg.norm(right_array, axis=1)
        similarities = dot_product / np.outer(norm_left, norm_right)
        return [
            (i, j, sim)
            for i, row in enumerate(similarities)
            for j, sim in enumerate(row)
        ]

    def _perform_comparisons_resolve(
        self, input_data: List[Dict[str, Any]], pairs: List[Tuple[int, int]]
    ) -> Tuple[List[Tuple[int, int, bool]], float]:
        comparisons, total_cost = [], 0
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [
                executor.submit(
                    compare_pair_resolve,
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

        self.console.log(
            f"[bold]Cost of pairwise comparisons on the sample: ${total_cost:.4f}[/bold]"
        )
        return comparisons, total_cost

    def _perform_comparisons_equijoin(
        self,
        left_data: List[Dict[str, Any]],
        right_data: List[Dict[str, Any]],
        pairs: List[Tuple[int, int]],
    ) -> Tuple[List[Tuple[int, int, bool]], float]:
        comparisons, total_cost = [], 0
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [
                executor.submit(
                    compare_pair_equijoin,
                    self.op_config["comparison_prompt"],
                    self.op_config.get(
                        "comparison_model", self.config.get("model", "gpt-4o-mini")
                    ),
                    left_data[i],
                    right_data[j] if right_data else left_data[j],
                )
                for i, j in pairs
            ]
            for future, (i, j) in zip(futures, pairs):
                is_match, cost = future.result()
                comparisons.append((i, j, is_match))
                total_cost += cost

        self.console.log(
            f"[bold]Cost of pairwise comparisons on the sample: ${total_cost:.4f}[/bold]"
        )
        return comparisons, total_cost

    def _find_optimal_threshold(
        self,
        comparisons: List[Tuple[int, int, bool]],
        similarities: List[Tuple[int, int, float]],
    ) -> Tuple[float, float, float]:
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
        numerator = np.sum(weights * true_labels)
        denominator = np.sum(weights)
        selectivity_estimate = numerator / denominator

        self.console.log(
            f"[bold cyan]┌─ Estimated Self-Join Selectivity ─────────────────────────┐[/bold cyan]"
        )
        self.console.log(
            f"[bold cyan]│[/bold cyan] [yellow]Target Recall:[/yellow] {self.target_recall:.0%}"
        )
        self.console.log(
            f"[bold cyan]│[/bold cyan] [yellow]Estimate:[/yellow] {selectivity_estimate:.4f}"
        )
        self.console.log(
            f"[bold cyan]└───────────────────────────────────────────────────────────┘[/bold cyan]"
        )
        self.console.log(
            f"[bold]Chosen similarity threshold for blocking: {optimal_threshold:.4f}[/bold]"
        )

        return round(optimal_threshold, 4), selectivity_estimate

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
                    "required": ["blocking_rule"],
                },
            )

            # Extract the blocking rule from the LLM response
            blocking_rule = response.choices[0].message.content
            blocking_rule = json.loads(blocking_rule).get("blocking_rule")

            if blocking_rule:
                self.console.log("")  # Print a newline

                if blocking_rule.strip() == "True":
                    self.console.log(
                        "[yellow]No suitable blocking rule could be found. Proceeding without a blocking rule.[/yellow]"
                    )
                    return []

                self.console.log(
                    f"[bold]Generated blocking rule (Attempt {attempt + 1}):[/bold] {blocking_rule}"
                )

                # Test the blocking rule
                filtered_pairs = self._test_blocking_rule(
                    input_data, blocking_keys, blocking_rule, comparisons
                )

                if not filtered_pairs:
                    self.console.log(
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
                self.console.log("[yellow]No blocking rule generated.[/yellow]")
                return []

        self.console.log(
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
                self.console.log(f"[red]Error applying blocking rule: {e}[/red]")
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
            self.console.log(
                f"[yellow italic]LLM Correction: The blocking rule incorrectly filtered out {len(filtered_pairs)} known positive matches.[/yellow italic]"
            )
            for i, j in filtered_pairs[:5]:  # Show up to 5 examples
                self.console.log(
                    f"  Incorrectly filtered pair 1: {json.dumps({key: input_data[i][key] for key in blocking_keys})}  and pair 2: {json.dumps({key: input_data[j][key] for key in blocking_keys})}"
                )
            if len(filtered_pairs) > 5:
                self.console.log(
                    f"  ... and {len(filtered_pairs) - 5} more incorrect pairs."
                )

        return filtered_pairs

    def _generate_blocking_rules_equijoin(
        self,
        left_key: str,
        right_key: str,
        left_data: List[Dict[str, Any]],
        right_data: List[Dict[str, Any]],
        comparisons: List[Tuple[int, int, bool]],
    ) -> List[str]:
        # Sample 2 true and 2 false comparisons
        true_comparisons = [comp for comp in comparisons if comp[2]][:2]
        false_comparisons = [comp for comp in comparisons if not comp[2]][:2]
        sample_datas = [
            (
                {left_key: left_data[i][left_key]},
                {right_key: right_data[j][right_key]},
                is_match,
            )
            for i, j, is_match in true_comparisons + false_comparisons
        ]

        messages = [
            {
                "role": "user",
                "content": f"""Given the following sample comparisons between entities, generate a single-line Python statement that acts as a blocking rule for equijoin. This rule will be used in the form: `eval(blocking_rule, {{"left": item1, "right": item2}})`.

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
    "left['{left_key}'].lower() == right['{right_key}'].lower()"
    "abs(len(left['{left_key}']) - len(right['{right_key}'])) <= 5"
    "any(word in left['{left_key}'].lower() for word in right['{right_key}'].lower().split())"

    If there's no clear rule that can be generated based on the given information, return the string "True" to ensure all pairs are compared.

    Remember, the primary goal of the blocking rule is to safely reduce the number of comparisons by quickly identifying pairs that are definitely not matches, while keeping all potential matches for further evaluation.""",
            }
        ]

        for attempt in range(self.agent_max_retries):
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
                    "required": ["blocking_rule"],
                },
            )

            blocking_rule = response.choices[0].message.content
            blocking_rule = json.loads(blocking_rule).get("blocking_rule")

            if blocking_rule:
                self.console.log("")

                if blocking_rule.strip() == "True":
                    self.console.log(
                        "[yellow]No suitable blocking rule could be found. Proceeding without a blocking rule.[/yellow]"
                    )
                    return []

                self.console.log(
                    f"[bold]Generated blocking rule (Attempt {attempt + 1}):[/bold] {blocking_rule}"
                )

                # Test the blocking rule
                filtered_pairs = self._test_blocking_rule_equijoin(
                    left_data,
                    right_data,
                    left_key,
                    right_key,
                    blocking_rule,
                    comparisons,
                )

                if not filtered_pairs:
                    self.console.log(
                        "[green]Blocking rule looks good! No known matches were filtered out.[/green]"
                    )
                    return [blocking_rule]
                else:
                    feedback = f"The previous rule incorrectly filtered out {len(filtered_pairs)} known matches. "
                    feedback += (
                        "Here are up to 3 examples of incorrectly filtered pairs:\n"
                    )
                    for i, j in filtered_pairs[:3]:
                        feedback += (
                            f"Left: {json.dumps({left_key: left_data[i][left_key]})}\n"
                        )
                        feedback += f"Right: {json.dumps({right_key: right_data[j][right_key]})}\n"
                        feedback += "These pairs are known matches but were filtered out by the rule.\n"
                    feedback += "Please generate a new rule that doesn't filter out these matches."

                    messages.append({"role": "assistant", "content": blocking_rule})
                    messages.append({"role": "user", "content": feedback})
            else:
                self.console.log("[yellow]No blocking rule generated.[/yellow]")
                return []

        self.console.log(
            f"[yellow]Failed to generate a suitable blocking rule after {self.agent_max_retries} attempts. Proceeding without a blocking rule.[/yellow]"
        )
        return []

    def _test_blocking_rule_equijoin(
        self,
        left_data: List[Dict[str, Any]],
        right_data: List[Dict[str, Any]],
        left_key: str,
        right_key: str,
        blocking_rule: str,
        comparisons: List[Tuple[int, int, bool]],
    ) -> List[Tuple[int, int]]:
        def apply_blocking_rule(left, right):
            try:
                return eval(blocking_rule, {"left": left, "right": right})
            except Exception as e:
                self.console.log(f"[red]Error applying blocking rule: {e}[/red]")
                return True  # If there's an error, we default to comparing the pair

        filtered_pairs = []

        for i, j, is_match in comparisons:
            if is_match:
                left = {left_key: left_data[i][left_key]}
                right = {right_key: right_data[j][right_key]}

                if not apply_blocking_rule(left, right):
                    filtered_pairs.append((i, j))

        if filtered_pairs:
            self.console.log(
                f"[yellow italic]LLM Correction: The blocking rule incorrectly filtered out {len(filtered_pairs)} known positive matches.[/yellow italic]"
            )
            for i, j in filtered_pairs[:5]:  # Show up to 5 examples
                self.console.log(
                    f"  Incorrectly filtered pair - Left: {json.dumps({left_key: left_data[i][left_key]})}  Right: {json.dumps({right_key: right_data[j][right_key]})}"
                )
            if len(filtered_pairs) > 5:
                self.console.log(
                    f"  ... and {len(filtered_pairs) - 5} more incorrect pairs."
                )

        return filtered_pairs

    def _verify_blocking_rule_equijoin(
        self,
        left_data: List[Dict[str, Any]],
        right_data: List[Dict[str, Any]],
        blocking_rule: str,
        left_key: str,
        right_key: str,
        comparison_results: List[Tuple[int, int, bool]],
    ) -> Tuple[List[Tuple[int, int]], float]:
        def apply_blocking_rule(left, right):
            try:
                return eval(blocking_rule, {"left": left, "right": right})
            except Exception as e:
                self.console.log(f"[red]Error applying blocking rule: {e}[/red]")
                return True  # If there's an error, we default to comparing the pair

        false_negatives = []
        total_pairs = 0
        blocked_pairs = 0

        for i, j, is_match in comparison_results:
            total_pairs += 1
            left = {left_key: left_data[i][left_key]}
            right = {right_key: right_data[j][right_key]}

            if apply_blocking_rule(left, right):
                blocked_pairs += 1
                if is_match:
                    false_negatives.append((i, j))

        rule_selectivity = blocked_pairs / total_pairs if total_pairs > 0 else 0

        return false_negatives, rule_selectivity

    def _update_config_equijoin(
        self, threshold: float, left_key: str, right_key: str, blocking_rules: List[str]
    ) -> Dict[str, Any]:
        optimized_config = self.op_config.copy()
        optimized_config["join_key"] = {
            "left": {"name": left_key},
            "right": {"name": right_key},
        }
        optimized_config["blocking_threshold"] = threshold
        if blocking_rules:
            optimized_config["blocking_conditions"] = blocking_rules
        if "embedding_model" not in optimized_config:
            optimized_config["embedding_model"] = "text-embedding-3-small"
        return optimized_config

    def _verify_blocking_rule(
        self,
        input_data: List[Dict[str, Any]],
        blocking_rule: str,
        blocking_keys: List[str],
        comparison_results: List[Tuple[int, int, bool]],
    ) -> Tuple[List[Tuple[int, int]], float]:
        def apply_blocking_rule(item1, item2):
            try:
                return eval(blocking_rule, {"input1": item1, "input2": item2})
            except Exception as e:
                self.console.log(f"[red]Error applying blocking rule: {e}[/red]")
                return True  # If there's an error, we default to comparing the pair

        false_negatives = []
        total_pairs = 0
        blocked_pairs = 0

        for i, j, is_match in comparison_results:
            total_pairs += 1
            item1 = {k: input_data[i][k] for k in blocking_keys if k in input_data[i]}
            item2 = {k: input_data[j][k] for k in blocking_keys if k in input_data[j]}

            if apply_blocking_rule(item1, item2):
                blocked_pairs += 1
                if is_match:
                    false_negatives.append((i, j))

        rule_selectivity = blocked_pairs / total_pairs if total_pairs > 0 else 0

        return false_negatives, rule_selectivity

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
