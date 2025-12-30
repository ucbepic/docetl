"""
Runtime blocking threshold optimization utilities.

This module provides functionality for automatically computing embedding-based
blocking thresholds at runtime when no blocking configuration is provided.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import numpy as np
from litellm import model_cost
from rich.console import Console

from docetl.utils import completion_cost, extract_jinja_variables


class RuntimeBlockingOptimizer:
    """
    Computes optimal embedding-based blocking thresholds at runtime.

    This class samples pairs from the dataset, performs LLM comparisons,
    and finds the optimal cosine similarity threshold that achieves a
    target recall rate.
    """

    def __init__(
        self,
        runner,
        config: dict[str, Any],
        default_model: str,
        max_threads: int,
        console: Console,
        target_recall: float = 0.95,
        sample_size: int = 100,
        sampling_weight: float = 20.0,
    ):
        """
        Initialize the RuntimeBlockingOptimizer.

        Args:
            runner: The pipeline runner instance.
            config: Operation configuration.
            default_model: Default LLM model for comparisons.
            max_threads: Maximum threads for parallel processing.
            console: Rich console for logging.
            target_recall: Target recall rate (default 0.95).
            sample_size: Number of pairs to sample for threshold estimation.
            sampling_weight: Weight for exponential sampling towards higher similarities.
        """
        self.runner = runner
        self.config = config
        self.default_model = default_model
        self.max_threads = max_threads
        self.console = console
        self.target_recall = target_recall
        self.sample_size = sample_size
        self.sampling_weight = sampling_weight

    def compute_embeddings(
        self,
        input_data: list[dict[str, Any]],
        keys: list[str],
        embedding_model: str | None = None,
        batch_size: int = 1000,
    ) -> tuple[list[list[float]], float]:
        """
        Compute embeddings for the input data.

        Args:
            input_data: List of input documents.
            keys: Keys to use for embedding text.
            embedding_model: Model to use for embeddings.
            batch_size: Batch size for embedding computation.

        Returns:
            Tuple of (embeddings list, total cost).
        """
        embedding_model = embedding_model or self.config.get(
            "embedding_model", "text-embedding-3-small"
        )
        model_input_context_length = model_cost.get(embedding_model, {}).get(
            "max_input_tokens", 8192
        )
        texts = [
            " ".join(str(item[key]) for key in keys if key in item)[
                : model_input_context_length * 3
            ]
            for item in input_data
        ]

        self.console.log(f"[cyan]Creating embeddings for {len(texts)} items...[/cyan]")

        embeddings = []
        total_cost = 0.0
        num_batches = (len(texts) + batch_size - 1) // batch_size
        for batch_idx, i in enumerate(range(0, len(texts), batch_size)):
            batch = texts[i : i + batch_size]
            if num_batches > 1:
                self.console.log(
                    f"[dim]  Batch {batch_idx + 1}/{num_batches} "
                    f"({len(embeddings) + len(batch)}/{len(texts)} items)[/dim]"
                )
            response = self.runner.api.gen_embedding(
                model=embedding_model,
                input=batch,
            )
            embeddings.extend([data["embedding"] for data in response["data"]])
            total_cost += completion_cost(response)
        return embeddings, total_cost

    def calculate_cosine_similarities_self(
        self, embeddings: list[list[float]]
    ) -> list[tuple[int, int, float]]:
        """
        Calculate pairwise cosine similarities for self-join.

        Args:
            embeddings: List of embedding vectors.

        Returns:
            List of (i, j, similarity) tuples for all pairs where i < j.
        """
        embeddings_array = np.array(embeddings)
        norms = np.linalg.norm(embeddings_array, axis=1)
        # Avoid division by zero
        norms = np.where(norms == 0, 1e-10, norms)
        dot_products = np.dot(embeddings_array, embeddings_array.T)
        similarities_matrix = dot_products / np.outer(norms, norms)
        i, j = np.triu_indices(len(embeddings), k=1)
        similarities = list(
            zip(i.tolist(), j.tolist(), similarities_matrix[i, j].tolist())
        )
        return similarities

    def calculate_cosine_similarities_cross(
        self,
        left_embeddings: list[list[float]],
        right_embeddings: list[list[float]],
    ) -> list[tuple[int, int, float]]:
        """
        Calculate cosine similarities between two sets of embeddings.

        Args:
            left_embeddings: Embeddings for left dataset.
            right_embeddings: Embeddings for right dataset.

        Returns:
            List of (left_idx, right_idx, similarity) tuples.
        """
        left_array = np.array(left_embeddings)
        right_array = np.array(right_embeddings)
        dot_product = np.dot(left_array, right_array.T)
        norm_left = np.linalg.norm(left_array, axis=1)
        norm_right = np.linalg.norm(right_array, axis=1)
        # Avoid division by zero
        norm_left = np.where(norm_left == 0, 1e-10, norm_left)
        norm_right = np.where(norm_right == 0, 1e-10, norm_right)
        similarities = dot_product / np.outer(norm_left, norm_right)
        return [
            (i, j, float(sim))
            for i, row in enumerate(similarities)
            for j, sim in enumerate(row)
        ]

    def sample_pairs(
        self,
        similarities: list[tuple[int, int, float]],
        num_bins: int = 10,
        stratified_fraction: float = 0.5,
    ) -> list[tuple[int, int]]:
        """
        Sample pairs using a hybrid of stratified and exponential-weighted sampling.

        This ensures coverage across the similarity distribution while still
        focusing on high-similarity pairs where matches are more likely.

        Args:
            similarities: List of (i, j, similarity) tuples.
            num_bins: Number of bins for stratified sampling.
            stratified_fraction: Fraction of samples to allocate to stratified sampling.

        Returns:
            List of sampled (i, j) pairs.
        """
        if len(similarities) == 0:
            return []

        sample_count = min(self.sample_size, len(similarities))
        stratified_count = int(sample_count * stratified_fraction)
        exponential_count = sample_count - stratified_count

        sampled_indices = set()
        sim_values = np.array([sim[2] for sim in similarities])

        # Part 1: Stratified sampling across bins
        if stratified_count > 0:
            bin_edges = np.linspace(
                sim_values.min(), sim_values.max() + 1e-9, num_bins + 1
            )
            samples_per_bin = max(1, stratified_count // num_bins)

            for bin_idx in range(num_bins):
                bin_mask = (sim_values >= bin_edges[bin_idx]) & (
                    sim_values < bin_edges[bin_idx + 1]
                )
                bin_indices = np.where(bin_mask)[0]

                if len(bin_indices) > 0:
                    # Within each bin, use exponential weighting
                    bin_sims = sim_values[bin_indices]
                    bin_weights = np.exp(self.sampling_weight * bin_sims)
                    bin_weights /= bin_weights.sum()

                    n_to_sample = min(samples_per_bin, len(bin_indices))
                    chosen = np.random.choice(
                        bin_indices,
                        size=n_to_sample,
                        replace=False,
                        p=bin_weights,
                    )
                    sampled_indices.update(chosen.tolist())

        # Part 2: Exponential-weighted sampling for remaining slots
        if exponential_count > 0:
            remaining_indices = [
                i for i in range(len(similarities)) if i not in sampled_indices
            ]
            if remaining_indices:
                remaining_sims = sim_values[remaining_indices]
                weights = np.exp(self.sampling_weight * remaining_sims)
                weights /= weights.sum()

                n_to_sample = min(exponential_count, len(remaining_indices))
                chosen = np.random.choice(
                    remaining_indices,
                    size=n_to_sample,
                    replace=False,
                    p=weights,
                )
                sampled_indices.update(chosen.tolist())

        sampled_pairs = [
            (similarities[i][0], similarities[i][1]) for i in sampled_indices
        ]
        return sampled_pairs

    def _print_similarity_histogram(
        self,
        similarities: list[tuple[int, int, float]],
        comparison_results: list[tuple[int, int, bool]],
        threshold: float | None = None,
    ):
        """
        Print a histogram of embedding cosine similarity distribution.

        Args:
            similarities: List of (i, j, similarity) tuples.
            comparison_results: List of (i, j, is_match) from LLM comparisons.
            threshold: Optional threshold to highlight in the histogram.
        """
        # Filter out self-similarities (similarity == 1)
        flat_similarities = [sim[2] for sim in similarities if sim[2] != 1]
        if not flat_similarities:
            return

        hist, bin_edges = np.histogram(flat_similarities, bins=20)
        max_bar_width, max_count = 40, max(hist) if max(hist) > 0 else 1
        normalized_hist = [int(count / max_count * max_bar_width) for count in hist]

        # Create a dictionary to store true labels
        true_labels = {(i, j): is_match for i, j, is_match in comparison_results}

        # Count pairs above threshold
        pairs_above_threshold = (
            sum(1 for sim in flat_similarities if sim >= threshold) if threshold else 0
        )
        total_pairs = len(flat_similarities)

        lines = []
        for i, count in enumerate(normalized_hist):
            bar = "█" * count
            bin_start, bin_end = bin_edges[i], bin_edges[i + 1]
            label = f"{bin_start:.2f}-{bin_end:.2f}"

            # Count true matches and not matches in this bin
            true_matches = 0
            not_matches = 0
            labeled_count = 0
            for sim in similarities:
                if bin_start <= sim[2] < bin_end:
                    if (sim[0], sim[1]) in true_labels:
                        labeled_count += 1
                        if true_labels[(sim[0], sim[1])]:
                            true_matches += 1
                        else:
                            not_matches += 1

            # Calculate percentages of labeled pairs
            if labeled_count > 0:
                true_match_percent = (true_matches / labeled_count) * 100
                label_info = f"[green]{true_match_percent:5.1f}%[/green] match"
            else:
                label_info = "[dim]--[/dim]"

            # Highlight the bin containing the threshold
            if threshold is not None and bin_start <= threshold < bin_end:
                lines.append(
                    f"[bold yellow]{label}[/bold yellow] {bar:<{max_bar_width}} "
                    f"[dim]n={hist[i]:>5}[/dim] {label_info} [bold yellow]◀ threshold[/bold yellow]"
                )
            else:
                lines.append(
                    f"{label} {bar:<{max_bar_width}} "
                    f"[dim]n={hist[i]:>5}[/dim] {label_info}"
                )

        from rich.panel import Panel

        histogram_content = "\n".join(lines)
        title = f"Similarity Distribution ({pairs_above_threshold:,} of {total_pairs:,} pairs ≥ {threshold:.4f})"
        self.console.log(Panel(histogram_content, title=title, border_style="cyan"))

    def find_optimal_threshold(
        self,
        comparisons: list[tuple[int, int, bool]],
        similarities: list[tuple[int, int, float]],
    ) -> tuple[float, float]:
        """
        Find the optimal similarity threshold that achieves target recall.

        Args:
            comparisons: List of (i, j, is_match) from LLM comparisons.
            similarities: List of (i, j, similarity) tuples.

        Returns:
            Tuple of (optimal_threshold, achieved_recall).
        """
        if not comparisons or not any(comp[2] for comp in comparisons):
            # No matches found, use a high threshold to be conservative
            self.console.log(
                "[yellow]No matches found in sample. Using 99th percentile "
                "similarity as threshold.[/yellow]"
            )
            all_sims = [sim[2] for sim in similarities]
            threshold = float(np.percentile(all_sims, 99)) if all_sims else 0.9
            return threshold, 0.0

        true_labels = np.array([comp[2] for comp in comparisons])
        sim_dict = {(i, j): sim for i, j, sim in similarities}
        sim_scores = np.array([sim_dict.get((i, j), 0.0) for i, j, _ in comparisons])
        thresholds = np.linspace(0, 1, 100)
        recalls = []
        for threshold in thresholds:
            predictions = sim_scores >= threshold
            tp = np.sum(predictions & true_labels)
            fn = np.sum(~predictions & true_labels)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            recalls.append(recall)

        # Find highest threshold that achieves target recall
        valid_indices = [i for i, r in enumerate(recalls) if r >= self.target_recall]
        if not valid_indices:
            # If no threshold achieves target recall, use the one with highest recall
            best_idx = int(np.argmax(recalls))
            optimal_threshold = float(thresholds[best_idx])
            achieved_recall = float(recalls[best_idx])
            self.console.log(
                f"[yellow]Warning: Could not achieve target recall {self.target_recall:.0%}. "
                f"Using threshold {optimal_threshold:.4f} with recall {achieved_recall:.2%}.[/yellow]"
            )
        else:
            best_idx = max(valid_indices)
            optimal_threshold = float(thresholds[best_idx])
            achieved_recall = float(recalls[best_idx])

        return round(optimal_threshold, 4), achieved_recall

    def optimize_resolve(
        self,
        input_data: list[dict[str, Any]],
        compare_fn: Callable[[dict, dict], tuple[bool, float, str]],
        blocking_keys: list[str] | None = None,
    ) -> tuple[float, list[list[float]], float]:
        """
        Compute optimal blocking threshold for resolve operation.

        Args:
            input_data: Input dataset.
            compare_fn: Function to compare two items, returns (is_match, cost, prompt).
            blocking_keys: Keys to use for blocking. If None, extracted from prompt.

        Returns:
            Tuple of (threshold, embeddings, total_cost).
        """
        from rich.panel import Panel

        # Determine blocking keys
        if not blocking_keys:
            prompt_template = self.config.get("comparison_prompt", "")
            prompt_vars = extract_jinja_variables(prompt_template)
            prompt_vars = [
                var for var in prompt_vars if var not in ["input", "input1", "input2"]
            ]
            blocking_keys = list(set([var.split(".")[-1] for var in prompt_vars]))
        if not blocking_keys:
            blocking_keys = list(input_data[0].keys())

        # Compute embeddings
        embeddings, embedding_cost = self.compute_embeddings(input_data, blocking_keys)

        # Calculate similarities
        similarities = self.calculate_cosine_similarities_self(embeddings)

        # Sample pairs
        sampled_pairs = self.sample_pairs(similarities)
        if not sampled_pairs:
            self.console.log(
                "[yellow]No pairs to sample. Using default threshold 0.8.[/yellow]"
            )
            return 0.8, embeddings, embedding_cost

        # Perform comparisons
        comparisons = []
        comparison_cost = 0.0
        matches_found = 0
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = {
                executor.submit(compare_fn, input_data[i], input_data[j]): (i, j)
                for i, j in sampled_pairs
            }
            for future in as_completed(futures):
                i, j = futures[future]
                try:
                    is_match, cost, _ = future.result()
                    comparisons.append((i, j, is_match))
                    comparison_cost += cost
                    if is_match:
                        matches_found += 1
                except Exception as e:
                    self.console.log(f"[red]Comparison error: {e}[/red]")
                    comparisons.append((i, j, False))

        # Find optimal threshold
        threshold, achieved_recall = self.find_optimal_threshold(
            comparisons, similarities
        )
        total_cost = embedding_cost + comparison_cost

        # Print histogram visualization
        self._print_similarity_histogram(similarities, comparisons, threshold)

        # Print summary
        n = len(input_data)
        total_pairs = n * (n - 1) // 2
        pairs_above = sum(1 for s in similarities if s[2] >= threshold)

        summary = (
            f"[bold]Blocking keys:[/bold] {blocking_keys}\n"
            f"[bold]Sampled:[/bold] {len(sampled_pairs)} pairs → {matches_found} matches ({matches_found/len(sampled_pairs)*100:.1f}%)\n"
            f"[bold]Threshold:[/bold] {threshold:.4f} → {achieved_recall:.1%} recall (target: {self.target_recall:.0%})\n"
            f"[bold]Pairs to compare:[/bold] {pairs_above:,} of {total_pairs:,} ({pairs_above/total_pairs*100:.1f}%)\n"
            f"[bold]Optimization cost:[/bold] ${total_cost:.4f}"
        )
        self.console.log(
            Panel(
                summary, title="Blocking Threshold Optimization", border_style="green"
            )
        )

        return threshold, embeddings, total_cost

    def optimize_equijoin(
        self,
        left_data: list[dict[str, Any]],
        right_data: list[dict[str, Any]],
        compare_fn: Callable[[dict, dict], tuple[bool, float]],
        left_keys: list[str] | None = None,
        right_keys: list[str] | None = None,
    ) -> tuple[float, list[list[float]], list[list[float]], float]:
        """
        Compute optimal blocking threshold for equijoin operation.

        Args:
            left_data: Left dataset.
            right_data: Right dataset.
            compare_fn: Function to compare two items, returns (is_match, cost).
            left_keys: Keys to use for left dataset embeddings.
            right_keys: Keys to use for right dataset embeddings.

        Returns:
            Tuple of (threshold, left_embeddings, right_embeddings, total_cost).
        """
        from rich.panel import Panel

        # Determine keys
        if not left_keys:
            left_keys = list(left_data[0].keys()) if left_data else []
        if not right_keys:
            right_keys = list(right_data[0].keys()) if right_data else []

        # Compute embeddings
        left_embeddings, left_cost = self.compute_embeddings(left_data, left_keys)
        right_embeddings, right_cost = self.compute_embeddings(right_data, right_keys)
        embedding_cost = left_cost + right_cost

        # Calculate cross similarities
        similarities = self.calculate_cosine_similarities_cross(
            left_embeddings, right_embeddings
        )

        # Sample pairs
        sampled_pairs = self.sample_pairs(similarities)
        if not sampled_pairs:
            self.console.log(
                "[yellow]No pairs to sample. Using default threshold 0.8.[/yellow]"
            )
            return 0.8, left_embeddings, right_embeddings, embedding_cost

        # Perform comparisons
        comparisons = []
        comparison_cost = 0.0
        matches_found = 0
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = {
                executor.submit(compare_fn, left_data[i], right_data[j]): (i, j)
                for i, j in sampled_pairs
            }
            for future in as_completed(futures):
                i, j = futures[future]
                try:
                    is_match, cost = future.result()
                    comparisons.append((i, j, is_match))
                    comparison_cost += cost
                    if is_match:
                        matches_found += 1
                except Exception as e:
                    self.console.log(f"[red]Comparison error: {e}[/red]")
                    comparisons.append((i, j, False))

        # Find optimal threshold
        threshold, achieved_recall = self.find_optimal_threshold(
            comparisons, similarities
        )
        total_cost = embedding_cost + comparison_cost

        # Print histogram visualization
        self._print_similarity_histogram(similarities, comparisons, threshold)

        # Print summary
        total_pairs = len(left_data) * len(right_data)
        pairs_above = sum(1 for s in similarities if s[2] >= threshold)

        summary = (
            f"[bold]Left keys:[/bold] {left_keys}  [bold]Right keys:[/bold] {right_keys}\n"
            f"[bold]Sampled:[/bold] {len(sampled_pairs)} pairs → {matches_found} matches ({matches_found/len(sampled_pairs)*100:.1f}%)\n"
            f"[bold]Threshold:[/bold] {threshold:.4f} → {achieved_recall:.1%} recall (target: {self.target_recall:.0%})\n"
            f"[bold]Pairs to compare:[/bold] {pairs_above:,} of {total_pairs:,} ({pairs_above/total_pairs*100:.1f}%)\n"
            f"[bold]Optimization cost:[/bold] ${total_cost:.4f}"
        )
        self.console.log(
            Panel(
                summary, title="Blocking Threshold Optimization", border_style="green"
            )
        )

        return threshold, left_embeddings, right_embeddings, total_cost
