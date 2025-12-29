"""
Runtime blocking threshold optimization utilities.

This module provides functionality for automatically computing embedding-based
blocking thresholds at runtime when no blocking configuration is provided.
"""

import random
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
        embeddings = []
        total_cost = 0.0
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
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
        self, similarities: list[tuple[int, int, float]]
    ) -> list[tuple[int, int]]:
        """
        Sample pairs weighted towards higher similarity scores.
        
        Args:
            similarities: List of (i, j, similarity) tuples.
            
        Returns:
            List of sampled (i, j) pairs.
        """
        if len(similarities) == 0:
            return []
        # Sort by similarity in descending order
        sorted_similarities = sorted(similarities, key=lambda x: x[2], reverse=True)
        # Calculate weights using exponential weighting
        similarities_array = np.array([sim[2] for sim in sorted_similarities])
        weights = np.exp(self.sampling_weight * similarities_array)
        weights /= weights.sum()  # Normalize
        # Sample pairs
        sample_count = min(self.sample_size, len(sorted_similarities))
        sampled_indices = np.random.choice(
            len(sorted_similarities),
            size=sample_count,
            replace=False,
            p=weights,
        )
        sampled_pairs = [
            (sorted_similarities[i][0], sorted_similarities[i][1])
            for i in sampled_indices
        ]
        return sampled_pairs
    
    def find_optimal_threshold(
        self,
        comparisons: list[tuple[int, int, bool]],
        similarities: list[tuple[int, int, float]],
    ) -> float:
        """
        Find the optimal similarity threshold that achieves target recall.
        
        Args:
            comparisons: List of (i, j, is_match) from LLM comparisons.
            similarities: List of (i, j, similarity) tuples.
            
        Returns:
            Optimal threshold value.
        """
        if not comparisons or not any(comp[2] for comp in comparisons):
            # No matches found, use a high threshold to be conservative
            self.console.log(
                "[yellow]No matches found in sample. Using 99th percentile "
                "similarity as threshold.[/yellow]"
            )
            all_sims = [sim[2] for sim in similarities]
            return float(np.percentile(all_sims, 99)) if all_sims else 0.9
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
            optimal_threshold = float(thresholds[np.argmax(recalls)])
            self.console.log(
                f"[yellow]Could not achieve target recall {self.target_recall:.0%}. "
                f"Using threshold {optimal_threshold:.4f} with recall "
                f"{max(recalls):.2%}.[/yellow]"
            )
        else:
            optimal_threshold = float(thresholds[max(valid_indices)])
            achieved_recall = recalls[max(valid_indices)]
            self.console.log(
                f"[green]Found threshold {optimal_threshold:.4f} achieving "
                f"{achieved_recall:.2%} recall (target: {self.target_recall:.0%}).[/green]"
            )
        return round(optimal_threshold, 4)
    
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
        self.console.log(
            "[cyan]Computing embedding-based blocking threshold automatically...[/cyan]"
        )
        # Determine blocking keys
        if not blocking_keys:
            prompt_template = self.config.get("comparison_prompt", "")
            prompt_vars = extract_jinja_variables(prompt_template)
            prompt_vars = [
                var for var in prompt_vars
                if var not in ["input", "input1", "input2"]
            ]
            blocking_keys = list(set([var.split(".")[-1] for var in prompt_vars]))
        if not blocking_keys:
            blocking_keys = list(input_data[0].keys())
        self.console.log(f"Using blocking keys: {blocking_keys}")
        # Compute embeddings
        embeddings, embedding_cost = self.compute_embeddings(
            input_data, blocking_keys
        )
        self.console.log(
            f"[bold]Embedding cost for threshold optimization: "
            f"${embedding_cost:.4f}[/bold]"
        )
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
        self.console.log(
            f"Comparing {len(sampled_pairs)} sampled pairs to find optimal threshold..."
        )
        comparisons = []
        comparison_cost = 0.0
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
                except Exception as e:
                    self.console.log(f"[red]Comparison error: {e}[/red]")
                    comparisons.append((i, j, False))
        self.console.log(
            f"[bold]Comparison cost for threshold optimization: "
            f"${comparison_cost:.4f}[/bold]"
        )
        # Find optimal threshold
        threshold = self.find_optimal_threshold(comparisons, similarities)
        total_cost = embedding_cost + comparison_cost
        self.console.log(
            f"[bold green]Auto-computed blocking threshold: {threshold} "
            f"(total optimization cost: ${total_cost:.4f})[/bold green]"
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
        self.console.log(
            "[cyan]Computing embedding-based blocking threshold automatically...[/cyan]"
        )
        # Determine keys
        if not left_keys:
            left_keys = list(left_data[0].keys()) if left_data else []
        if not right_keys:
            right_keys = list(right_data[0].keys()) if right_data else []
        self.console.log(f"Using left keys: {left_keys}, right keys: {right_keys}")
        # Compute embeddings
        left_embeddings, left_cost = self.compute_embeddings(left_data, left_keys)
        right_embeddings, right_cost = self.compute_embeddings(right_data, right_keys)
        embedding_cost = left_cost + right_cost
        self.console.log(
            f"[bold]Embedding cost for threshold optimization: "
            f"${embedding_cost:.4f}[/bold]"
        )
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
        self.console.log(
            f"Comparing {len(sampled_pairs)} sampled pairs to find optimal threshold..."
        )
        comparisons = []
        comparison_cost = 0.0
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
                except Exception as e:
                    self.console.log(f"[red]Comparison error: {e}[/red]")
                    comparisons.append((i, j, False))
        self.console.log(
            f"[bold]Comparison cost for threshold optimization: "
            f"${comparison_cost:.4f}[/bold]"
        )
        # Find optimal threshold
        threshold = self.find_optimal_threshold(comparisons, similarities)
        total_cost = embedding_cost + comparison_cost
        self.console.log(
            f"[bold green]Auto-computed blocking threshold: {threshold} "
            f"(total optimization cost: ${total_cost:.4f})[/bold green]"
        )
        return threshold, left_embeddings, right_embeddings, total_cost
