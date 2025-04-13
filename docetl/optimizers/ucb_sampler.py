"""
This module provides UCB (Upper Confidence Bound) sampling utilities
for intelligent exploration and exploitation in pipeline optimization.
"""

import copy
import math
import random
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from rich.console import Console

from docetl.optimizers.pipeline_utils import (
    execute_pipeline,
    rank_pipeline_outputs_with_llm,
)
from docetl.optimizers.utils import LLMClient


class UCBSampler:
    """
    Implements UCB (Upper Confidence Bound) algorithm for pipeline sampling.

    This class maintains state about pipeline rewards and samples to enable
    intelligent exploration and exploitation when evaluating pipeline alternatives.
    """

    def __init__(
        self,
        run_operation_func: Callable,
        console: Console,
        exploration_weight: float = 2.0,
        dampening_factor: float = 0.8,
    ):
        """
        Initialize the UCB sampler.

        Args:
            run_operation_func: Function to execute operations in the pipeline
            console: Console for logging
            exploration_weight: Weight for the exploration term in UCB formula (default: 2.0)
            dampening_factor: Factor to dampen rewards when propagating to related pipelines (default: 0.8)
        """
        # Pipeline state tracking
        self.pipeline_rewards = {}  # Maps pipeline hash to cumulative reward
        self.pipeline_samples = {}  # Maps pipeline hash to number of samples
        self.skeleton_to_pipelines = {}  # Maps skeleton hash to list of pipeline hashes
        self.pipeline_to_skeleton = {}  # Maps pipeline hash to its skeleton hash
        self.sampled_pipelines = (
            set()
        )  # Set of pipeline hashes that have already been sampled
        self.sampled_skeletons = (
            set()
        )  # Set of skeleton hashes that have already been sampled
        self.estimated_rewards = {}  # Maps pipeline hash to initial estimated reward

        # Configuration
        self.exploration_weight = exploration_weight
        self.dampening_factor = dampening_factor
        self.total_samples = 0

        # Runner components
        self.run_operation_func = run_operation_func
        self.console = console

    def _hash_pipeline(self, pipeline: List[Dict[str, Any]]) -> str:
        """
        Create a unique hash for a pipeline configuration.

        Args:
            pipeline: Pipeline configuration to hash

        Returns:
            String hash identifying the pipeline
        """
        # Simple string-based hash for now - can be improved later
        pipeline_str = str(pipeline)
        return str(hash(pipeline_str) % 10000000)

    def _hash_skeleton(self, skeleton: Any) -> str:
        """
        Create a unique hash for a pipeline skeleton.

        Args:
            skeleton: Pipeline skeleton to hash

        Returns:
            String hash identifying the skeleton
        """
        # Simple string-based hash - can be improved later
        skeleton_str = str(skeleton)
        return str(hash(skeleton_str) % 10000000)

    def initialize_pipelines(
        self,
        skeleton_to_pipelines: Mapping,
        scoring_func: Optional[Callable] = None,
        sample_docs: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Initialize tracking for a set of pipelines organized by skeleton.
        If scoring_func and sample_docs are provided, estimates initial rewards.

        Args:
            skeleton_to_pipelines: Mapping of skeleton objects to lists of (pipeline, estimated_cost) tuples
            scoring_func: Optional function to score pipeline outputs (higher is better)
            sample_docs: Optional sample documents to use for initial reward estimation
        """
        # Reset state
        self.pipeline_rewards = {}
        self.pipeline_samples = {}
        self.skeleton_to_pipelines = {}
        self.pipeline_to_skeleton = {}
        self.sampled_pipelines = set()
        self.sampled_skeletons = set()
        self.estimated_rewards = {}
        self.total_samples = 0

        # Process each skeleton and its pipelines
        for skeleton, pipelines in skeleton_to_pipelines.items():
            skeleton_hash = self._hash_skeleton(skeleton)
            self.skeleton_to_pipelines[skeleton_hash] = []

            for pipeline, _ in pipelines:
                pipeline_hash = self._hash_pipeline(pipeline)
                self.pipeline_rewards[pipeline_hash] = 0
                self.pipeline_samples[pipeline_hash] = 0
                self.skeleton_to_pipelines[skeleton_hash].append(pipeline_hash)
                self.pipeline_to_skeleton[pipeline_hash] = skeleton_hash

        # Estimate initial rewards if scoring function and sample docs are provided
        if scoring_func and sample_docs and len(sample_docs) > 0:
            self._estimate_initial_rewards(
                skeleton_to_pipelines, scoring_func, sample_docs
            )

    def _estimate_initial_rewards(
        self,
        skeleton_to_pipelines: Mapping,
        scoring_func: Callable,
        sample_docs: List[Dict[str, Any]],
    ) -> None:
        """
        Estimate initial rewards for pipelines using static analysis and/or lightweight runs.

        Args:
            skeleton_to_pipelines: Mapping of skeleton objects to lists of (pipeline, estimated_cost) tuples
            scoring_func: Function to score pipeline outputs (higher is better)
            sample_docs: Sample documents to use for estimation
        """
        self.console.log("Estimating initial rewards for UCB sampling...")

        # Use a single sample document to reduce overhead
        estimation_doc = sample_docs[0] if sample_docs else None
        if not estimation_doc:
            self.console.log(
                "No sample documents available for initial reward estimation"
            )
            return

        # For each pipeline, create a rough estimate
        for skeleton, pipelines in skeleton_to_pipelines.items():
            for pipeline, estimated_cost in pipelines:
                pipeline_hash = self._hash_pipeline(pipeline)

                # Use pipeline characteristics to estimate quality
                initial_reward = 0.0

                # Factor 1: Operation count - more operations might be more complete
                # but could also introduce more error, so use a curve that peaks
                op_count = len(pipeline)
                if op_count > 0:
                    # Operations increase reward up to a point, then decrease
                    op_factor = min(op_count / 5.0, 2.0 - (op_count / 10.0))
                    op_factor = max(
                        0.1, min(1.0, op_factor)
                    )  # Clamp between 0.1 and 1.0
                    initial_reward += op_factor * 0.3  # 30% weight to operation count

                # Factor 2: Estimated cost - higher cost might correlate with quality
                # but with diminishing returns
                if estimated_cost > 0:
                    # Log scale for cost to account for diminishing returns
                    cost_factor = min(1.0, 0.1 + 0.3 * math.log(1 + estimated_cost))
                    initial_reward += cost_factor * 0.2  # 20% weight to cost

                # Add some randomness to encourage exploration
                random_factor = random.uniform(0.01, 0.1)
                initial_reward += random_factor

                # Store the estimated reward
                self.estimated_rewards[pipeline_hash] = initial_reward

                # Log the estimate
                if (
                    len(self.estimated_rewards) <= 5 or random.random() < 0.05
                ):  # Log only some to avoid spam
                    self.console.log(
                        f"Pipeline {pipeline_hash[:8]} initial reward estimate: {initial_reward:.4f}"
                    )

        self.console.log(
            f"Estimated initial rewards for {len(self.estimated_rewards)} pipelines"
        )

    def calculate_ucb_score(self, pipeline_hash: str) -> float:
        """
        Calculate the UCB score for a pipeline.

        Args:
            pipeline_hash: Hash of the pipeline to score

        Returns:
            UCB score (higher means more promising for sampling)
            Returns -inf for already sampled pipelines to exclude them
        """
        # If already sampled, return negative infinity to exclude
        if pipeline_hash in self.sampled_pipelines:
            return float("-inf")

        # If never sampled, use the estimated reward plus a large exploration bonus
        if self.pipeline_samples.get(pipeline_hash, 0) == 0:
            estimated_reward = self.estimated_rewards.get(
                pipeline_hash, 0.5
            )  # Default to 0.5 if no estimate
            exploration_bonus = self.exploration_weight * math.sqrt(
                2
            )  # Larger bonus for unsampled
            return estimated_reward + exploration_bonus

        # Get average reward for this pipeline
        reward = self.pipeline_rewards.get(pipeline_hash, 0)
        samples = self.pipeline_samples.get(pipeline_hash, 0)
        avg_reward = reward / samples if samples > 0 else 0

        # Calculate exploration term
        exploration = math.sqrt(
            self.exploration_weight * math.log(self.total_samples) / samples
        )

        # UCB score = average reward + exploration term
        return avg_reward + exploration

    def update_reward(self, pipeline: List[Dict[str, Any]], reward: float) -> None:
        """
        Update rewards for a pipeline and propagate to related pipelines.

        Args:
            pipeline: Pipeline that was executed
            reward: Reward value for this pipeline execution
        """
        pipeline_hash = self._hash_pipeline(pipeline)

        # Mark this pipeline as sampled
        self.sampled_pipelines.add(pipeline_hash)

        # Mark this skeleton as sampled
        skeleton_hash = self.pipeline_to_skeleton.get(pipeline_hash)
        if skeleton_hash:
            self.sampled_skeletons.add(skeleton_hash)

        # Update pipeline's own reward and sample count
        self.pipeline_rewards[pipeline_hash] = (
            self.pipeline_rewards.get(pipeline_hash, 0) + reward
        )
        self.pipeline_samples[pipeline_hash] = (
            self.pipeline_samples.get(pipeline_hash, 0) + 1
        )
        self.total_samples += 1

        # Propagate dampened reward to other pipelines in the same skeleton
        if skeleton_hash:
            related_pipelines = self.skeleton_to_pipelines.get(skeleton_hash, [])
            for related_hash in related_pipelines:
                if related_hash != pipeline_hash:
                    # Add dampened reward but don't increment sample count
                    dampened_reward = reward * self.dampening_factor
                    self.pipeline_rewards[related_hash] = (
                        self.pipeline_rewards.get(related_hash, 0) + dampened_reward
                    )

    def get_unsampled_pipeline_count(self, skeleton_to_pipelines: Mapping) -> int:
        """
        Count the number of unsampled pipelines remaining.

        Args:
            skeleton_to_pipelines: Mapping of skeleton objects to lists of (pipeline, estimated_cost) tuples

        Returns:
            Number of pipelines that have not been sampled yet
        """
        total_pipelines = 0
        sampled_count = 0

        for skeleton, pipelines in skeleton_to_pipelines.items():
            total_pipelines += len(pipelines)
            for pipeline, _ in pipelines:
                pipeline_hash = self._hash_pipeline(pipeline)
                if pipeline_hash in self.sampled_pipelines:
                    sampled_count += 1

        return total_pipelines - sampled_count

    def get_unsampled_skeleton_count(self, skeleton_to_pipelines: Mapping) -> int:
        """
        Count the number of skeletons that have not had any pipeline sampled yet.

        Args:
            skeleton_to_pipelines: Mapping of skeleton objects to lists of (pipeline, estimated_cost) tuples

        Returns:
            Number of skeletons that have not been sampled yet
        """
        total_skeletons = len(skeleton_to_pipelines)
        sampled_skeletons = len(self.sampled_skeletons)

        return total_skeletons - sampled_skeletons

    def sample_pipeline_from_unsampled_skeleton(
        self, skeleton_to_pipelines: Mapping
    ) -> Tuple[Any, List[Dict[str, Any]], float]:
        """
        Sample a pipeline from a skeleton that has not been sampled yet.

        Args:
            skeleton_to_pipelines: Mapping of skeleton objects to lists of (pipeline, estimated_cost) tuples

        Returns:
            Tuple containing:
              - The selected skeleton
              - The selected pipeline configuration
              - Estimated cost of the selected pipeline

        Raises:
            ValueError: If all skeletons have been sampled or no pipelines are available
        """
        unsampled_skeletons = []

        # Find skeletons that haven't been sampled yet
        for skeleton in skeleton_to_pipelines.keys():
            skeleton_hash = self._hash_skeleton(skeleton)
            if skeleton_hash not in self.sampled_skeletons:
                unsampled_skeletons.append((skeleton, skeleton_hash))

        if not unsampled_skeletons:
            raise ValueError("All skeletons have already been sampled")

        # Randomly select an unsampled skeleton
        selected_skeleton, selected_skeleton_hash = random.choice(unsampled_skeletons)

        # Get pipelines for this skeleton
        pipelines = skeleton_to_pipelines[selected_skeleton]

        # Find unsampled pipelines for this skeleton
        unsampled_pipelines = []
        for pipeline, estimated_cost in pipelines:
            pipeline_hash = self._hash_pipeline(pipeline)
            if pipeline_hash not in self.sampled_pipelines:
                unsampled_pipelines.append((pipeline, pipeline_hash, estimated_cost))

        if not unsampled_pipelines:
            # This should be rare, as it means all pipelines from this skeleton
            # were somehow sampled but the skeleton wasn't marked as sampled
            self.sampled_skeletons.add(selected_skeleton_hash)
            return self.sample_pipeline_from_unsampled_skeleton(skeleton_to_pipelines)

        # Randomly select an unsampled pipeline from this skeleton
        selected_pipeline, _, selected_cost = random.choice(unsampled_pipelines)

        return selected_skeleton, selected_pipeline, selected_cost

    def sample_pipeline(
        self, skeleton_to_pipelines: Mapping
    ) -> Tuple[Any, List[Dict[str, Any]], float]:
        """
        Sample a pipeline using UCB algorithm, never sampling the same pipeline twice.

        Args:
            skeleton_to_pipelines: Mapping of skeleton objects to lists of (pipeline, estimated_cost) tuples

        Returns:
            Tuple containing:
              - The selected skeleton
              - The selected pipeline configuration
              - Estimated cost of the selected pipeline

        Raises:
            ValueError: If all pipelines have been sampled or no pipelines are available
        """
        # Check if we've already sampled all pipelines
        unsampled_count = self.get_unsampled_pipeline_count(skeleton_to_pipelines)
        if unsampled_count == 0:
            raise ValueError("All pipelines have already been sampled")

        # Map of skeleton objects to their hashes
        skeleton_hash_map = {
            self._hash_skeleton(skeleton): skeleton
            for skeleton in skeleton_to_pipelines.keys()
        }

        # Calculate UCB scores for all pipelines
        pipeline_scores = {}
        pipeline_objects = {}
        pipeline_costs = {}

        for skeleton, pipelines in skeleton_to_pipelines.items():
            for pipeline, estimated_cost in pipelines:
                pipeline_hash = self._hash_pipeline(pipeline)

                # Skip already sampled pipelines
                if pipeline_hash in self.sampled_pipelines:
                    continue

                pipeline_scores[pipeline_hash] = self.calculate_ucb_score(pipeline_hash)
                pipeline_objects[pipeline_hash] = pipeline
                pipeline_costs[pipeline_hash] = estimated_cost

        # Select pipeline with highest UCB score
        if not pipeline_scores:
            raise ValueError("No unsampled pipelines available")

        best_pipeline_hash = max(pipeline_scores, key=pipeline_scores.get)
        selected_pipeline = pipeline_objects[best_pipeline_hash]
        selected_cost = pipeline_costs[best_pipeline_hash]

        # Determine which skeleton this pipeline belongs to
        skeleton_hash = self.pipeline_to_skeleton.get(best_pipeline_hash)
        selected_skeleton = skeleton_hash_map.get(skeleton_hash)

        return selected_skeleton, selected_pipeline, selected_cost

    def execute_pipeline(
        self,
        input_docs: List[Dict[str, Any]],
        pipeline: List[Dict[str, Any]],
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Executes a pipeline using the run_operation_func provided at initialization.

        Args:
            input_docs: Input documents to process
            pipeline: Pipeline configuration to execute

        Returns:
            Tuple of (total_cost, output_documents)
        """
        return execute_pipeline(
            input_docs, pipeline, self.run_operation_func, self.console
        )


def sample_pipeline_execution_with_ucb(
    sample_docs: List[Dict[str, Any]],
    console: Console,
    skeleton_to_pipelines: Mapping,
    run_operation_func: Callable,
    llm_client: LLMClient,
    original_pipeline_config: Dict[str, Any],
    scoring_func: Callable,
    sample_size: int = 5,
    budget_num_pipelines: int = 20,
    exploration_weight: float = 2.0,
    dampening_factor: float = 0.8,
) -> Tuple[
    List[Tuple[List[Dict[str, Any]], float, List[Dict[str, Any]], Dict[str, Any]]],
    float,
]:
    """
    Samples execution of pipelines using UCB algorithm to intelligently explore and exploit
    the space of possible pipelines. Each pipeline is sampled at most once, with at least one
    pipeline from each skeleton sampled first before UCB selection begins.

    Args:
        sample_docs: A list of sample documents to use as input
        console: Console for logging
        skeleton_to_pipelines: Mapping of skeleton objects to lists of (pipeline, estimated_cost) tuples
        run_operation_func: Reference to the runner's _run_operation function
        llm_client: LLM client for querying and ranking
        original_pipeline_config: The original pipeline configuration
        scoring_func: Function to score pipeline outputs (higher is better)
        sample_size: Number of documents to use in sampling (default: 5)
        budget_num_pipelines: Maximum number of pipelines to sample (default: 20)
        exploration_weight: Weight for exploration in UCB formula (default: 2.0)
        dampening_factor: Factor to dampen rewards when propagating (default: 0.8)

    Returns:
        Tuple containing:
          - List of tuples, each containing (pipeline configuration, actual cost, output_docs, ranking_info)
          - Total sampling cost
    """
    # Initialize UCB sampler
    ucb = UCBSampler(
        run_operation_func=run_operation_func,
        console=console,
        exploration_weight=exploration_weight,
        dampening_factor=dampening_factor,
    )

    # Pass the scoring function and sample docs for initial reward estimation
    ucb.initialize_pipelines(
        skeleton_to_pipelines,
        scoring_func=scoring_func,
        sample_docs=sample_docs[:1],  # Use just one document for estimation
    )

    results = []

    # Use a limited number of docs for sampling
    input_docs = copy.deepcopy(sample_docs[:sample_size])
    console.log(f"Sampling with {len(input_docs)} documents using UCB algorithm")
    console.log("Each pipeline will be sampled at most once")
    sampling_cost = 0

    # Calculate the total number of available pipelines and skeletons
    total_pipelines = sum(
        len(pipelines) for pipelines in skeleton_to_pipelines.values()
    )
    total_skeletons = len(skeleton_to_pipelines)
    actual_budget = min(budget_num_pipelines, total_pipelines)

    console.log(f"Total available pipelines: {total_pipelines}")
    console.log(f"Total available skeletons: {total_skeletons}")
    console.log(f"Sampling budget: {actual_budget}")

    # PHASE 1: Sample at least one pipeline from each skeleton
    console.log("\n--- PHASE 1: Sampling one pipeline from each skeleton ---")
    sampled_in_phase1 = 0

    while sampled_in_phase1 < min(total_skeletons, actual_budget):
        try:
            # Sample a pipeline from an unsampled skeleton
            skeleton, pipeline, estimated_cost = (
                ucb.sample_pipeline_from_unsampled_skeleton(skeleton_to_pipelines)
            )

            # Log the selected pipeline
            console.log(
                f"PHASE 1 - Sampling from unsampled skeleton {sampled_in_phase1+1}/{total_skeletons}"
            )
            console.log(f"Selected pipeline for skeleton: {str(skeleton)[:50]}...")
            console.log(
                f"Operations: {len(pipeline)}, Estimated cost: ${estimated_cost:.6f}"
            )

            # Execute the pipeline on sample docs
            actual_cost, output_docs = ucb.execute_pipeline(input_docs, pipeline)

            # Log the cost comparison
            cost_diff = actual_cost - estimated_cost
            cost_diff_pct = (
                (cost_diff / estimated_cost) * 100
                if estimated_cost > 0
                else float("inf")
            )

            console.log(
                f"  Actual cost: ${actual_cost:.6f} ({len(output_docs)} output docs)"
            )
            console.log(f"  Cost difference: ${cost_diff:.6f} ({cost_diff_pct:.1f}%)")

            # Calculate reward from scoring function
            reward = 0
            if output_docs:
                try:
                    # Calculate reward as average of scoring function across documents
                    doc_rewards = [scoring_func(doc) for doc in output_docs]
                    reward = sum(doc_rewards) / len(doc_rewards)
                    console.log(f"  Pipeline reward: {reward:.4f}")
                except Exception as e:
                    console.log(f"  Error calculating reward: {e}")

            # Update UCB sampler with reward
            ucb.update_reward(pipeline, reward)

            # Store results for this pipeline if there are output docs
            if len(output_docs) > 0:
                results.append((pipeline, actual_cost, output_docs))
            sampling_cost += actual_cost
            sampled_in_phase1 += 1

            # Log remaining unsampled skeletons
            unsampled_skeleton_count = ucb.get_unsampled_skeleton_count(
                skeleton_to_pipelines
            )
            console.log(f"  Remaining unsampled skeletons: {unsampled_skeleton_count}")

            # Check if we've exhausted our budget
            if sampled_in_phase1 >= actual_budget:
                console.log("Sampling budget exhausted during Phase 1")
                break

        except ValueError:
            console.log("All skeletons have been sampled. Moving to Phase 2.")
            break
        except Exception as e:
            console.log(f"Error in Phase 1 sampling: {e}")
            break

    # PHASE 2: Use UCB algorithm to sample remaining budget
    remaining_budget = actual_budget - sampled_in_phase1

    if remaining_budget > 0:
        console.log(
            f"\n--- PHASE 2: UCB sampling for remaining budget ({remaining_budget}) ---"
        )

        for iteration in range(remaining_budget):
            try:
                # Sample a pipeline using UCB
                skeleton, pipeline, estimated_cost = ucb.sample_pipeline(
                    skeleton_to_pipelines
                )

                # Log the selected pipeline
                console.log(
                    f"PHASE 2 - UCB Sampling iteration {iteration+1}/{remaining_budget}"
                )
                console.log(f"Selected pipeline for skeleton: {str(skeleton)[:50]}...")
                console.log(
                    f"Operations: {len(pipeline)}, Estimated cost: ${estimated_cost:.6f}"
                )

                # Execute the pipeline on sample docs
                actual_cost, output_docs = ucb.execute_pipeline(input_docs, pipeline)

                # Log the cost comparison
                cost_diff = actual_cost - estimated_cost
                cost_diff_pct = (
                    (cost_diff / estimated_cost) * 100
                    if estimated_cost > 0
                    else float("inf")
                )

                console.log(
                    f"  Actual cost: ${actual_cost:.6f} ({len(output_docs)} output docs)"
                )
                console.log(
                    f"  Cost difference: ${cost_diff:.6f} ({cost_diff_pct:.1f}%)"
                )

                # Calculate reward from scoring function
                reward = 0
                if output_docs:
                    try:
                        # Calculate reward as average of scoring function across documents
                        doc_rewards = [scoring_func(doc) for doc in output_docs]
                        reward = sum(doc_rewards) / len(doc_rewards)
                        console.log(f"  Pipeline reward: {reward:.4f}")
                    except Exception as e:
                        console.log(f"  Error calculating reward: {e}")

                # Update UCB sampler with reward
                ucb.update_reward(pipeline, reward)

                # Store results for this pipeline if there are output docs
                if len(output_docs) > 0:
                    results.append((pipeline, actual_cost, output_docs))
                sampling_cost += actual_cost

                # Log remaining unsampled pipelines
                unsampled_count = ucb.get_unsampled_pipeline_count(
                    skeleton_to_pipelines
                )
                console.log(f"  Remaining unsampled pipelines: {unsampled_count}")

            except ValueError as e:
                console.log(f"Stopping UCB sampling: {str(e)}")
                break
            except Exception as e:
                console.log(f"Error in UCB sampling iteration: {e}")
                continue
    else:
        console.log("No remaining budget for Phase 2 UCB sampling")

    # Rank the results using LLM as a judge and scoring function
    console.log(f"Ranking {len(results)} pipeline results based on output quality...")
    ranked_results = rank_pipeline_outputs_with_llm(
        results, original_pipeline_config, llm_client, console, scoring_func
    )

    return ranked_results, sampling_cost
