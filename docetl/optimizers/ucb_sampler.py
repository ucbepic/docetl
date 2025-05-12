"""
This module provides UCB (Upper Confidence Bound) sampling utilities
for intelligent exploration and exploitation in pipeline optimization.
"""

import copy
import math
import random
from typing import Any, Callable, Dict, List, Mapping, Set, Tuple

from rich.console import Console

from docetl.optimizers.pipeline_utils import (
    execute_pipeline,
    rank_pipeline_outputs_with_llm,
)
from docetl.optimizers.utils import LLMClient


class UCBSampler:
    """
    Implements UCB (Upper Confidence Bound) algorithm for pipeline sampling.

    This class maintains state about pipeline rewards (direct, skeleton-based, model-based)
    and samples to enable intelligent exploration and exploitation when evaluating
    pipeline alternatives.
    """

    def __init__(
        self,
        run_operation_func: Callable,
        console: Console,
        exploration_weight: float = 2.0,
        direct_reward_weight: float = 1.0,
        skeleton_reward_weight: float = 0.5,
        model_reward_weight: float = 0.3,
    ):
        """
        Initialize the UCB sampler.

        Args:
            run_operation_func: Function to execute operations in the pipeline.
            console: Console for logging.
            exploration_weight: Weight for the exploration term in UCB formula.
            direct_reward_weight: Weight for the pipeline's direct average reward in UCB score.
            skeleton_reward_weight: Weight for the pipeline's average skeleton peer reward.
            model_reward_weight: Weight for the pipeline's average model peer reward.
        """
        # Pipeline state tracking
        self.pipeline_direct_rewards: Dict[str, float] = {}
        self.pipeline_direct_samples: Dict[str, int] = {}
        self.pipeline_skeleton_rewards: Dict[str, float] = {}
        self.pipeline_skeleton_samples: Dict[str, int] = {}
        self.pipeline_model_rewards: Dict[str, float] = {}
        self.pipeline_model_samples: Dict[str, int] = {}

        self.skeleton_map: Dict[str, List[str]] = (
            {}
        )  # Skeleton hash -> List[pipeline hashes]
        self.model_map: Dict[str, List[str]] = {}  # Model name -> List[pipeline hashes]
        self.pipeline_to_skeleton_map: Dict[str, str] = (
            {}
        )  # Pipeline hash -> Skeleton hash
        self.pipeline_to_models_map: Dict[str, List[str]] = (
            {}
        )  # Pipeline hash -> List[model names]

        self.sampled_pipelines: Set[str] = (
            set()
        )  # Set of pipeline hashes directly executed
        self.total_samples: int = 0  # Total number of direct pipeline executions

        # Configuration
        self.exploration_weight = exploration_weight
        self.direct_reward_weight = direct_reward_weight
        self.skeleton_reward_weight = skeleton_reward_weight
        self.model_reward_weight = model_reward_weight

        # Runner components
        self.run_operation_func = run_operation_func
        self.console = console
        # Large constant for initial exploration bonus for un-sampled pipelines
        self.INITIAL_EXPLORATION_BONUS = 1e6  # 1,000,000

    def _hash_pipeline(self, pipeline: List[Dict[str, Any]]) -> str:
        """
        Create a unique hash for a pipeline configuration.

        Args:
            pipeline: Pipeline configuration to hash

        Returns:
            String hash identifying the pipeline
        """
        # Simple string-based hash for now - can be improved later
        pipeline_str = str(sorted(pipeline, key=lambda op: op.get("name", "")))
        return str(hash(pipeline_str))

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
        return str(hash(skeleton_str))

    def initialize_pipelines(
        self,
        skeleton_to_pipelines: Mapping,
    ) -> None:
        """
        Initialize tracking for a set of pipelines organized by skeleton.
        Populates internal maps and resets reward/sample counts.

        Args:
            skeleton_to_pipelines: Mapping of skeleton objects to lists of (pipeline, estimated_cost) tuples
        """
        # Reset state
        self.pipeline_direct_rewards = {}
        self.pipeline_direct_samples = {}
        self.pipeline_skeleton_rewards = {}
        self.pipeline_skeleton_samples = {}
        self.pipeline_model_rewards = {}
        self.pipeline_model_samples = {}
        self.skeleton_map = {}
        self.model_map = {}
        self.pipeline_to_skeleton_map = {}
        self.pipeline_to_models_map = {}
        self.sampled_pipelines = set()
        self.total_samples = 0

        # Process each skeleton and its pipelines
        for skeleton, pipelines in skeleton_to_pipelines.items():
            skeleton_hash = self._hash_skeleton(skeleton)
            self.skeleton_map[skeleton_hash] = []

            for pipeline, _ in pipelines:  # Ignore estimated_cost
                pipeline_hash = self._hash_pipeline(pipeline)

                # Initialize rewards and samples
                self.pipeline_direct_rewards[pipeline_hash] = 0.0
                self.pipeline_direct_samples[pipeline_hash] = 0
                self.pipeline_skeleton_rewards[pipeline_hash] = 0.0
                self.pipeline_skeleton_samples[pipeline_hash] = 0
                self.pipeline_model_rewards[pipeline_hash] = 0.0
                self.pipeline_model_samples[pipeline_hash] = 0

                # Populate skeleton maps
                self.skeleton_map[skeleton_hash].append(pipeline_hash)
                self.pipeline_to_skeleton_map[pipeline_hash] = skeleton_hash

                # Extract models and populate model maps
                pipeline_models: Set[str] = set()
                for operation in pipeline:
                    model_name = operation.get("model")
                    if model_name and isinstance(model_name, str):
                        pipeline_models.add(model_name)
                        if model_name not in self.model_map:
                            self.model_map[model_name] = []
                        # Avoid adding duplicates to model_map list
                        if pipeline_hash not in self.model_map[model_name]:
                            self.model_map[model_name].append(pipeline_hash)

                self.pipeline_to_models_map[pipeline_hash] = list(pipeline_models)

        self.console.log(
            f"Initialized UCB Sampler with {len(self.pipeline_direct_rewards)} pipelines."
        )
        self.console.log(f"Found {len(self.skeleton_map)} skeletons.")
        self.console.log(f"Found {len(self.model_map)} unique models.")

    def calculate_ucb_score(self, pipeline_hash: str) -> float:
        """
        Calculate the UCB score for a pipeline based on direct, skeleton, and model rewards.
        Prioritizes un-sampled pipelines based on peer rewards.

        Args:
            pipeline_hash: Hash of the pipeline to score

        Returns:
            UCB score (higher means more promising for sampling)
            Returns -inf for already directly sampled pipelines to exclude them
        """
        # If already directly sampled, return negative infinity to exclude
        if pipeline_hash in self.sampled_pipelines:
            return float("-inf")

        direct_samples = self.pipeline_direct_samples.get(pipeline_hash, 0)

        # Calculate average peer rewards (handle division by zero)
        skeleton_samples = self.pipeline_skeleton_samples.get(pipeline_hash, 0)
        skeleton_reward = self.pipeline_skeleton_rewards.get(pipeline_hash, 0.0)
        avg_skeleton = (
            skeleton_reward / skeleton_samples if skeleton_samples > 0 else 0.0
        )

        model_samples = self.pipeline_model_samples.get(pipeline_hash, 0)
        model_reward = self.pipeline_model_rewards.get(pipeline_hash, 0.0)
        avg_model = model_reward / model_samples if model_samples > 0 else 0.0

        # Calculate weighted peer reward component (used in both cases)
        peer_reward_component = (
            self.skeleton_reward_weight * avg_skeleton
            + self.model_reward_weight * avg_model
        )

        # If never directly sampled, score is based only on peer rewards + tie-breaker
        if direct_samples == 0:
            # Add small random value to break ties among un-sampled pipelines
            # Using a very small multiplier to ensure peer reward is the primary factor
            base_score = peer_reward_component  # Reverted
            return base_score + random.random() * 1e-6

        # --- Calculate score for pipelines that HAVE been directly sampled ---

        # Calculate average direct reward
        direct_reward = self.pipeline_direct_rewards.get(pipeline_hash, 0.0)
        avg_direct = direct_reward / direct_samples  # direct_samples > 0 here

        # Calculate weighted exploitation term (direct + peer rewards)
        exploitation_term = (
            self.direct_reward_weight * avg_direct
            + peer_reward_component  # Reuse calculated peer component
        )

        # Calculate exploration term (based on direct samples)
        # Ensure total_samples is at least 1 for log calculation
        safe_total_samples = max(1, self.total_samples)
        exploration_term = math.sqrt(
            self.exploration_weight * math.log(safe_total_samples) / direct_samples
        )

        # UCB score = weighted exploitation + exploration
        return exploitation_term + exploration_term

    def update_reward(self, pipeline: List[Dict[str, Any]], reward: float) -> None:
        """
        Update rewards for a directly executed pipeline and propagate the full
        reward to its skeleton and model peers.

        Args:
            pipeline: Pipeline configuration that was executed
            reward: Reward value obtained from this pipeline execution
        """
        pipeline_hash = self._hash_pipeline(pipeline)

        # --- 1. Update Direct Reward and Samples ---
        self.pipeline_direct_rewards[pipeline_hash] = (
            self.pipeline_direct_rewards.get(pipeline_hash, 0.0) + reward
        )
        self.pipeline_direct_samples[pipeline_hash] = (
            self.pipeline_direct_samples.get(pipeline_hash, 0) + 1
        )
        self.total_samples += 1
        self.sampled_pipelines.add(pipeline_hash)  # Mark as directly sampled

        # --- 2. Propagate Skeleton Reward ---
        skeleton_hash = self.pipeline_to_skeleton_map.get(pipeline_hash)
        if skeleton_hash:
            skeleton_peers = self.skeleton_map.get(skeleton_hash, [])
            for peer_hash in skeleton_peers:
                if peer_hash != pipeline_hash:  # Don't update itself
                    self.pipeline_skeleton_rewards[peer_hash] = (
                        self.pipeline_skeleton_rewards.get(peer_hash, 0.0) + reward
                    )
                    self.pipeline_skeleton_samples[peer_hash] = (
                        self.pipeline_skeleton_samples.get(peer_hash, 0) + 1
                    )

        # --- 3. Propagate Model Reward ---
        pipeline_models = self.pipeline_to_models_map.get(pipeline_hash, [])

        for model_name in pipeline_models:
            model_peers = self.model_map.get(model_name, [])
            for peer_hash in model_peers:
                # Don't update itself
                if peer_hash != pipeline_hash:  # MODIFIED: Only check against self
                    self.pipeline_model_rewards[peer_hash] = (
                        self.pipeline_model_rewards.get(peer_hash, 0.0) + reward
                    )
                    self.pipeline_model_samples[peer_hash] = (
                        self.pipeline_model_samples.get(peer_hash, 0) + 1
                    )

    def sample_pipeline(
        self, skeleton_to_pipelines: Mapping
    ) -> Tuple[Any, List[Dict[str, Any]], float]:
        """
        Sample a pipeline using the UCB algorithm, prioritizing pipelines
        that have not been directly executed yet.

        Args:
            skeleton_to_pipelines: Mapping of skeleton objects to lists of (pipeline, estimated_cost) tuples.
                                   Used to retrieve original objects and costs.

        Returns:
            Tuple containing:
              - The selected skeleton object
              - The selected pipeline configuration
              - Estimated cost of the selected pipeline

        Raises:
            ValueError: If all pipelines have been directly sampled or no pipelines are available.
        """
        # Build necessary lookups from the input mapping
        pipeline_hash_to_objects: Dict[str, Tuple[Any, List[Dict[str, Any]], float]] = (
            {}
        )
        all_pipeline_hashes: List[str] = []

        for skeleton, pipelines in skeleton_to_pipelines.items():
            for pipeline, estimated_cost in pipelines:
                pipeline_hash = self._hash_pipeline(pipeline)
                if pipeline_hash not in pipeline_hash_to_objects:  # Ensure uniqueness
                    pipeline_hash_to_objects[pipeline_hash] = (
                        skeleton,
                        pipeline,
                        estimated_cost,
                    )
                    all_pipeline_hashes.append(pipeline_hash)

        # Calculate UCB scores only for pipelines not yet directly sampled
        pipeline_scores: Dict[str, float] = {}
        for pipeline_hash in all_pipeline_hashes:
            if pipeline_hash not in self.sampled_pipelines:
                score = self.calculate_ucb_score(pipeline_hash)
                pipeline_scores[pipeline_hash] = score

        # Select pipeline with highest UCB score
        if not pipeline_scores:
            raise ValueError("All pipelines have already been sampled directly.")

        best_pipeline_hash = max(pipeline_scores, key=pipeline_scores.get)

        # Retrieve the original objects using the hash
        selected_skeleton, selected_pipeline, selected_cost = pipeline_hash_to_objects[
            best_pipeline_hash
        ]

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
        # Ensure deep copy of input docs for isolated execution
        current_input_docs = copy.deepcopy(input_docs)
        return execute_pipeline(
            current_input_docs, pipeline, self.run_operation_func, self.console
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
    direct_reward_weight: float = 1.0,
    skeleton_reward_weight: float = 0.5,
    model_reward_weight: float = 0.3,
) -> Tuple[
    List[
        Tuple[List[Dict[str, Any]], float, float, List[Dict[str, Any]], Dict[str, Any]]
    ],
    float,
]:
    """
    Samples execution of pipelines using a UCB algorithm that considers direct,
    skeleton-based, and model-based rewards. Each pipeline is directly sampled
    at most once.

    Args:
        sample_docs: A list of sample documents to use as input.
        console: Console for logging.
        skeleton_to_pipelines: Mapping of skeleton objects to lists of (pipeline, estimated_cost) tuples.
        run_operation_func: Reference to the runner's _run_operation function.
        llm_client: LLM client for querying and ranking.
        original_pipeline_config: The original pipeline configuration.
        scoring_func: Function to score pipeline outputs (higher is better).
        sample_size: Number of documents to use in sampling (default: 5).
        budget_num_pipelines: Maximum number of pipelines to sample directly (default: 20).
        exploration_weight: Weight for exploration in UCB formula.
        direct_reward_weight: Weight for direct reward component in UCB score.
        skeleton_reward_weight: Weight for skeleton reward component in UCB score.
        model_reward_weight: Weight for model reward component in UCB score.

    Returns:
        Tuple containing:
          - List of tuples, each containing (pipeline configuration, estimated_cost, actual_cost, output_docs, ranking_info)
          - Total sampling cost
    """
    # Initialize UCB sampler without cost weight
    ucb = UCBSampler(
        run_operation_func=run_operation_func,
        console=console,
        exploration_weight=exploration_weight,
        direct_reward_weight=direct_reward_weight,
        skeleton_reward_weight=skeleton_reward_weight,
        model_reward_weight=model_reward_weight,
    )

    # Initialize the sampler with the pipeline structures
    ucb.initialize_pipelines(skeleton_to_pipelines)

    results_data: List[
        Tuple[List[Dict[str, Any]], float, float, List[Dict[str, Any]]]
    ] = []  # Store (pipeline, estimated_cost, actual_cost, output_docs)

    # Use a limited number of docs for sampling
    input_docs = copy.deepcopy(sample_docs[:sample_size])
    console.log(f"Sampling with {len(input_docs)} documents using UCB algorithm.")
    console.log("Each pipeline will be directly sampled at most once.")
    sampling_cost = 0.0

    # Calculate the total number of available pipelines
    total_pipelines = len(ucb.pipeline_direct_rewards)  # Get count after initialization
    actual_budget = min(budget_num_pipelines, total_pipelines)

    console.log(f"Total available pipelines: {total_pipelines}")
    console.log(f"Sampling budget (max direct executions): {actual_budget}")

    # --- Single Sampling Loop using UCB ---
    console.log("\n--- UCB Sampling Phase ---")
    for iteration in range(actual_budget):
        try:
            # Sample a pipeline using UCB
            # Pass skeleton_to_pipelines again to retrieve original objects/costs
            skeleton, pipeline, estimated_cost = ucb.sample_pipeline(
                skeleton_to_pipelines
            )
            pipeline_hash = ucb._hash_pipeline(pipeline)  # Get hash for logging

            # Log the selected pipeline
            console.log(f"\nUCB Sampling iteration {iteration+1}/{actual_budget}")
            ucb_score = ucb.calculate_ucb_score(
                pipeline_hash
            )  # Recalculate for logging (before it becomes -inf)
            console.log(
                f"Selected pipeline hash: {pipeline_hash[:12]} (Score: {ucb_score:.4f})"
            )
            console.log(f"  Skeleton: {str(skeleton)[:60]}...")
            console.log(
                f"  Operations: {len(pipeline)}, Estimated cost: ${estimated_cost:.6f}"
            )

            # Execute the pipeline on sample docs
            actual_cost, output_docs = ucb.execute_pipeline(input_docs, pipeline)
            sampling_cost += actual_cost

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
            reward = 0.0
            if output_docs:
                try:
                    # Calculate reward as average of scoring function across documents
                    doc_rewards = [scoring_func(doc) for doc in output_docs]
                    if doc_rewards:  # Avoid division by zero if no valid scores
                        reward = sum(doc_rewards) / len(doc_rewards)
                    console.log(f"  Pipeline reward: {reward:.4f}")
                except Exception as e:
                    console.log(f"[bold red]  Error calculating reward: {e}[/bold red]")
            else:
                console.log("  No output documents generated, reward is 0.")

            # Update UCB sampler with reward
            ucb.update_reward(pipeline, reward)

            # Store results for this pipeline if there are output docs
            if len(output_docs) > 0:
                results_data.append(
                    (pipeline, estimated_cost, actual_cost, output_docs)
                )

            # Log remaining pipelines to sample directly
            remaining_to_sample = total_pipelines - len(ucb.sampled_pipelines)
            console.log(
                f"  Remaining pipelines to sample directly: {remaining_to_sample}"
            )

        except ValueError as e:
            # This happens when sample_pipeline raises error (all pipelines sampled)
            console.log(f"\n[bold yellow]Stopping UCB sampling: {str(e)}[/bold yellow]")
            break
        except Exception as e:
            console.log(
                f"\n[bold red]Error during UCB sampling iteration {iteration+1}: {e}[/bold red]"
            )
            # Optionally continue to the next iteration or break
            continue  # Continue for now

    # Rank the results using LLM as a judge and scoring function
    console.log(
        f"\nRanking {len(results_data)} pipeline results based on output quality..."
    )
    if not results_data:
        console.log("[yellow]No successful pipeline executions to rank.[/yellow]")
        return [], sampling_cost

    # Pass results_data which now includes estimated_cost
    # Assume rank_pipeline_outputs_with_llm preserves all necessary info, including estimated_cost
    ranked_results = rank_pipeline_outputs_with_llm(
        results_data,  # Input: List[Tuple[pipeline, estimated_cost, actual_cost, output_docs]]
        original_pipeline_config,
        llm_client,
        console,
        scoring_func,
        run_operation_func,
    )

    return ranked_results, sampling_cost
