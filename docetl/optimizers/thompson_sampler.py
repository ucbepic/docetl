"""
This module provides Thompson Sampling utilities for intelligent
exploration and exploitation in pipeline optimization.
"""

import copy
import math
import random
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
from rich.console import Console

from docetl.optimizers.pipeline_utils import (
    execute_pipeline,
    rank_pipeline_outputs_with_llm,
)
from docetl.optimizers.utils import LLMClient


class ThompsonSampler:
    """
    Implements Thompson Sampling algorithm for pipeline sampling.

    This class maintains state about pipeline rewards and samples to enable
    intelligent exploration and exploitation when evaluating pipeline alternatives.
    Uses Gaussian priors/posteriors for reward modeling.
    """

    def __init__(
        self,
        run_operation_func: Callable,
        console: Console,
        prior_variance: float = 1.0,
        likelihood_variance: float = 0.5,
        dampening_factor: float = 0.8,
    ):
        """
        Initialize the Thompson sampler.

        Args:
            run_operation_func: Function to execute operations in the pipeline
            console: Console for logging
            prior_variance: Initial variance for reward priors (higher = more exploration)
            likelihood_variance: Assumed variance of the reward likelihood function
            dampening_factor: Factor to dampen rewards when propagating to related pipelines
        """
        # Pipeline state tracking
        self.pipeline_rewards = {}  # Maps pipeline hash to list of observed rewards
        self.pipeline_samples = {}  # Maps pipeline hash to number of samples
        self.skeleton_to_pipelines = {}  # Maps skeleton hash to list of pipeline hashes
        self.pipeline_to_skeleton = {}  # Maps pipeline hash to its skeleton hash
        self.sampled_pipelines = (
            set()
        )  # Set of pipeline hashes that have already been sampled
        self.sampled_skeletons = (
            set()
        )  # Set of skeleton hashes that have already been sampled

        # Bayesian model parameters
        self.prior_means = {}  # Maps pipeline hash to prior mean
        self.prior_variance = prior_variance
        self.likelihood_variance = likelihood_variance
        self.posterior_means = {}  # Maps pipeline hash to posterior mean
        self.posterior_variances = {}  # Maps pipeline hash to posterior variance

        # Configuration
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
        If scoring_func and sample_docs are provided, estimates initial priors.

        Args:
            skeleton_to_pipelines: Mapping of skeleton objects to lists of (pipeline, estimated_cost) tuples
            scoring_func: Optional function to score pipeline outputs (higher is better)
            sample_docs: Optional sample documents to use for initial prior estimation
        """
        # Reset state
        self.pipeline_rewards = {}
        self.pipeline_samples = {}
        self.skeleton_to_pipelines = {}
        self.pipeline_to_skeleton = {}
        self.sampled_pipelines = set()
        self.sampled_skeletons = set()
        self.prior_means = {}
        self.posterior_means = {}
        self.posterior_variances = {}
        self.total_samples = 0

        # Process each skeleton and its pipelines
        for skeleton, pipelines in skeleton_to_pipelines.items():
            skeleton_hash = self._hash_skeleton(skeleton)
            self.skeleton_to_pipelines[skeleton_hash] = []

            for pipeline, _ in pipelines:
                pipeline_hash = self._hash_pipeline(pipeline)
                self.pipeline_rewards[pipeline_hash] = []
                self.pipeline_samples[pipeline_hash] = 0
                self.skeleton_to_pipelines[skeleton_hash].append(pipeline_hash)
                self.pipeline_to_skeleton[pipeline_hash] = skeleton_hash

        # Estimate initial priors if scoring function and sample docs are provided
        if scoring_func and sample_docs and len(sample_docs) > 0:
            self._estimate_initial_priors(
                skeleton_to_pipelines, scoring_func, sample_docs
            )
        else:
            # Set default priors
            for pipeline_hash in self.pipeline_samples.keys():
                self.prior_means[pipeline_hash] = 0.5  # Default prior mean
                self.posterior_means[pipeline_hash] = 0.5
                self.posterior_variances[pipeline_hash] = self.prior_variance

    def _estimate_initial_priors(
        self,
        skeleton_to_pipelines: Mapping,
        scoring_func: Callable,
        sample_docs: List[Dict[str, Any]],
    ) -> None:
        """
        Estimate initial prior means for pipelines using static analysis and/or lightweight runs.

        Args:
            skeleton_to_pipelines: Mapping of skeleton objects to lists of (pipeline, estimated_cost) tuples
            scoring_func: Function to score pipeline outputs (higher is better)
            sample_docs: Sample documents to use for estimation
        """
        self.console.log("Estimating initial priors for Thompson Sampling...")

        # Use a single sample document to reduce overhead
        estimation_doc = sample_docs[0] if sample_docs else None
        if not estimation_doc:
            self.console.log(
                "No sample documents available for initial prior estimation"
            )
            return

        # For each pipeline, create a prior mean estimate
        for skeleton, pipelines in skeleton_to_pipelines.items():
            for pipeline, estimated_cost in pipelines:
                pipeline_hash = self._hash_pipeline(pipeline)

                # Use pipeline characteristics to estimate quality
                prior_mean = 0.0

                # Factor 1: Operation count - more operations might be more complete
                # but could also introduce more error, so use a curve that peaks
                op_count = len(pipeline)
                if op_count > 0:
                    # Operations increase prior mean up to a point, then decrease
                    op_factor = min(op_count / 5.0, 2.0 - (op_count / 10.0))
                    op_factor = max(
                        0.1, min(1.0, op_factor)
                    )  # Clamp between 0.1 and 1.0
                    prior_mean += op_factor * 0.3  # 30% weight to operation count

                # Factor 2: Estimated cost - higher cost might correlate with quality
                # but with diminishing returns
                if estimated_cost > 0:
                    # Log scale for cost to account for diminishing returns
                    cost_factor = min(1.0, 0.1 + 0.3 * math.log(1 + estimated_cost))
                    prior_mean += cost_factor * 0.2  # 20% weight to cost

                # Add small jitter to prevent identical priors
                jitter = random.uniform(-0.05, 0.05)
                prior_mean += jitter + 0.4  # Baseline of 0.4 + factors + jitter

                # Ensure prior is within reasonable bounds
                prior_mean = max(0.1, min(0.9, prior_mean))

                # Store the estimated prior mean
                self.prior_means[pipeline_hash] = prior_mean
                self.posterior_means[pipeline_hash] = prior_mean
                self.posterior_variances[pipeline_hash] = self.prior_variance

                # Log the estimate
                if (
                    len(self.prior_means) <= 5 or random.random() < 0.05
                ):  # Log only some to avoid spam
                    self.console.log(
                        f"Pipeline {pipeline_hash[:8]} prior mean estimate: {prior_mean:.4f}"
                    )

        self.console.log(
            f"Estimated initial priors for {len(self.prior_means)} pipelines"
        )

    def sample_from_posterior(self, pipeline_hash: str) -> float:
        """
        Sample a value from the posterior distribution for a pipeline.

        Args:
            pipeline_hash: Hash of the pipeline

        Returns:
            A random sample from the posterior distribution
        """
        # If already sampled, return negative infinity to exclude
        if pipeline_hash in self.sampled_pipelines:
            return float("-inf")

        mean = self.posterior_means.get(pipeline_hash, 0.5)
        var = self.posterior_variances.get(pipeline_hash, self.prior_variance)

        # Sample from Gaussian posterior
        return np.random.normal(mean, np.sqrt(var))

    def update_posterior(self, pipeline_hash: str, new_reward: float) -> None:
        """
        Update the posterior distribution for a pipeline given a new reward observation.

        Args:
            pipeline_hash: Hash of the pipeline
            new_reward: New observed reward
        """
        # Get current posterior parameters
        current_mean = self.posterior_means.get(
            pipeline_hash, self.prior_means.get(pipeline_hash, 0.5)
        )
        current_var = self.posterior_variances.get(pipeline_hash, self.prior_variance)

        # Bayesian update for Gaussian - weighted average based on precisions
        precision_prior = 1.0 / current_var
        precision_likelihood = 1.0 / self.likelihood_variance
        precision_posterior = precision_prior + precision_likelihood

        # New posterior variance
        posterior_var = 1.0 / precision_posterior

        # New posterior mean
        posterior_mean = (
            precision_prior * current_mean + precision_likelihood * new_reward
        ) / precision_posterior

        # Store updated posterior
        self.posterior_means[pipeline_hash] = posterior_mean
        self.posterior_variances[pipeline_hash] = posterior_var

    def update_reward(self, pipeline: List[Dict[str, Any]], reward: float) -> None:
        """
        Update rewards for a pipeline and propagate to related pipelines.
        Also updates posterior distributions.

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

        # Update pipeline's own rewards and sample count
        self.pipeline_rewards[pipeline_hash].append(reward)
        self.pipeline_samples[pipeline_hash] = (
            self.pipeline_samples.get(pipeline_hash, 0) + 1
        )
        self.total_samples += 1

        # Update posterior distribution with new reward
        self.update_posterior(pipeline_hash, reward)

        # Propagate dampened reward to other pipelines in the same skeleton
        if skeleton_hash:
            related_pipelines = self.skeleton_to_pipelines.get(skeleton_hash, [])
            for related_hash in related_pipelines:
                if related_hash != pipeline_hash:
                    # Add dampened soft "virtual" observations to related pipelines
                    # by updating their posteriors with less certainty
                    dampened_reward = reward * self.dampening_factor

                    # Use higher likelihood variance for propagated rewards
                    # to reflect lower confidence
                    orig_variance = self.likelihood_variance
                    self.likelihood_variance = orig_variance * 2.0  # Lower certainty

                    # Update posterior with dampened reward
                    self.update_posterior(related_hash, dampened_reward)

                    # Restore original likelihood variance
                    self.likelihood_variance = orig_variance

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

        # Random selection for Phase 1 (could use Thompson within skeleton too)
        selected_pipeline, _, selected_cost = random.choice(unsampled_pipelines)

        return selected_skeleton, selected_pipeline, selected_cost

    def sample_pipeline(
        self, skeleton_to_pipelines: Mapping
    ) -> Tuple[Any, List[Dict[str, Any]], float]:
        """
        Sample a pipeline using Thompson Sampling algorithm, never sampling the same pipeline twice.

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

        # Sample from posterior for all pipelines
        pipeline_samples = {}
        pipeline_objects = {}
        pipeline_costs = {}

        for skeleton, pipelines in skeleton_to_pipelines.items():
            for pipeline, estimated_cost in pipelines:
                pipeline_hash = self._hash_pipeline(pipeline)

                # Skip already sampled pipelines
                if pipeline_hash in self.sampled_pipelines:
                    continue

                # Draw a sample from the posterior distribution
                sampled_value = self.sample_from_posterior(pipeline_hash)
                pipeline_samples[pipeline_hash] = sampled_value
                pipeline_objects[pipeline_hash] = pipeline
                pipeline_costs[pipeline_hash] = estimated_cost

        # Select pipeline with highest sampled value from posterior
        if not pipeline_samples:
            raise ValueError("No unsampled pipelines available")

        best_pipeline_hash = max(pipeline_samples, key=pipeline_samples.get)
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


def sample_pipeline_execution_with_thompson(
    sample_docs: List[Dict[str, Any]],
    console: Console,
    skeleton_to_pipelines: Mapping,
    run_operation_func: Callable,
    llm_client: LLMClient,
    original_pipeline_config: Dict[str, Any],
    scoring_func: Callable,
    sample_size: int = 5,
    budget_num_pipelines: int = 20,
    prior_variance: float = 1.0,
    likelihood_variance: float = 0.5,
    dampening_factor: float = 0.8,
) -> Tuple[
    List[Tuple[List[Dict[str, Any]], float, List[Dict[str, Any]], Dict[str, Any]]],
    float,
]:
    """
    Samples execution of pipelines using Thompson Sampling algorithm to intelligently explore and exploit
    the space of possible pipelines. Each pipeline is sampled at most once, with at least one
    pipeline from each skeleton sampled first before Thompson selection begins.

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
        prior_variance: Initial variance for reward priors (higher = more exploration)
        likelihood_variance: Assumed variance of the reward likelihood function
        dampening_factor: Factor to dampen rewards when propagating to related pipelines

    Returns:
        Tuple containing:
          - List of tuples, each containing (pipeline configuration, actual cost, output_docs, ranking_info)
          - Total sampling cost
    """
    # Initialize Thompson sampler
    thompson = ThompsonSampler(
        run_operation_func=run_operation_func,
        console=console,
        prior_variance=prior_variance,
        likelihood_variance=likelihood_variance,
        dampening_factor=dampening_factor,
    )

    # Pass the scoring function and sample docs for initial prior estimation
    thompson.initialize_pipelines(
        skeleton_to_pipelines,
        scoring_func=scoring_func,
        sample_docs=sample_docs[:1],  # Use just one document for estimation
    )

    results = []

    # Use a limited number of docs for sampling
    input_docs = copy.deepcopy(sample_docs[:sample_size])
    console.log(
        f"Sampling with {len(input_docs)} documents using Thompson Sampling algorithm"
    )
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
                thompson.sample_pipeline_from_unsampled_skeleton(skeleton_to_pipelines)
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
            actual_cost, output_docs = thompson.execute_pipeline(input_docs, pipeline)

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

            # Update Thompson sampler with reward
            thompson.update_reward(pipeline, reward)

            # Store results for this pipeline if there are output docs
            if len(output_docs) > 0:
                results.append((pipeline, actual_cost, output_docs))
            sampling_cost += actual_cost
            sampled_in_phase1 += 1

            # Log remaining unsampled skeletons
            unsampled_skeleton_count = thompson.get_unsampled_skeleton_count(
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

    # PHASE 2: Use Thompson Sampling algorithm to sample remaining budget
    remaining_budget = actual_budget - sampled_in_phase1

    if remaining_budget > 0:
        console.log(
            f"\n--- PHASE 2: Thompson Sampling for remaining budget ({remaining_budget}) ---"
        )

        for iteration in range(remaining_budget):
            try:
                # Sample a pipeline using Thompson Sampling
                skeleton, pipeline, estimated_cost = thompson.sample_pipeline(
                    skeleton_to_pipelines
                )

                # Log the selected pipeline
                console.log(
                    f"PHASE 2 - Thompson Sampling iteration {iteration+1}/{remaining_budget}"
                )
                console.log(f"Selected pipeline for skeleton: {str(skeleton)[:50]}...")
                console.log(
                    f"Operations: {len(pipeline)}, Estimated cost: ${estimated_cost:.6f}"
                )

                # Execute the pipeline on sample docs
                actual_cost, output_docs = thompson.execute_pipeline(
                    input_docs, pipeline
                )

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

                        # Log posterior parameters for this pipeline
                        pipeline_hash = thompson._hash_pipeline(pipeline)
                        posterior_mean = thompson.posterior_means.get(pipeline_hash, 0)
                        posterior_var = thompson.posterior_variances.get(
                            pipeline_hash, 0
                        )
                        console.log(
                            f"  Updated posterior: mean={posterior_mean:.4f}, var={posterior_var:.4f}"
                        )

                    except Exception as e:
                        console.log(f"  Error calculating reward: {e}")

                # Update Thompson sampler with reward
                thompson.update_reward(pipeline, reward)

                # Store results for this pipeline if there are output docs
                if len(output_docs) > 0:
                    results.append((pipeline, actual_cost, output_docs))
                sampling_cost += actual_cost

                # Log remaining unsampled pipelines
                unsampled_count = thompson.get_unsampled_pipeline_count(
                    skeleton_to_pipelines
                )
                console.log(f"  Remaining unsampled pipelines: {unsampled_count}")

            except ValueError as e:
                console.log(f"Stopping Thompson Sampling: {str(e)}")
                break
            except Exception as e:
                console.log(f"Error in Thompson Sampling iteration: {e}")
                continue
    else:
        console.log("No remaining budget for Phase 2 Thompson Sampling")

    # Rank the results using LLM as a judge and scoring function
    console.log(f"Ranking {len(results)} pipeline results based on output quality...")
    ranked_results = rank_pipeline_outputs_with_llm(
        results, original_pipeline_config, llm_client, console, scoring_func
    )

    return ranked_results, sampling_cost
