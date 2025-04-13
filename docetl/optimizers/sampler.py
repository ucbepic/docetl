"""
This module provides sampling utilities for pipeline execution estimation.
It enables cost estimation and validation of operation chains without
full dataset processing.
"""

import copy
import random
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Tuple

from rich.console import Console

from docetl.optimizers.pipeline_utils import (
    compare_sampling_strategies,
    execute_pipeline,
    rank_pipeline_outputs_with_llm,
)
from docetl.optimizers.thompson_sampler import sample_pipeline_execution_with_thompson
from docetl.optimizers.ucb_sampler import sample_pipeline_execution_with_ucb
from docetl.optimizers.utils import LLMClient


def sample_pipeline_execution(
    sample_docs: List[Dict[str, Any]],
    console: Console,
    skeleton_to_pipelines: Mapping,
    run_operation_func: Callable,
    llm_client: LLMClient,
    original_pipeline_config: Dict[str, Any],
    scoring_func: Callable,
    sample_size: int = 5,
    log_dir: Optional[str] = None,
    budget_num_pipelines: int = 40,
    sampling_strategy: Literal["random", "ucb", "thompson", "all"] = "all",
    ucb_exploration_weight: float = 2.0,
    ucb_dampening_factor: float = 0.8,
) -> Dict[
    str,
    Tuple[
        List[Tuple[List[Dict[str, Any]], float, List[Dict[str, Any]], Dict[str, Any]]],
        float,
    ],
]:
    """
    Samples execution of pipelines from a skeleton-to-pipeline mapping,
    running each pipeline on a subset of documents to estimate costs.

    Args:
        sample_docs: A list of sample documents to use as input
        console: Console for logging
        skeleton_to_pipelines: Mapping of skeleton objects to lists of (pipeline, estimated_cost) tuples
        run_operation_func: Reference to the runner's _run_operation function
        llm_client: LLM client for querying and ranking
        original_pipeline_config: The original pipeline configuration
        scoring_func: Function to score pipeline outputs (higher is better)
        sample_size: Number of documents to use in sampling (default: 5)
        log_dir: Directory to save visualization results (default: None)
        budget_num_pipelines: Maximum number of pipelines to sample (default: 40)
        sampling_strategy: Strategy to use for sampling: "random", "ucb", "thompson", or "all" (default: "all")
        ucb_exploration_weight: Weight for exploration in UCB formula (default: 2.0)
        ucb_dampening_factor: Factor to dampen rewards when propagating (default: 0.8)

    Returns:
        Dictionary mapping strategy names to tuples containing:
          - List of tuples, each containing (pipeline configuration, actual cost, output_docs, ranking_info)
          - Total sampling cost
    """
    results = {}

    # Run random sampling if requested or both
    if sampling_strategy in ["random", "all"]:
        console.log("\n=== RUNNING RANDOM SAMPLING STRATEGY ===\n")
        random_results, random_cost = _run_random_sampling(
            sample_docs=sample_docs,
            console=console,
            skeleton_to_pipelines=skeleton_to_pipelines,
            run_operation_func=run_operation_func,
            llm_client=llm_client,
            original_pipeline_config=original_pipeline_config,
            scoring_func=scoring_func,
            sample_size=sample_size,
            budget_num_pipelines=budget_num_pipelines,
        )
        results["random"] = (random_results, random_cost)

    # Run Thompson sampling if requested or both
    if sampling_strategy in ["thompson", "all"]:
        console.log("\n=== RUNNING THOMPSON SAMPLING STRATEGY ===\n")
        thompson_results, thompson_cost = sample_pipeline_execution_with_thompson(
            sample_docs=sample_docs,
            console=console,
            skeleton_to_pipelines=skeleton_to_pipelines,
            run_operation_func=run_operation_func,
            llm_client=llm_client,
            original_pipeline_config=original_pipeline_config,
            scoring_func=scoring_func,
            sample_size=sample_size,
            budget_num_pipelines=budget_num_pipelines,
        )
        results["thompson"] = (thompson_results, thompson_cost)

    # Run UCB sampling if requested or both
    if sampling_strategy in ["ucb", "all"]:
        console.log("\n=== RUNNING UCB SAMPLING STRATEGY ===\n")
        try:
            ucb_results, ucb_cost = sample_pipeline_execution_with_ucb(
                sample_docs=sample_docs,
                console=console,
                skeleton_to_pipelines=skeleton_to_pipelines,
                run_operation_func=run_operation_func,
                llm_client=llm_client,
                original_pipeline_config=original_pipeline_config,
                scoring_func=scoring_func,
                sample_size=sample_size,
                budget_num_pipelines=budget_num_pipelines,
                exploration_weight=ucb_exploration_weight,
                dampening_factor=ucb_dampening_factor,
            )
            results["ucb"] = (ucb_results, ucb_cost)
        except ImportError:
            console.log("UCB sampling requested but module not available")

    # If both strategies were run, print a comparison of the top pipelines
    if len(sampling_strategy) > 1 or sampling_strategy == "all":
        compare_sampling_strategies(results, console, budget_num_pipelines, log_dir)

    return results


def _run_random_sampling(
    sample_docs: List[Dict[str, Any]],
    console: Console,
    skeleton_to_pipelines: Mapping,
    run_operation_func: Callable,
    llm_client: LLMClient,
    original_pipeline_config: Dict[str, Any],
    scoring_func: Callable,
    sample_size: int = 5,
    budget_num_pipelines: int = 20,
) -> Tuple[
    List[Tuple[List[Dict[str, Any]], float, List[Dict[str, Any]], Dict[str, Any]]],
    float,
]:
    """
    Runs the random sampling strategy.

    Args:
        Same as sample_pipeline_execution

    Returns:
        Tuple containing:
          - List of tuples, each containing (pipeline configuration, actual cost, output_docs, ranking_info)
          - Total sampling cost
    """
    console.log("Using random sampling strategy")
    results = []

    # Use a limited number of docs for sampling
    input_docs = copy.deepcopy(sample_docs[:sample_size])
    console.log(f"Sampling with {len(input_docs)} documents")
    sampling_cost = 0

    num_pipelines = sum(len(pipelines) for pipelines in skeleton_to_pipelines.values())
    skeleton_idx = 0

    # For each skeleton, sample just one pipeline
    for skeleton, pipelines in skeleton_to_pipelines.items():
        skeleton_idx += 1
        # Figure out the budget for this skeleton, proportional to the number of pipelines
        budget = int(budget_num_pipelines * len(pipelines) / num_pipelines)

        num_to_sample = min(budget, len(pipelines))

        # Take the first pipeline from each skeleton
        if pipelines:
            # Sample indices without replacement
            sampled_indices = random.sample(range(len(pipelines)), num_to_sample)

            for i in sampled_indices:
                pipeline, estimated_cost = pipelines[i]
                console.log(
                    f"Sampling pipeline for skeleton: {skeleton_idx}... Operations: {len(pipeline)}, Estimated cost: ${estimated_cost:.6f}"
                )

                # Execute the pipeline on sample docs
                actual_cost, output_docs = execute_pipeline(
                    input_docs, pipeline, run_operation_func, console
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

                # Store results for this skeleton if there are output docs
                if len(output_docs) > 0:
                    results.append((pipeline, actual_cost, output_docs))
                sampling_cost += actual_cost
        else:
            # Handle empty pipelines case
            console.log(f"No pipelines found for skeleton: {str(skeleton)[:50]}...")

    # Rank the results using LLM as a judge and scoring function
    console.log(f"Ranking {len(results)} pipeline results based on output quality...")
    ranked_results = rank_pipeline_outputs_with_llm(
        results, original_pipeline_config, llm_client, console, scoring_func
    )

    return ranked_results, sampling_cost
