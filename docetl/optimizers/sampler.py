"""
This module provides sampling utilities for pipeline execution estimation.
It enables cost estimation and validation of operation chains without
full dataset processing.
"""

import copy
import random
import traceback
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Tuple

from rich.console import Console

from docetl.optimizers.pipeline_utils import (
    compare_sampling_strategies,
    execute_pipeline,
    rank_pipeline_outputs_with_llm,
)
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
    budget_num_pipelines: int = 50,
    sampling_strategy: Literal["random", "ucb", "all"] = ["all"],
) -> Dict[
    str,
    Tuple[
        List[
            Tuple[
                List[Dict[str, Any]], float, float, List[Dict[str, Any]], Dict[str, Any]
            ]
        ],
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
        budget_num_pipelines: Maximum number of pipelines to sample (default: 10)
        sampling_strategy: Strategy to use for sampling: "random", "ucb", or "all" (default: "ucb")

    Returns:
        Dictionary mapping strategy names to tuples containing:
          - List of tuples, each containing (pipeline configuration, estimated_cost, actual_cost, output_docs, ranking_info)
          - Total sampling cost
    """
    results = {}
    strategies_to_run = set()
    if isinstance(sampling_strategy, str):
        strategies_to_run.add(sampling_strategy)
    elif isinstance(sampling_strategy, list):
        strategies_to_run.update(sampling_strategy)

    # Run random sampling if requested or 'all'
    if "random" in strategies_to_run or "all" in strategies_to_run:
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

    # Run UCB sampling if requested or 'all'
    if "ucb" in strategies_to_run or "all" in strategies_to_run:
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
            )
            results["ucb"] = (ucb_results, ucb_cost)
        except ImportError:
            console.log(
                "[bold yellow]UCB sampling requested but module seems unavailable or has issues.[/bold yellow]"
            )
        except TypeError as e:
            console.log(f"[bold red]TypeError calling UCB sampler: {e}[/bold red]")
            console.log(
                "[yellow]Check if function arguments match the definition in ucb_sampler.py[/yellow]"
            )
            traceback.print_exc()
        except Exception as e:
            console.log(f"[bold red]Error during UCB sampling: {e}[/bold red]")
            traceback.print_exc()

    # If multiple strategies were run ('all' or specific list > 1), compare results
    if len(results) > 1:
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
    List[
        Tuple[List[Dict[str, Any]], float, float, List[Dict[str, Any]], Dict[str, Any]]
    ],
    float,
]:
    """
    Runs the random sampling strategy by selecting pipelines randomly from the entire pool.

    Args:
        sample_docs: A list of sample documents to use as input
        console: Console for logging
        skeleton_to_pipelines: Mapping of skeleton objects to lists of (pipeline, estimated_cost) tuples
        run_operation_func: Reference to the runner's _run_operation function
        llm_client: LLM client for querying and ranking
        original_pipeline_config: The original pipeline configuration
        scoring_func: Function to score pipeline outputs (higher is better)
        sample_size: Number of documents to use in sampling (default: 5)
        budget_num_pipelines: Maximum number of pipelines to sample randomly (default: 20)

    Returns:
        Tuple containing:
          - List of tuples, each containing (pipeline configuration, estimated_cost, actual_cost, output_docs, ranking_info)
          - Total sampling cost
    """
    console.log("Using random sampling strategy")
    results_data: List[
        Tuple[List[Dict[str, Any]], float, float, List[Dict[str, Any]]]
    ] = []

    # Use a limited number of docs for sampling
    input_docs = copy.deepcopy(sample_docs[:sample_size])
    console.log(f"Sampling with {len(input_docs)} documents")
    sampling_cost = 0.0  # Initialize sampling_cost as float

    # Flatten the list of pipelines from all skeletons
    all_pipelines_with_details = []
    for skeleton, pipelines in skeleton_to_pipelines.items():
        for pipeline, estimated_cost in pipelines:
            # Store skeleton info if needed later, otherwise just pipeline and cost
            all_pipelines_with_details.append((pipeline, estimated_cost, skeleton))

    num_pipelines_total = len(all_pipelines_with_details)
    # Determine the number of pipelines to sample based on budget and availability
    num_to_sample = min(budget_num_pipelines, num_pipelines_total)

    if num_pipelines_total == 0:
        console.log("[yellow]No pipelines found to sample.[/yellow]")
        return [], 0.0

    console.log(
        f"Randomly selecting {num_to_sample} out of {num_pipelines_total} total pipelines."
    )

    # Sample indices without replacement from the flattened list
    sampled_indices = random.sample(range(num_pipelines_total), num_to_sample)

    # Execute the selected pipelines
    for i, idx in enumerate(sampled_indices):
        pipeline, estimated_cost, skeleton = all_pipelines_with_details[idx]
        # Use a consistent hash for logging/identification if needed
        pipeline_hash = str(
            hash(str(sorted(pipeline, key=lambda op: op.get("name", ""))))
        )

        console.log(
            f"\nSampling pipeline {i+1}/{num_to_sample} (Hash: {pipeline_hash[:8]}, Est. Cost: ${estimated_cost:.6f})"
        )
        console.log(f"  Operations: {len(pipeline)}")

        try:
            # Execute the pipeline on a fresh copy of sample docs
            actual_cost, output_docs = execute_pipeline(
                copy.deepcopy(input_docs),  # Ensure fresh copy for each run
                pipeline,
                run_operation_func,
                console,
            )
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

            # Store results for this pipeline only if it produced output documents
            if output_docs:  # Check if list is not empty and not None
                results_data.append(
                    (pipeline, estimated_cost, actual_cost, output_docs)
                )
            else:
                console.log("  No output documents generated.")

        except Exception as e:
            console.log(
                f"[bold red]  Error executing pipeline {pipeline_hash[:8]}: {e}[/bold red]"
            )
            traceback.print_exc()  # Print traceback for execution errors

    # Rank the results using LLM as a judge and scoring function
    console.log(
        f"\nRanking {len(results_data)} successful pipeline results based on output quality..."
    )
    if not results_data:
        console.log("[yellow]No successful pipeline executions to rank.[/yellow]")
        return [], sampling_cost

    # Assume rank_pipeline_outputs_with_llm returns the final 5-item tuple structure directly
    try:
        # Input to ranking: List[Tuple[pipeline, estimated_cost, actual_cost, output_docs]]
        ranked_results = rank_pipeline_outputs_with_llm(
            results_data,
            original_pipeline_config,
            llm_client,
            console,
            scoring_func,
            run_operation_func,
        )
        # Return the results directly from the ranking function
        return ranked_results, sampling_cost
    except Exception as e:
        console.log(f"[bold red]Error during ranking: {e}[/bold red]")
        traceback.print_exc()  # Print traceback for ranking errors
        # Return unranked results with an error indicator if ranking fails
        unranked_results_formatted = [
            (p, est_c, act_c, out_d, {"error": f"Ranking failed: {e}"})
            for p, est_c, act_c, out_d in results_data
        ]
        return unranked_results_formatted, sampling_cost
