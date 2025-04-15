"""
This module provides shared utility functions for pipeline execution and evaluation.
It contains common functionality used by different sampling strategies.
"""

import copy
import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from rich.console import Console
from rich.table import Table

from docetl.optimizers.utils import LLMClient


def execute_pipeline(
    input_docs: List[Dict[str, Any]],
    pipeline: List[Dict[str, Any]],
    run_operation_func: Callable,
    console: Console,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Executes a single pipeline on the provided input documents.

    Args:
        input_docs: Sample documents to process
        pipeline: List of operation configurations to execute
        run_operation_func: Function to execute operations
        console: Console for logging

    Returns:
        Tuple of (total_cost, output_documents)
    """
    # Start with the input documents
    current_docs = copy.deepcopy(input_docs)

    # Track total cost
    total_cost = 0.0

    # Process each operation in sequence
    for op_idx, op_config in enumerate(pipeline):
        op_name = op_config.get("name", f"operation_{op_idx}")
        op_type = op_config.get("type")

        # Skip operations we can't run in sampling mode
        if op_type in ["equijoin"]:
            console.log(
                f"  Skipping {op_type} operation '{op_name}' (not supported in sampling)"
            )
            continue

        console.log(
            f"  Executing {op_type} operation '{op_name}' with {len(current_docs)} docs"
        )

        try:
            # Execute the operation with cost tracking
            start_time = time.time()
            output_docs, cost = run_operation_func(
                op_config,
                current_docs,
                return_instance=False,
                is_build=True,
                return_cost=True,
            )
            execution_time = time.time() - start_time

            # Update tracking metrics
            total_cost += cost
            current_docs = output_docs

            # Log operation results
            console.log(
                f"    Cost: ${cost:.6f}, Time: {execution_time:.2f}s, Output docs: {len(output_docs)}"
            )

        except Exception as e:
            # If operation fails, pipeline can't continue
            console.log(f"    Failed to execute operation '{op_name}': {str(e)}")
            current_docs = []
            break

        # Stop if we have no documents left
        if not current_docs:
            console.log(
                f"    No documents remaining after operation '{op_name}', stopping pipeline"
            )
            break

    console.log(f"  Pipeline execution complete. Total cost: ${total_cost:.6f}")
    return total_cost, current_docs


def rank_pipeline_outputs_with_llm(
    pipeline_results: List[
        Tuple[List[Dict[str, Any]], float, float, List[Dict[str, Any]]]
    ],
    original_pipeline_config: Dict[str, Any],
    llm_client: LLMClient,
    console: Console,
    scoring_func: Callable,
) -> List[
    Tuple[List[Dict[str, Any]], float, float, List[Dict[str, Any]], Dict[str, Any]]
]:
    """
    Ranks multiple pipeline outputs using both LLM as a judge and a scoring function
    to determine which best meets the intended goals of the original pipeline.

    Args:
        pipeline_results: List of tuples containing (pipeline_config, cost, output_docs)
        original_pipeline_config: The original pipeline configuration to compare against
        llm_client: LLM client for judging outputs
        console: Console for logging
        scoring_func: Function to score pipeline outputs (higher is better)

    Returns:
        Ranked list of pipeline results from best to worst (based on LLM judge),
        each with additional ranking information
    """
    # If we have fewer than 2 pipelines, no need to rank
    if len(pipeline_results) < 2:
        return [
            (
                p[0],
                p[1],
                p[2],
                p[3],
                {
                    "pipeline_id": f"P1_{hash(str(p[0])) % 10000}",
                    "llm_rank": 1,
                    "scoring_rank": 1,
                    "llm_score": 0,
                    "scoring_value": 0,
                },
            )
            for p in pipeline_results
        ]

    console.log("Ranking pipeline outputs using LLM judge and scoring function...")

    # Assign unique IDs to each pipeline
    pipeline_ids = []
    for i, (pipeline, _, _, _) in enumerate(pipeline_results):
        # Create a simple hash of the pipeline structure
        pipeline_str = str(pipeline)
        pipeline_hash = hash(pipeline_str) % 10000  # Limit to 4 digits for readability
        pipeline_id = f"P{i+1}_{pipeline_hash}"
        pipeline_ids.append(pipeline_id)

    # Find the minimum length of output docs across all pipelines
    min_docs_length = min(len(pipeline[-1]) for pipeline in pipeline_results)

    if min_docs_length == 0:
        console.log("Warning: Some pipelines produced no output documents")
        # Filter out pipelines with no outputs
        valid_pipelines = [p for p in pipeline_results if len(p[3]) > 0]
        if not valid_pipelines:
            console.log("No valid pipelines with outputs to rank")
            return pipeline_results
        pipeline_results = valid_pipelines
        min_docs_length = min(len(pipeline[2]) for pipeline in pipeline_results)

    # Run scoring function evaluation independently
    scoring_results = evaluate_with_scoring_function(
        pipeline_results, pipeline_ids, scoring_func, min_docs_length, console
    )

    # Run LLM judge evaluation independently
    llm_results = evaluate_with_llm_judge(
        pipeline_results,
        original_pipeline_config,
        pipeline_ids,
        llm_client,
        min_docs_length,
        console,
    )

    # Combine results and sort by LLM judge ranking
    combined_results = combine_ranking_results(
        pipeline_results, llm_results, scoring_results, pipeline_ids, console
    )

    return combined_results


def evaluate_with_scoring_function(
    pipeline_results: List[
        Tuple[List[Dict[str, Any]], float, float, List[Dict[str, Any]]]
    ],
    pipeline_ids: List[str],
    scoring_func: Callable,
    min_docs_length: int,
    console: Console,
) -> Dict[int, Dict[str, Any]]:
    """
    Evaluates pipelines using a scoring function.

    Args:
        pipeline_results: List of tuples containing (pipeline_config, cost, output_docs)
        pipeline_ids: List of unique pipeline identifiers
        scoring_func: Function to score pipeline outputs (higher is better)
        min_docs_length: Minimum number of documents across all pipelines
        console: Console for logging

    Returns:
        Dictionary mapping pipeline index to scoring results
    """
    console.log("Evaluating pipelines using scoring function...")

    # Score pipelines using the scoring function
    scoring_function_scores = []
    for i, (_, _, _, output_docs) in enumerate(pipeline_results):
        try:
            # Calculate score for each document and average them
            doc_scores = []
            for doc_idx in range(min(min_docs_length, len(output_docs))):
                if doc_idx < len(output_docs):
                    score = scoring_func(output_docs[doc_idx])
                    doc_scores.append(score)

            # Average score across documents
            avg_score = sum(doc_scores) / len(doc_scores) if doc_scores else 0
            scoring_function_scores.append((i, avg_score, pipeline_ids[i]))
            console.log(
                f"Pipeline {pipeline_ids[i]} scoring function score: {avg_score:.2f}"
            )
        except Exception as e:
            console.log(f"Error scoring pipeline {pipeline_ids[i]}: {e}")
            scoring_function_scores.append((i, 0.0, pipeline_ids[i]))

    # Sort by scoring function score (higher is better)
    scoring_function_scores.sort(key=lambda x: x[1], reverse=True)

    # Assign ranks based on scoring function
    scoring_rank_map = {
        idx: rank + 1 for rank, (idx, _, _) in enumerate(scoring_function_scores)
    }

    # Create result dictionary with all scoring information
    scoring_results = {}
    for idx, score, _ in scoring_function_scores:
        scoring_results[idx] = {
            "scoring_rank": scoring_rank_map[idx],
            "scoring_value": score,
        }

    return scoring_results


def evaluate_with_llm_judge(
    pipeline_results: List[
        Tuple[List[Dict[str, Any]], float, float, List[Dict[str, Any]]]
    ],
    original_pipeline_config: Dict[str, Any],
    pipeline_ids: List[str],
    llm_client: LLMClient,
    min_docs_length: int,
    console: Console,
) -> Dict[int, Dict[str, Any]]:
    """
    Evaluates pipelines using an LLM as judge.

    Args:
        pipeline_results: List of tuples containing (pipeline_config, cost, output_docs)
        original_pipeline_config: The original pipeline configuration to compare against
        pipeline_ids: List of unique pipeline identifiers
        llm_client: LLM client for judging outputs
        min_docs_length: Minimum number of documents across all pipelines
        console: Console for logging

    Returns:
        Dictionary mapping pipeline index to LLM judge results
    """
    console.log("Evaluating pipelines using LLM as judge...")

    # Convert original pipeline config to string representation
    original_config_str = json.dumps(original_pipeline_config, indent=2)

    # For each document index, compare outputs across all pipelines using LLM
    rankings = []
    for doc_idx in range(min_docs_length):
        console.log(
            f"Comparing document at index {doc_idx} across {len(pipeline_results)} pipelines"
        )

        # Create letters for plan labels (A, B, C, etc.)
        letters = [chr(65 + i) for i in range(len(pipeline_results))]

        # Collect output document at this index from each pipeline
        pipeline_outputs = []
        letter_to_idx_map = {}  # Maps letter labels to original indices

        # Create a list of indices and shuffle it to randomize presentation order
        pipeline_indices = list(range(len(pipeline_results)))
        import random

        random.shuffle(pipeline_indices)

        for order_idx, original_idx in enumerate(pipeline_indices):
            pipeline_config, estimated_cost, actual_cost, output_docs = (
                pipeline_results[original_idx]
            )
            if doc_idx < len(output_docs):
                # Use letter label instead of numerical ID
                letter_label = letters[order_idx]
                letter_to_idx_map[letter_label] = original_idx

                # Add plan with letter label and its output
                pipeline_outputs.append(
                    {
                        "plan_id": letter_label,
                        "pipeline_config": f"Plan {letter_label}: {str(pipeline_config)[:200]}...",
                        "output": output_docs[doc_idx],
                    }
                )

        # Define the system prompt for the judge
        system_prompt = """
You are an expert evaluator of data pipeline outputs. Your task is to rank multiple pipeline implementations
based on how well their outputs align with the original pipeline's intended purpose.

The ranking criteria are:
1. Accuracy and correctness of the output
2. Completeness of information processing
3. Adherence to the original pipeline's goals
4. Quality and usability of the output

You will receive information about the original pipeline configuration and outputs from different implementations.
"""

        # Define message content
        message_content = f"""
ORIGINAL PIPELINE CONFIGURATION:
{original_config_str}

I will show you outputs from {len(pipeline_outputs)} different pipeline implementations for the same document.
Please analyze these outputs and rank them from best to worst based on how well they align with the original pipeline's purpose.

Each plan is labeled with a letter (A, B, C, etc.). The order of presentation is random and does not indicate quality.

OUTPUTS TO EVALUATE:
{json.dumps(pipeline_outputs, indent=2)}
"""

        # Define structured output schema for the response
        parameters = {
            "type": "object",
            "properties": {
                "rankings": {
                    "type": "array",
                    "description": "List of plan_ids (letters) ordered from best to worst",
                    "items": {
                        "type": "string",
                        "description": "Plan letter identifier (A, B, C, etc.)",
                    },
                },
                "explanation": {
                    "type": "string",
                    "description": "A brief explanation of your ranking rationale",
                },
            },
            "required": ["rankings", "explanation"],
        }

        try:
            # Query LLM for ranking using the structured output parameters
            response = llm_client.generate_judge(
                messages=[{"role": "user", "content": message_content}],
                system_prompt=system_prompt,
                parameters=parameters,
            )

            # Extract the structured response
            ranking_data = response.choices[0].message.content

            if isinstance(ranking_data, str):
                # If it's a string for some reason, try to parse it
                try:
                    ranking_data = json.loads(ranking_data)
                except json.JSONDecodeError:
                    console.log("Warning: Failed to parse LLM response as JSON")
                    console.log(f"Response content: {ranking_data}")
                    continue

            if "rankings" in ranking_data:
                # Convert letter rankings back to original indices
                try:
                    plan_rankings = [
                        letter_to_idx_map[letter]
                        for letter in ranking_data["rankings"]
                        if letter in letter_to_idx_map
                    ]

                    # Log the rankings with explanation
                    console.log(f"LLM ranking for document {doc_idx}:")
                    console.log(f"  Rankings (letters): {ranking_data['rankings']}")

                    # Also show the mapping for clarity
                    readable_mapping = [
                        f"{letter}→{pipeline_ids[letter_to_idx_map[letter]]}"
                        for letter in letter_to_idx_map.keys()
                    ]
                    console.log(
                        f"  Letter to plan mapping: {', '.join(readable_mapping)}"
                    )

                    # Show rankings converted to original plan IDs
                    ranked_plan_ids = [pipeline_ids[idx] for idx in plan_rankings]
                    console.log(f"  Rankings (plan IDs): {ranked_plan_ids}")

                    console.log(
                        f"  Explanation: {ranking_data.get('explanation', 'No explanation provided')}"
                    )

                    rankings.append(plan_rankings)
                except KeyError as e:
                    console.log(
                        f"Warning: Letter in rankings not found in mapping: {e}"
                    )
                    console.log(f"  Rankings: {ranking_data['rankings']}")
                    console.log(
                        f"  Available letters: {list(letter_to_idx_map.keys())}"
                    )
            else:
                console.log("Warning: LLM response missing 'rankings' key")
                console.log(f"Response content: {ranking_data}")

        except Exception as e:
            console.log(f"Error querying LLM for ranking: {e}")

    # If we have no rankings, return empty results
    if not rankings:
        console.log("No LLM rankings were produced")
        return {
            i: {"llm_rank": -1, "llm_score": -1} for i in range(len(pipeline_results))
        }

    # Score each pipeline: lower is better (1st place = 0 points, 2nd place = 1 point, etc.)
    llm_scores = [0] * len(pipeline_results)
    for ranking in rankings:
        for position, plan_idx in enumerate(ranking):
            if 0 <= plan_idx < len(llm_scores):
                llm_scores[plan_idx] += position

    console.log(f"LLM scores for {len(llm_scores)} pipelines: {llm_scores}")

    # Create sorted indices for LLM ranking (lower scores are better)
    llm_ranked_indices = sorted(range(len(llm_scores)), key=lambda x: llm_scores[x])

    # Assign LLM ranks
    llm_rank_map = {idx: rank + 1 for rank, idx in enumerate(llm_ranked_indices)}

    # Create result dictionary with all LLM judge information
    llm_results = {}
    for idx in range(len(pipeline_results)):
        llm_results[idx] = {
            "llm_rank": llm_rank_map.get(idx, -1),
            "llm_score": llm_scores[idx],
        }

    return llm_results


def combine_ranking_results(
    pipeline_results: List[
        Tuple[List[Dict[str, Any]], float, float, List[Dict[str, Any]]]
    ],
    llm_results: Dict[int, Dict[str, Any]],
    scoring_results: Dict[int, Dict[str, Any]],
    pipeline_ids: List[str],
    console: Console,
) -> List[
    Tuple[List[Dict[str, Any]], float, float, List[Dict[str, Any]], Dict[str, Any]]
]:
    """
    Combines LLM judge and scoring function results and sorts by LLM ranking.

    Args:
        pipeline_results: List of tuples containing (pipeline_config, cost, output_docs)
        llm_results: Dictionary mapping pipeline index to LLM judge results
        scoring_results: Dictionary mapping pipeline index to scoring function results
        pipeline_ids: List of unique pipeline identifiers
        console: Console for logging

    Returns:
        Sorted list of pipeline results with ranking information
    """
    # Check if we have LLM rankings
    has_llm_rankings = any(result["llm_rank"] != -1 for result in llm_results.values())

    # Create combined results with all ranking information
    combined_results = []
    ranking_info = []

    for i, (pipeline, estimated_cost, actual_cost, output_docs) in enumerate(
        pipeline_results
    ):
        # Combine ranking information
        combined_info = {
            "pipeline_id": pipeline_ids[i],
            "llm_rank": llm_results.get(i, {}).get("llm_rank", -1),
            "scoring_rank": scoring_results.get(i, {}).get("scoring_rank", -1),
            "llm_score": llm_results.get(i, {}).get("llm_score", -1),
            "scoring_value": scoring_results.get(i, {}).get("scoring_value", 0),
        }

        combined_results.append(
            (pipeline, estimated_cost, actual_cost, output_docs, combined_info)
        )
        ranking_info.append((i, combined_info))

    # Sort by LLM rank if available, otherwise by scoring rank
    if has_llm_rankings:
        sorted_indices = sorted(
            range(len(combined_results)),
            key=lambda i: llm_results.get(i, {}).get("llm_rank", float("inf")),
        )
    else:
        sorted_indices = sorted(
            range(len(combined_results)),
            key=lambda i: scoring_results.get(i, {}).get("scoring_rank", float("inf")),
        )

    # Create final sorted results
    sorted_results = [combined_results[i] for i in sorted_indices]

    # Print comparison table
    print_ranking_comparison_table(sorted_results, console)

    return sorted_results


def print_ranking_comparison_table(
    ranked_results: List[
        Tuple[List[Dict[str, Any]], float, float, List[Dict[str, Any]], Dict[str, Any]]
    ],
    console: Console,
    title: str = "Pipeline Ranking Comparison",
) -> None:
    """
    Prints a comparison table of pipeline rankings.

    Args:
        ranked_results: List of pipeline results with ranking information
        console: Console for logging
        title: Title for the table
    """
    table = Table(title=title)
    table.add_column("Pipeline ID", style="cyan")
    table.add_column("Operations", justify="right")
    table.add_column("Cost ($)", justify="right")
    table.add_column("LLM Judge Rank", justify="center", style="green")
    table.add_column("Scoring Func Rank", justify="center", style="yellow")
    table.add_column("LLM Score (lower is better)", justify="right")
    table.add_column("Scoring Func Value", justify="right")

    # Add rows to the table
    for pipeline, estimated_cost, actual_cost, _, ranking_info in ranked_results:
        pipeline_id = ranking_info["pipeline_id"]
        llm_rank = ranking_info["llm_rank"]
        scoring_rank = ranking_info["scoring_rank"]
        llm_score = ranking_info["llm_score"]
        scoring_value = ranking_info["scoring_value"]

        table.add_row(
            pipeline_id,
            str(len(pipeline)),
            f"{actual_cost:.6f}",
            str(llm_rank) if llm_rank != -1 else "N/A",
            str(scoring_rank) if scoring_rank != -1 else "N/A",
            str(llm_score) if llm_score != -1 else "N/A",
            (
                f"{scoring_value:.2f}"
                if isinstance(scoring_value, (int, float))
                else "N/A"
            ),
        )

    console.log(table)


def compare_sampling_strategies(
    results: Dict[
        str,
        Tuple[
            List[
                Tuple[
                    List[Dict[str, Any]],
                    float,
                    float,
                    List[Dict[str, Any]],
                    Dict[str, Any],
                ]
            ],
            float,
        ],
    ],
    console: Console,
    budget_num_pipelines: int = 20,
    log_dir: Optional[str] = None,
) -> None:
    """
    Compares the results of different sampling strategies and prints a comparison table.
    Also generates a scatter plot of cost vs scoring function value if matplotlib is available.

    Args:
        results: Dictionary mapping strategy names to (ranked_results, sampling_cost) tuples
        console: Console for logging
        budget_num_pipelines: Maximum number of pipelines sampled
        log_dir: Directory to save the plot to (if None, plot is not saved)
    """
    console.log("\n=== SAMPLING STRATEGIES COMPARISON ===\n")

    table = Table(title="Sampling Strategies Comparison")
    table.add_column("Strategy", style="cyan")
    table.add_column("Pipelines Sampled", justify="right")
    table.add_column("Successful Pipelines", justify="right")
    table.add_column("Total Cost ($)", justify="right")
    table.add_column("Best Pipeline ID", style="green")
    table.add_column("Best Pipeline Cost ($)", justify="right")
    table.add_column("Best Pipeline Operations", justify="right")

    for strategy, (ranked_results, cost) in results.items():
        # Get information about the best pipeline
        if ranked_results:
            best_pipeline, best_estimated_cost, best_actual_cost, _, best_info = (
                ranked_results[0]
            )
            best_pipeline_id = best_info.get("pipeline_id", "N/A")
            best_pipeline_ops = len(best_pipeline)
        else:
            best_pipeline_id = "None found"
            best_actual_cost = 0
            best_pipeline_ops = 0

        table.add_row(
            strategy.upper(),
            str(
                len(ranked_results) + (budget_num_pipelines - len(ranked_results))
            ),  # Total attempted
            str(len(ranked_results)),  # Successful
            f"{cost:.6f}",
            best_pipeline_id,
            f"{best_actual_cost:.6f}" if best_actual_cost > 0 else "N/A",
            str(best_pipeline_ops),
        )

    console.log(table)

    # Print a table of top 3 pipelines from each strategy for comparison
    console.log("\n=== TOP PIPELINES BY STRATEGY ===\n")
    for strategy, (ranked_results, _) in results.items():
        console.log(f"\nTop pipelines from {strategy.upper()} strategy:")
        print_ranking_comparison_table(
            ranked_results[:3], console, title=f"Top {strategy.upper()} Pipelines"
        )

    # Create a visualization of cost vs scoring function value
    try:
        import os
        from datetime import datetime

        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import Normalize

        console.log("\n=== GENERATING VISUALIZATION ===\n")

        # Create the figure for the first plot
        plt.figure(figsize=(10, 6))

        # Define colors for different strategies
        colors = {"random": "blue", "ucb": "red"}

        # Plot each strategy's results
        for strategy, (ranked_results, _) in results.items():
            costs = []
            scores = []
            labels = []

            for pipeline, estimated_cost, actual_cost, _, info in ranked_results:
                # Extract cost and scoring value
                costs.append(actual_cost)
                score = info.get("scoring_value", 0)
                scores.append(score)
                labels.append(info.get("pipeline_id", ""))

            # Plot this strategy's points
            plt.scatter(
                costs,
                scores,
                color=colors.get(strategy, "gray"),
                alpha=0.7,
                label=strategy.upper(),
                s=80,
            )

            # Add pipeline IDs as annotations
            for i, label in enumerate(labels):
                plt.annotate(
                    label,
                    (costs[i], scores[i]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                )

        # Add trend lines (optional)
        for strategy, (ranked_results, _) in results.items():
            if len(ranked_results) > 1:
                costs = [result[2] for result in ranked_results]
                scores = [
                    result[4].get("scoring_value", 0) for result in ranked_results
                ]

                if len(costs) > 1:
                    try:
                        # Simple linear regression
                        z = np.polyfit(costs, scores, 1)
                        p = np.poly1d(z)

                        # Plot trend line
                        x_trend = np.linspace(min(costs), max(costs), 100)
                        plt.plot(
                            x_trend,
                            p(x_trend),
                            linestyle="--",
                            color=colors.get(strategy, "gray"),
                            alpha=0.5,
                        )
                    except Exception as e:
                        console.log(
                            f"Could not generate trend line for {strategy}: {e}"
                        )

        # Add details to the plot
        plt.title("Pipeline Comparison: Cost vs Scoring Value")
        plt.xlabel("Pipeline Execution Cost ($)")
        plt.ylabel("Scoring Function Value (higher is better)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Generate timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save the plot if log_dir is provided
        if log_dir:
            try:
                # Make sure the directory exists
                os.makedirs(log_dir, exist_ok=True)

                # Generate filename for the first plot
                filename = os.path.join(log_dir, f"pipeline_comparison_{timestamp}.png")

                # Save the figure
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                console.log(f"Plot saved to: {filename}")
            except Exception as e:
                console.log(f"Could not save plot: {e}")

        # Close the first figure
        plt.close()

        # Create a new figure for the cost comparison (second plot)
        plt.figure(figsize=(10, 6))

        # Track min and max operations across all strategies for consistent color scaling
        min_ops = float("inf")
        max_ops = 0

        # First pass to find min/max operations for color normalization
        for strategy, (ranked_results, _) in results.items():
            for pipeline, _, _, _, _ in ranked_results:
                num_ops = len(pipeline)
                min_ops = min(min_ops, num_ops)
                max_ops = max(max_ops, num_ops)

        # Create the color normalization
        norm = Normalize(vmin=min_ops, vmax=max_ops)
        cmap = cm.viridis

        # Keep reference to one scatter plot for the colorbar
        scatter_for_colorbar = None

        # Plot data for each strategy
        for strategy, (ranked_results, _) in results.items():
            estimated_costs = []
            actual_costs = []
            num_operations = []
            labels = []

            for pipeline, estimated_cost, actual_cost, _, info in ranked_results:
                estimated_costs.append(estimated_cost)
                actual_costs.append(actual_cost)
                num_operations.append(len(pipeline))
                labels.append(info.get("pipeline_id", ""))

            # Create scatter plot with colors based on operation count
            scatter = plt.scatter(
                estimated_costs,
                actual_costs,
                c=num_operations,
                cmap=cmap,
                norm=norm,
                alpha=0.7,
                s=80,
            )

            # Save reference to one scatter plot for colorbar
            if scatter_for_colorbar is None and len(num_operations) > 0:
                scatter_for_colorbar = scatter

            # Add pipeline IDs as annotations
            for i, label in enumerate(labels):
                plt.annotate(
                    label,
                    (estimated_costs[i], actual_costs[i]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                )

        # Add reference line for perfect estimation (y=x)
        all_costs = []
        for strategy, (ranked_results, _) in results.items():
            all_costs.extend([r[1] for r in ranked_results])  # estimated costs
            all_costs.extend([r[2] for r in ranked_results])  # actual costs

        if all_costs:
            min_cost = min(all_costs)
            max_cost = max(all_costs)
            plt.plot([min_cost, max_cost], [min_cost, max_cost], "k--", alpha=0.5)

        # Add colorbar for operation count - link to a specific scatter plot
        if scatter_for_colorbar is not None:
            cbar = plt.colorbar(scatter_for_colorbar)
            cbar.set_label("Number of Operations")
        else:
            # Fallback if no scatter plots were created
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm)
            cbar.set_label("Number of Operations")

        # Add details to the plot
        plt.title("Pipeline Cost Estimation Analysis")
        plt.xlabel("Estimated Pipeline Cost ($)")
        plt.ylabel("Actual Pipeline Cost ($)")
        plt.grid(True, alpha=0.3)

        # Make the plot square to emphasize the reference line
        plt.axis("equal")

        # Save the plot if log_dir is provided
        if log_dir:
            try:
                # Generate filename for the second plot
                filename = os.path.join(log_dir, f"cost_comparison_{timestamp}.png")

                # Save the figure
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                console.log(f"Cost comparison plot saved to: {filename}")
            except Exception as e:
                console.log(f"Could not save cost comparison plot: {e}")

        plt.close()

    except ImportError:
        console.log("Could not generate visualization - matplotlib not available")
    except Exception as e:
        console.log(f"Error generating visualization: {e}")
