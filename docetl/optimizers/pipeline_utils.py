"""
This module provides shared utility functions for pipeline execution and evaluation.
It contains common functionality used by different sampling strategies.
"""

import copy
import json
import os  # Added for directory creation
import time
from datetime import datetime  # Added for timestamp
from typing import Any, Callable, Dict, List, Optional, Tuple

from rich.console import Console
from rich.table import Table

from docetl.optimizers.utils import LLMClient

# Try importing kendalltau, handle if scipy is not installed
try:
    from scipy.stats import kendalltau
except ImportError:
    kendalltau = None

# Try importing metrics, handle if dependencies are not installed
try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None

try:
    import numpy as np  # Required for ndcg_score input format
    from sklearn.metrics import ndcg_score
except ImportError:
    ndcg_score = None
    np = None  # Ensure np is None if sklearn is not available


# Default criteria if dynamic generation fails
DEFAULT_EVALUATION_CRITERIA = """
1.  **Relevance and Correctness (Precision):** Do the output documents accurately reflect the transformations and goals specified in the original pipeline? Are the results correct given the (unseen) input data? For instance, if the original pipeline aimed to extract specific entities, does the output contain only those entities and are they accurate?
2.  **Completeness (Recall):** Does the output include all the expected information or results based on the original pipeline's purpose? For example, if the original pipeline aimed to extract *all unique* instances of X, does the output capture the maximum number of valid unique instances? If it aimed to generate reports covering specific topics (X, Y, Z), does the output cover these topics comprehensively?
"""


def _generate_evaluation_criteria(
    original_pipeline_config: Dict[str, Any],
    llm_client: LLMClient,
    console: Console,
) -> str:
    """
    Generates task-specific evaluation criteria using the rewrite LLM agent.

    Args:
        original_pipeline_config: The configuration of the original pipeline.
        llm_client: The LLM client with a rewrite agent configured.
        console: Console for logging.

    Returns:
        A string containing numbered evaluation criteria, or default criteria on failure.
    """
    console.log("Generating dynamic evaluation criteria for LLM judge...")
    try:
        original_config_str = json.dumps(original_pipeline_config, indent=2)
        system_prompt = "You are an AI assistant expert in evaluating data processing pipelines. Your task is to define specific criteria for judging pipeline outputs based on an original pipeline's configuration."

        prompt = f"""
        Analyze the following original data pipeline configuration:
        ```json
        {original_config_str}
        ```

        Based *only* on this configuration, define 2-3 specific evaluation criteria (as a numbered list) that can be used to judge how well the output of a *rewritten* version of this pipeline achieves the original pipeline's intended purpose.

        Focus on criteria related to:
        - **Recall:** Does the output include all the necessary information? (e.g., finding *all* instances, covering *all* required topics). More is better.
        - **Quality:** Is the output free of irrelevant information? Does it have high information density? Is it simple and concise?

        **Example Task:** If the original pipeline extracts company names, criteria might be:
        1. Does the output list include *all* company names present in the source? More is better.
        2. Is the output format clean and focused, containing only the extracted company names without extraneous text, metadata, or formatting artifacts?

        **Your Output:** Provide *only* the numbered list of criteria as a single string. Do not include any preamble or explanation.
        """

        parameters = {
            "type": "object",
            "properties": {"evaluation_criteria": {"type": "string"}},
            "required": ["evaluation_criteria"],
        }

        response = llm_client.generate_rewrite(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system_prompt,
            parameters=parameters,
        )

        criteria = json.loads(response.choices[0].message.content)[
            "evaluation_criteria"
        ]
        console.log(f"Successfully generated evaluation criteria:\n{criteria}")
        return criteria

    except Exception as e:
        console.log(
            f"[bold yellow]Warning:[/bold yellow] Failed to generate dynamic evaluation criteria: {e}. Falling back to default criteria."
        )
        return DEFAULT_EVALUATION_CRITERIA


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
    run_operation_func: Callable,
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

    # Generate dynamic evaluation criteria for the LLM judge
    evaluation_criteria = _generate_evaluation_criteria(
        original_pipeline_config, llm_client, console
    )

    # Run LLM judge evaluation independently using the generated criteria
    llm_results = evaluate_with_llm_judge(
        pipeline_results,
        original_pipeline_config,
        pipeline_ids,
        llm_client,
        min_docs_length,
        console,
        run_operation_func,
        evaluation_criteria,  # Pass the generated criteria
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
    run_operation_func: Callable,
    evaluation_criteria: str,  # Added parameter for criteria
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
        run_operation_func: Function to execute operations
        evaluation_criteria: Task-specific criteria for the LLM judge.

    Returns:
        Dictionary mapping pipeline index to LLM judge results
    """
    console.log("Evaluating pipelines using LLM as judge...")

    # Convert original pipeline config to string representation
    original_config_str = json.dumps(original_pipeline_config, indent=2)

    # Create a list of docs where each doc represents the outputs
    outputs_as_docs = []
    for original_idx in range(len(pipeline_results)):
        pipeline_config, estimated_cost, actual_cost, output_docs = pipeline_results[
            original_idx
        ]
        pipeline_config_str = json.dumps(pipeline_config, indent=2)
        output_docs_str = json.dumps(output_docs[:10], indent=2)
        output_doc = f"Sample Outputs:\n{output_docs_str}"
        outputs_as_docs.append(
            {
                "idx": original_idx,
                "output": output_doc,
            }
        )

    # Now create a rank operation config using the dynamic criteria
    rank_op_config = {
        "name": "rank_pipeline_outputs",
        "type": "rank",
        "model": llm_client.judge_agent_model,
        "prompt": f"""
You are an expert evaluator specializing in data processing pipelines. Your task is to rank several *rewritten* versions of an original data pipeline. You will be given the configuration of the **original pipeline** and the **output documents** produced by each **rewritten pipeline** when run on a sample of input data.

**Your goal is to determine which rewritten pipeline best achieves the *intended purpose* of the original pipeline, based *only* on the provided output documents.**

**Original Pipeline Configuration:**
```json
{original_config_str}
```

**Evaluation Criteria (Rank from best to worst):**

Use these specific criteria, derived from the original pipeline's goals, to evaluate each rewritten pipeline's output:
{evaluation_criteria}

**Input Format:**
You will some sample outputs from the rewritten pipeline. One "Document" is a collection of outputs from a _single_ rewritten pipeline.

**Task:**
Analyze the `Sample Outputs` section for each provided rewritten pipeline. Evaluate them using the **Evaluation Criteria** provided above. Order them from meeting the criteria the best to worst.
""",
        "input_keys": ["output"],
        "rerank_call_budget": 10,
        "direction": "desc",
        "batch_size": 1,
        "num_calibration_docs": 3,
        "num_top_items_per_window": 2,
        "litellm_kwargs": {
            "temperature": 0.2,
        },
    }

    # Execute the rank operation
    try:
        ranked_outputs, ranking_cost = run_operation_func(
            rank_op_config,
            outputs_as_docs,
            return_instance=False,
            return_cost=True,
        )
        llm_client.add_cost(ranking_cost)  # Assumes run_operation_func returns cost
    except Exception as e:
        console.log(
            f"[bold red]Error:[/bold red] Failed to execute ranking operation: {e}"
        )
        # Handle failure: maybe return empty results or raise exception
        return {idx: {"llm_rank": -1} for idx in range(len(pipeline_results))}

    # Assign LLM ranks
    llm_rank_map = {
        ranked_output["idx"]: ranked_output["_rank"] for ranked_output in ranked_outputs
    }

    # Create result dictionary with all LLM judge information
    llm_results = {}
    for idx in range(len(pipeline_results)):
        llm_results[idx] = {
            "llm_rank": llm_rank_map.get(
                idx, len(ranked_outputs) + 1
            ),  # Use -1 for unranked items
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
    table.add_column("Scoring Func Value", justify="right")

    # Add rows to the table
    for pipeline, estimated_cost, actual_cost, _, ranking_info in ranked_results:
        pipeline_id = ranking_info["pipeline_id"]
        llm_rank = ranking_info["llm_rank"]
        scoring_rank = ranking_info["scoring_rank"]
        scoring_value = ranking_info["scoring_value"]

        table.add_row(
            pipeline_id,
            str(len(pipeline)),
            f"{actual_cost:.6f}",
            str(llm_rank) if llm_rank != -1 else "N/A",
            str(scoring_rank) if scoring_rank != -1 else "N/A",
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
    Compares the results of different sampling strategies, prints a comparison table,
    logs detailed results to JSON, and generates visualizations. Includes Kendall's Tau,
    Spearman's Rho, and nDCG@10 to compare LLM judge ranking vs. scoring function ranking.

    Args:
        results: Dictionary mapping strategy names to (ranked_results, sampling_cost) tuples
        console: Console for logging
        budget_num_pipelines: Maximum number of pipelines sampled
        log_dir: Directory to save plots and results JSON to (if None, not saved)
    """
    console.log("\n=== SAMPLING STRATEGIES COMPARISON ===\n")

    # Generate timestamp once for consistent filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy_results_data = {}  # To store data for JSON logging

    table = Table(title="Sampling Strategies Comparison")
    table.add_column("Strategy", style="cyan")
    table.add_column("Pipelines Sampled", justify="right")
    table.add_column("Successful Pipelines", justify="right")
    table.add_column("Total Cost ($)", justify="right")
    table.add_column("Best Pipeline ID", style="green")
    table.add_column("Best Pipeline Cost ($)", justify="right")
    table.add_column("Best Pipeline Ops", justify="right")
    table.add_column("Kendall's Tau", justify="right", style="magenta")
    table.add_column("Spearman's Rho", justify="right", style="blue")  # New column
    table.add_column("nDCG@10", justify="right", style="yellow")  # New column

    for strategy, (ranked_results, cost) in results.items():
        # Initialize metric values
        tau_value, rho_value, ndcg_value = "N/A", "N/A", "N/A"
        tau_float, rho_float, ndcg_float = None, None, None
        llm_ranks = []
        scoring_ranks = []
        llm_relevance_scores = []  # For nDCG: Higher score = better rank
        scoring_relevance_scores = []  # For nDCG: Higher score = better rank
        valid_rank_pairs = 0
        max_rank = 0  # Keep track of the highest rank number

        if len(ranked_results) >= 2:
            # Prepare rank lists for correlation calculation
            for _, _, _, _, info in ranked_results:
                llm_rank = info.get("llm_rank", -1)
                scoring_rank = info.get("scoring_rank", -1)
                # Only include if both ranks are valid
                if llm_rank != -1 and scoring_rank != -1:
                    llm_ranks.append(llm_rank)
                    scoring_ranks.append(scoring_rank)
                    max_rank = max(max_rank, llm_rank, scoring_rank)  # Update max rank
                    valid_rank_pairs += 1

            # Calculate relevance scores after finding max_rank
            if valid_rank_pairs > 0:
                for _, _, _, _, info in ranked_results:
                    llm_rank = info.get("llm_rank", -1)
                    scoring_rank = info.get("scoring_rank", -1)
                    if llm_rank != -1 and scoring_rank != -1:
                        # Relevance score: higher is better (max_rank - rank + 1)
                        llm_relevance_scores.append(max_rank - llm_rank + 1)
                        scoring_relevance_scores.append(max_rank - scoring_rank + 1)

            if valid_rank_pairs >= 2:
                # Calculate Kendall's Tau
                if kendalltau is None:
                    tau_value = "SciPy missing"
                else:
                    try:
                        tau, _ = kendalltau(llm_ranks, scoring_ranks)
                        tau_value = f"{tau:.3f}"
                        tau_float = tau
                    except Exception as e:
                        console.log(
                            f"Could not calculate Kendall's Tau for {strategy}: {e}"
                        )
                        tau_value = "Error"

                # Calculate Spearman's Rho
                if spearmanr is None:
                    rho_value = "SciPy missing"
                else:
                    try:
                        rho, _ = spearmanr(llm_ranks, scoring_ranks)
                        rho_value = f"{rho:.3f}"
                        rho_float = rho
                    except Exception as e:
                        console.log(
                            f"Could not calculate Spearman's Rho for {strategy}: {e}"
                        )
                        rho_value = "Error"

            elif valid_rank_pairs < 2:
                tau_value = "Too few pairs"
                rho_value = "Too few pairs"

            # Calculate nDCG@10 (needs at least 1 valid pair)
            if valid_rank_pairs >= 1:
                if ndcg_score is None or np is None:
                    ndcg_value = "Sklearn missing"
                else:
                    try:
                        # nDCG compares the ranking induced by scores against true relevance
                        # y_true: relevance based on the LLM judge (ground truth)
                        # y_score: scores that produced the scoring_func ranking (use its relevance)
                        # We need them as 2D arrays [[scores]]
                        true_relevance = np.asarray([llm_relevance_scores])
                        predicted_scores = np.asarray([scoring_relevance_scores])

                        ndcg = ndcg_score(true_relevance, predicted_scores, k=10)
                        ndcg_value = f"{ndcg:.3f}"
                        ndcg_float = ndcg
                    except Exception as e:
                        console.log(f"Could not calculate nDCG@10 for {strategy}: {e}")
                        ndcg_value = "Error"
            else:
                ndcg_value = "Too few pairs"

        else:  # Fewer than 2 results overall
            tau_value = "Too few results"
            rho_value = "Too few results"
            ndcg_value = "Too few results"

        # Store results for JSON logging
        strategy_results_data[strategy] = {
            "sampling_cost": cost,
            "kendall_tau": tau_float,
            "spearman_rho": rho_float,  # Add Spearman
            "ndcg_at_10": ndcg_float,  # Add nDCG
            "ranked_results": ranked_results,  # Store the full ranked results
        }

        # Get information about the best pipeline (based on LLM rank primarily)
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
            tau_value,
            rho_value,  # Add Spearman value to row
            ndcg_value,  # Add nDCG value to row
        )

    console.log(table)

    # Log detailed results to JSON if log_dir is provided
    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            json_filename = os.path.join(log_dir, f"strategy_results_{timestamp}.json")
            with open(json_filename, "w") as f:
                # Use default=str to handle potential non-serializable types gracefully
                json.dump(strategy_results_data, f, indent=2, default=str)
            console.log(f"Detailed strategy results saved to: {json_filename}")
        except Exception as e:
            console.log(f"[bold red]Error:[/bold red] Could not save results JSON: {e}")

    # Print a table of top 3 pipelines from each strategy for comparison
    console.log("\n=== TOP PIPELINES BY STRATEGY ===\n")
    for strategy, (ranked_results, _) in results.items():
        console.log(f"\nTop pipelines from {strategy.upper()} strategy:")
        print_ranking_comparison_table(
            ranked_results[:3], console, title=f"Top {strategy.upper()} Pipelines"
        )

    # Create visualizations (Cost vs Scoring Value and Cost Estimation Analysis)
    # Check for dependencies before attempting to plot
    if (
        log_dir
        and ("matplotlib" in sys.modules or "matplotlib.pyplot" in sys.modules)
        and np
    ):
        try:
            # import os # Already imported
            # from datetime import datetime # Already imported
            import matplotlib.cm as cm
            import matplotlib.pyplot as plt

            # import numpy as np # Already imported at top
            from matplotlib.colors import Normalize

            console.log("\n=== GENERATING VISUALIZATIONS ===\n")

            # --- Plot 1: Cost vs Scoring Value ---
            plt.figure(figsize=(10, 6))
            colors = {"random": "blue", "ucb": "red"}  # Example colors

            for strategy, (ranked_results, _) in results.items():
                costs = []
                scores = []
                labels = []
                for pipeline, estimated_cost, actual_cost, _, info in ranked_results:
                    costs.append(actual_cost)
                    score = info.get("scoring_value", 0)
                    if not isinstance(score, (int, float)):
                        score = 0
                    scores.append(score)
                    labels.append(info.get("pipeline_id", ""))

                if costs and scores:
                    plt.scatter(
                        costs,
                        scores,
                        color=colors.get(strategy, "gray"),
                        alpha=0.7,
                        label=strategy.upper(),
                        s=80,
                    )
                    for i, label in enumerate(labels):
                        plt.annotate(
                            label,
                            (costs[i], scores[i]),
                            textcoords="offset points",
                            xytext=(5, 5),
                            fontsize=8,
                        )

            # Optional: Add trend lines (excluding outliers)
            for strategy, (ranked_results, _) in results.items():
                costs, scores = [], []
                for _, _, actual_cost, _, info in ranked_results:
                    score = info.get("scoring_value", 0)
                    if isinstance(score, (int, float)):
                        costs.append(actual_cost)
                        scores.append(score)

                if len(scores) >= 3:
                    try:
                        scores_np, costs_np = np.array(scores), np.array(costs)
                        min_idx, max_idx = np.argmin(scores_np), np.argmax(scores_np)
                        mask = np.ones(len(scores_np), dtype=bool)
                        for idx in {min_idx, max_idx}:
                            mask[idx] = False
                        filtered_costs, filtered_scores = (
                            costs_np[mask],
                            scores_np[mask],
                        )

                        if len(filtered_costs) >= 2:
                            z = np.polyfit(filtered_costs, filtered_scores, 1)
                            p = np.poly1d(z)
                            x_trend = np.linspace(min(costs_np), max(costs_np), 100)
                            plt.plot(
                                x_trend,
                                p(x_trend),
                                linestyle="--",
                                color=colors.get(strategy, "gray"),
                                alpha=0.5,
                                label=f"{strategy.upper()} Trend (excl. outliers)",
                            )
                        else:
                            console.log(
                                f"Skipping trend line for {strategy}: Not enough points after removing min/max score."
                            )
                    except Exception as e:
                        console.log(
                            f"Could not generate trend line for {strategy} (excluding outliers): {e}"
                        )
                elif len(scores) > 0:
                    console.log(
                        f"Skipping trend line for {strategy}: Needs at least 3 points to exclude min/max."
                    )

            plt.title("Pipeline Comparison: Cost vs Scoring Value")
            plt.xlabel("Pipeline Execution Cost ($)")
            plt.ylabel("Scoring Function Value (higher is better)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            filename1 = os.path.join(log_dir, f"pipeline_comparison_{timestamp}.png")
            plt.savefig(filename1, dpi=300, bbox_inches="tight")
            console.log(f"Plot saved to: {filename1}")
            plt.close()  # Close the figure

            # --- Plot 2: Cost Estimation Analysis ---
            plt.figure(figsize=(10, 6))
            min_ops, max_ops = float("inf"), 0
            for strategy, (ranked_results, _) in results.items():
                for pipeline, _, _, _, _ in ranked_results:
                    num_ops = len(pipeline)
                    min_ops, max_ops = min(min_ops, num_ops), max(max_ops, num_ops)

            norm = Normalize(vmin=min_ops, vmax=max_ops)
            cmap = cm.viridis
            scatter_for_colorbar = None

            for strategy, (ranked_results, _) in results.items():
                est_costs, act_costs, num_ops_list, labels = [], [], [], []
                for pipeline, est_cost, act_cost, _, info in ranked_results:
                    est_costs.append(est_cost)
                    act_costs.append(act_cost)
                    num_ops_list.append(len(pipeline))
                    labels.append(info.get("pipeline_id", ""))

                if est_costs:  # Check if there's data to plot
                    scatter = plt.scatter(
                        est_costs,
                        act_costs,
                        c=num_ops_list,
                        cmap=cmap,
                        norm=norm,
                        alpha=0.7,
                        s=80,
                    )
                    if scatter_for_colorbar is None:
                        scatter_for_colorbar = scatter
                    for i, label in enumerate(labels):
                        plt.annotate(
                            label,
                            (est_costs[i], act_costs[i]),
                            textcoords="offset points",
                            xytext=(5, 5),
                            fontsize=8,
                        )

            all_costs = []
            for _, (ranked_results, _) in results.items():
                all_costs.extend([r[1] for r in ranked_results])  # estimated
                all_costs.extend([r[2] for r in ranked_results])  # actual
            if all_costs:
                min_c, max_c = min(all_costs) if all_costs else 0, (
                    max(all_costs) if all_costs else 1
                )
                plt.plot(
                    [min_c, max_c],
                    [min_c, max_c],
                    "k--",
                    alpha=0.5,
                    label="Perfect Estimation (y=x)",
                )

            if scatter_for_colorbar:
                cbar = plt.colorbar(scatter_for_colorbar)
                cbar.set_label("Number of Operations")
            else:  # Fallback if no data was plotted
                sm = cm.ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array([])
                cbar = plt.colorbar(sm)
                cbar.set_label("Number of Operations")

            plt.title("Pipeline Cost Estimation Analysis")
            plt.xlabel("Estimated Pipeline Cost ($)")
            plt.ylabel("Actual Pipeline Cost ($)")
            plt.grid(True, alpha=0.3)
            plt.axis("equal")  # Make axes equal for y=x line emphasis
            plt.legend()  # Show legend including the y=x line
            filename2 = os.path.join(log_dir, f"cost_comparison_{timestamp}.png")
            plt.savefig(filename2, dpi=300, bbox_inches="tight")
            console.log(f"Cost comparison plot saved to: {filename2}")
            plt.close()  # Close the figure

        except ImportError:
            console.log(
                "[bold yellow]Warning:[/bold yellow] Could not generate visualization - matplotlib or numpy not available."
            )
        except Exception as e:
            console.log(
                f"[bold red]Error:[/bold red] Error generating visualization: {e}"
            )
    elif log_dir:
        console.log(
            "[bold yellow]Warning:[/bold yellow] Skipping visualization generation - matplotlib or numpy not imported."
        )


# Need to import sys for the check above
import sys
