import pytest
from docetl.operations.rank import RankOperation
from docetl.operations.map import MapOperation
from docetl.runner import DSLRunner
import json
import time
from scipy.stats import kendalltau  # Import for Kendall's Tau calculation
from rich.console import Console
from rich.table import Table
import random
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

import lotus
from lotus.models import LM, LiteLLMRM
from lotus.vector_store import FaissVS
lm = LM(model="gemini/gemini-2.0-flash")
rm = LiteLLMRM(model="text-embedding-3-small")
# vs = FaissVS()


lotus.settings.configure(lm=lm, rm=rm)

console = Console()

api_wrapper = DSLRunner(
    {
        "default_model": "gemini/gemini-2.0-flash",
        "operations": [],
        "pipeline": {"steps": [], "output": {"path": "/tmp/testingdocetl.json"}},
    },
    max_threads=64,
)

def calculate_ndcg(y_true, y_pred, k=10):
    """
    Calculate the Normalized Discounted Cumulative Gain (NDCG) at k.
    
    Args:
        y_true (np.ndarray): Ground truth relevance scores (1D array).
        y_pred (np.ndarray): Predicted relevance scores (1D array).
        k (int): The number of items to consider for NDCG calculation.
        
    Returns:
        float: The NDCG@k score.
    """
    # Reshape to match scikit-learn's expected format if needed
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(1, -1)
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(1, -1)
    
    # Handle edge case where all relevance scores are 0
    if np.sum(y_true) == 0:
        return 0.0
    
    return ndcg_score(y_true, y_pred, k=k)

def test_order_square_dataset(api_wrapper):
    """
    Test ordering methods using a synthetic dataset of ASCII squares.
    Each square is n × n pixels, where n starts at 20 and increases by 3 for each subsequent square.
    The correct ordering is by square area (smallest to largest).
    """
    # Generate the synthetic dataset of ASCII squares
    squares_data = []
    num_squares = 100
    
    for i in range(num_squares):
        size = 20 + i
        # Create a square of '#' characters with dimensions size × size
        square = '\n'.join(['#' * size for _ in range(size)])
        
        # Create a document with the square and metadata
        squares_data.append({
            "id": f"square_{i}",
            "size": size,
            "area": size * size,  # The ground truth metric for ordering
            "content": square,
            "description": f"A square of size {size}×{size}"
        })
    
    # create the order operation
    order_config = {
        "name": "order_by_square_size",
        "type": "order",
        "batch_size": 10,  # As requested in the specification
        "rerank_call_budget": 10,
        "prompt": """
            Order these ASCII squares based on their size (area).
            Smaller squares should be ranked lower (first) and larger squares should be ranked higher (last).
            
            Consider:
            - The dimensions of each square
            - The total area of each square
            - Visual comparison of relative sizes
        """,
        "input_keys": ["content"],
        "direction": "asc",
        "verbose": True,
        # "bypass_cache": True
    }
    
    order_operation = RankOperation(
        api_wrapper,
        order_config,
        default_model="gemini/gemini-2.0-flash",
        max_threads=64,
    )
    likert_initial_config = order_config.copy()
    likert_initial_config["initial_ordering_method"] = "likert"
    order_operation_likert_initial = RankOperation(
        api_wrapper,
        likert_initial_config,
        default_model="gemini/gemini-2.0-flash",
        max_threads=64,
    )
    
    # Calculate ground truth ranks
    # Sort by area (which is our ground truth metric) and assign ranks
    ground_truth = sorted(squares_data, key=lambda x: x["area"])
    for i, doc in enumerate(ground_truth):
        doc["_ground_truth_rank"] = i + 1
    
    # Create a mapping of document IDs to ground truth ranks
    ground_truth_ranks = {doc["id"]: doc["_ground_truth_rank"] for doc in ground_truth}
    
    # Track execution time and cost for each method
    method_metrics = {}
    
    # Apply standard order operation
    start_time = time.time()
    order_results, order_cost = order_operation.execute(squares_data)
    method_metrics["Ours (Embedding Initial)"] = {
        "runtime": time.time() - start_time,
        "cost": order_cost
    }
    
    # Apply likert initial order operation
    start_time = time.time()
    order_results_likert_initial, order_cost_likert_initial = order_operation_likert_initial.execute(squares_data)
    method_metrics["Ours (Likert Initial)"] = {
        "runtime": time.time() - start_time,
        "cost": order_cost_likert_initial
    }
    
    # Apply baseline comparison method
    start_time = time.time()
    order_results_baseline, order_cost_baseline = order_operation._execute_comparison_qurk(squares_data)
    method_metrics["Baseline Comparison"] = {
        "runtime": time.time() - start_time,
        "cost": order_cost_baseline
    }
    
    # Apply rating based order operation
    start_time = time.time()
    order_results_rating, order_cost_rating = order_operation._execute_rating_embedding_qurk(squares_data)
    method_metrics["Embedding Rating"] = {
        "runtime": time.time() - start_time,
        "cost": order_cost_rating
    }
    
    # Apply calibrated embedding sort order operation
    start_time = time.time()
    order_results_calibrated_embedding_sort, order_cost_calibrated_embedding_sort = order_operation._execute_calibrated_embedding_sort(squares_data)
    method_metrics["Calibrated Embedding Rating"] = {
        "runtime": time.time() - start_time,
        "cost": order_cost_calibrated_embedding_sort
    }
    
    
    # Apply sliding window order operations
    start_time = time.time()
    order_results_sliding_window_embedding, order_cost_sliding_window_embedding = order_operation._execute_sliding_window_qurk(squares_data, initial_ordering_method="embedding")
    method_metrics["Embedding Sliding Window"] = {
        "runtime": time.time() - start_time,
        "cost": order_cost_sliding_window_embedding
    }
    
    start_time = time.time()
    order_results_sliding_window_likert, order_cost_sliding_window_likert = order_operation._execute_sliding_window_qurk(squares_data, initial_ordering_method="likert")
    method_metrics["Likert Sliding Window"] = {
        "runtime": time.time() - start_time,
        "cost": order_cost_sliding_window_likert
    }
    
    # Apply likert rating order operation
    start_time = time.time()
    order_results_likert_rating, order_cost_likert_rating = order_operation._execute_likert_rating_qurk(squares_data)
    method_metrics["Likert Rating"] = {
        "runtime": time.time() - start_time,
        "cost": order_cost_likert_rating
    }
    
    # Execute lotus top k
    console.print("[bold blue]Sorting squares with lotus top k method[/bold blue]")
    start_time = time.time()
    df = pd.DataFrame(squares_data)
    sorted_df, stats = df.sem_topk(
        "Which square is smallest? Here is the square: {content}",
        K=10,
        return_stats=True,
    )
    end_time = time.time()
    print(stats)
    lotus_num_calls = stats["total_llm_calls"]
    lotus_results = sorted_df.to_dict(orient="records")
    lotus_cost = stats["total_tokens"] * 0.15 / 1000000
    lotus_runtime = end_time - start_time
    method_metrics["Lotus Top K"] = {
        "runtime": lotus_runtime,
        "cost": lotus_cost
    }
    
    # Extract ranks for each document from the different methods
    standard_ranks = {doc["id"]: doc["_rank"] for doc in order_results}
    likert_initial_ranks = {doc["id"]: doc["_rank"] for doc in order_results_likert_initial}
    baseline_ranks = {doc["id"]: doc["_rank"] for doc in order_results_baseline}
    rating_ranks = {doc["id"]: doc["_rank"] for doc in order_results_rating}
    sliding_window_embedding_ranks = {doc["id"]: doc["_rank"] for doc in order_results_sliding_window_embedding}
    sliding_window_likert_ranks = {doc["id"]: doc["_rank"] for doc in order_results_sliding_window_likert}
    likert_rating_ranks = {doc["id"]: doc["_rank"] for doc in order_results_likert_rating}
    lotus_ranks = {doc["id"]: doc["_rank"] for doc in lotus_results}
    calibrated_embedding_sort_ranks = {doc["id"]: doc["_rank"] for doc in order_results_calibrated_embedding_sort}
    
    # Prepare lists for Kendall's Tau computation
    doc_ids = list(ground_truth_ranks.keys())
    
    ground_truth_rank_list = [ground_truth_ranks[doc_id] for doc_id in doc_ids]
    standard_rank_list = [standard_ranks[doc_id] for doc_id in doc_ids]
    likert_initial_rank_list = [likert_initial_ranks[doc_id] for doc_id in doc_ids]
    baseline_rank_list = [baseline_ranks[doc_id] for doc_id in doc_ids]
    rating_rank_list = [rating_ranks[doc_id] for doc_id in doc_ids]
    sliding_window_embedding_rank_list = [sliding_window_embedding_ranks[doc_id] for doc_id in doc_ids]
    sliding_window_likert_rank_list = [sliding_window_likert_ranks[doc_id] for doc_id in doc_ids]
    likert_rating_rank_list = [likert_rating_ranks[doc_id] for doc_id in doc_ids]
    lotus_rank_list = [lotus_ranks.get(doc_id, 11) for doc_id in doc_ids]
    calibrated_embedding_sort_rank_list = [calibrated_embedding_sort_ranks[doc_id] for doc_id in doc_ids]
    
    # Compute Kendall's Tau correlation coefficients against ground truth
    tau_standard, p_value_standard = kendalltau(ground_truth_rank_list, standard_rank_list)
    tau_likert_initial, p_value_likert_initial = kendalltau(ground_truth_rank_list, likert_initial_rank_list)
    tau_baseline, p_value_baseline = kendalltau(ground_truth_rank_list, baseline_rank_list)
    tau_rating, p_value_rating = kendalltau(ground_truth_rank_list, rating_rank_list)
    tau_sliding_window_embedding, p_value_sliding_window_embedding = kendalltau(ground_truth_rank_list, sliding_window_embedding_rank_list)
    tau_sliding_window_likert, p_value_sliding_window_likert = kendalltau(ground_truth_rank_list, sliding_window_likert_rank_list)
    tau_likert_rating, p_value_likert_rating = kendalltau(ground_truth_rank_list, likert_rating_rank_list)
    tau_lotus, p_value_lotus = kendalltau(ground_truth_rank_list, lotus_rank_list)
    tau_calibrated_embedding_rating, p_value_calibrated_embedding_rating = kendalltau(ground_truth_rank_list, calibrated_embedding_sort_rank_list)
    
    # Store results in a list of tuples for sorting, including runtime and cost
    results = [
        ("Ours (Embedding Initial)", tau_standard, p_value_standard, method_metrics["Ours (Embedding Initial)"]["runtime"], method_metrics["Ours (Embedding Initial)"]["cost"]),
        ("Ours (Likert Initial)", tau_likert_initial, p_value_likert_initial, method_metrics["Ours (Likert Initial)"]["runtime"], method_metrics["Ours (Likert Initial)"]["cost"]),
        ("All Pairs Comparison", tau_baseline, p_value_baseline, method_metrics["Baseline Comparison"]["runtime"], method_metrics["Baseline Comparison"]["cost"]),
        ("Embedding Rating", tau_rating, p_value_rating, method_metrics["Embedding Rating"]["runtime"], method_metrics["Embedding Rating"]["cost"]),
        ("Embedding Sliding Window", tau_sliding_window_embedding, p_value_sliding_window_embedding, method_metrics["Embedding Sliding Window"]["runtime"], method_metrics["Embedding Sliding Window"]["cost"]),
        ("Likert Sliding Window", tau_sliding_window_likert, p_value_sliding_window_likert, method_metrics["Likert Sliding Window"]["runtime"], method_metrics["Likert Sliding Window"]["cost"]),
        ("Likert Rating", tau_likert_rating, p_value_likert_rating, method_metrics["Likert Rating"]["runtime"], method_metrics["Likert Rating"]["cost"]),
        ("Lotus Top K", tau_lotus, p_value_lotus, method_metrics["Lotus Top K"]["runtime"], method_metrics["Lotus Top K"]["cost"]),
        ("Calibrated Embedding Rating", tau_calibrated_embedding_rating, p_value_calibrated_embedding_rating, method_metrics["Calibrated Embedding Rating"]["runtime"], method_metrics["Calibrated Embedding Rating"]["cost"])
    ]
    
    # Sort results by Kendall's tau value in descending order
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    # Create a Rich table
    table = Table(title="Square Dataset Ordering Results (vs Ground Truth)")
    
    # Add columns
    table.add_column("Method", style="cyan")
    table.add_column("Kendall's Tau", justify="right", style="green")
    table.add_column("p-value", justify="right", style="green")
    table.add_column("Runtime (s)", justify="right", style="yellow")
    table.add_column("Cost ($)", justify="right", style="yellow")
    
    # Add rows
    for method, tau, p_value, runtime, cost in sorted_results:
        table.add_row(
            method,
            f"{tau:.4f}",
            f"{p_value:.4f}",
            f"{runtime:.2f}",
            f"{cost:.4f}"
        )
    
    # Print the table
    console.print(table)

    # Calculate NDCG@10 for each method using ground truth
    y_true = np.zeros(len(squares_data))
    
    # Have it be in descending order of size
    for i, doc in enumerate(squares_data):
        y_true[i] = len(squares_data) - i
    
    # Calculate NDCG@10 for each method
    ndcg_table = Table(title="NDCG@10 Results (vs Ground Truth)")
    ndcg_table.add_column("Method", style="cyan")
    ndcg_table.add_column("NDCG@10", justify="right", style="green")
    
    # Prepare arrays for each method's ranking
    method_results = [
        ("Ours (Embedding Initial)", order_results, standard_ranks),
        ("Ours (Likert Initial)", order_results_likert_initial, likert_initial_ranks),
        ("All Pairs Comparison", order_results_baseline, baseline_ranks),
        ("Embedding Rating", order_results_rating, rating_ranks),
        ("Embedding Sliding Window", order_results_sliding_window_embedding, sliding_window_embedding_ranks),
        ("Likert Sliding Window", order_results_sliding_window_likert, sliding_window_likert_ranks),
        ("Likert Rating", order_results_likert_rating, likert_rating_ranks),
        ("Lotus Top K", lotus_results, lotus_ranks),
        ("Calibrated Embedding Rating", order_results_calibrated_embedding_sort, calibrated_embedding_sort_ranks)
    ]
    
    # Calculate NDCG for each method
    ndcg_scores = []
    for method_name, _, method_ranks in method_results:
        # Create predicted relevance array (higher ranks should have higher scores)
        y_pred = np.zeros(len(squares_data))
        
        for i, doc in enumerate(squares_data):
            # Invert rank for scoring (higher rank = higher relevance)
            y_pred[i] = len(squares_data) - method_ranks.get(doc["id"], 11) + 1
        
        # Calculate NDCG
        ndcg_value = calculate_ndcg(y_true, y_pred, k=10)
        ndcg_scores.append((method_name, ndcg_value))
        ndcg_table.add_row(method_name, f"{ndcg_value:.4f}")
    
    # Sort NDCG scores and recreate table with sorted results
    sorted_ndcg = sorted(ndcg_scores, key=lambda x: x[1], reverse=True)
    sorted_ndcg_table = Table(title="NDCG@10 Results (vs Ground Truth) - Sorted")
    sorted_ndcg_table.add_column("Method", style="cyan")
    sorted_ndcg_table.add_column("NDCG@10", justify="right", style="green")
    
    for method_name, ndcg_value in sorted_ndcg:
        sorted_ndcg_table.add_row(method_name, f"{ndcg_value:.4f}")
    
    # Print NDCG results
    console.print(sorted_ndcg_table)

    # Return the sorted results and NDCG scores
    return sorted_results, sorted_ndcg, lotus_num_calls

def test_order_medical_transcripts_by_pain(api_wrapper):
    """
    Test ordering methods for medical transcripts from workloads/medical/raw.json
    based on patient pain levels in descending order (most pain first).
    """
    # Load medical transcripts
    with open("workloads/medical/raw.json", "r") as f:
        medical_data = json.load(f)
    
    # Add IDs to data if not present
    for i, doc in enumerate(medical_data):
        if "id" not in doc:
            doc["id"] = f"medical_{i}"
    
    # Create order operation for pain level assessment
    order_config = {
        "name": "order_by_pain_level",
        "type": "order",
        "batch_size": 10,
        "prompt": """
            Order these medical transcripts based on how much pain the patient is experiencing or reporting, from most pain to least pain.
        """,
        "input_keys": ["src"],
        "direction": "desc",  # Highest pain first
        "verbose": True,
        "rerank_call_budget": 10,
        # "bypass_cache": True
    }
    
    order_operation = RankOperation(
        api_wrapper,
        order_config,
        default_model="gemini/gemini-2.0-flash",
        max_threads=64,
    )
    
    order_config_likert_initial = order_config.copy()
    order_config_likert_initial["initial_ordering_method"] = "likert"
    order_operation_likert_initial = RankOperation(
        api_wrapper,
        order_config_likert_initial,
        default_model="gemini/gemini-2.0-flash",
        max_threads=64,
    )
    
    # Track execution time and cost for each method
    method_metrics = {}
    
    # Apply standard order operation
    start_time = time.time()
    order_results, order_cost = order_operation.execute(medical_data)
    method_metrics["Ours (Embedding Initial)"] = {
        "runtime": time.time() - start_time,
        "cost": order_cost
    }
    
    # Apply likert initial order operation
    start_time = time.time()
    order_results_likert_initial, order_cost_likert_initial = order_operation_likert_initial.execute(medical_data)
    method_metrics["Ours (Likert Initial)"] = {
        "runtime": time.time() - start_time,
        "cost": order_cost_likert_initial
    }
    
    # Apply baseline comparison method
    start_time = time.time()
    order_results_baseline, order_cost_baseline = order_operation._execute_comparison_qurk(medical_data)
    method_metrics["Baseline Comparison"] = {
        "runtime": time.time() - start_time,
        "cost": order_cost_baseline
    }
    print(f"Baseline cost: {order_cost_baseline}; runtime: {time.time() - start_time}")
    
    # Apply rating based order operation
    start_time = time.time()
    order_results_rating, order_cost_rating = order_operation._execute_rating_embedding_qurk(medical_data)
    method_metrics["Embedding Rating"] = {
        "runtime": time.time() - start_time,
        "cost": order_cost_rating
    }
    
    # Apply calibrated embedding sort order operation
    start_time = time.time()
    order_results_calibrated_embedding_sort, order_cost_calibrated_embedding_sort = order_operation._execute_calibrated_embedding_sort(medical_data)
    method_metrics["Calibrated Embedding Rating"] = {
        "runtime": time.time() - start_time,
        "cost": order_cost_calibrated_embedding_sort
    }
    
    # Apply sliding window order operations
    start_time = time.time()
    order_results_sliding_window_embedding, order_cost_sliding_window_embedding = order_operation._execute_sliding_window_qurk(medical_data, initial_ordering_method="embedding")
    method_metrics["Embedding Sliding Window"] = {
        "runtime": time.time() - start_time,
        "cost": order_cost_sliding_window_embedding
    }
    
    start_time = time.time()
    order_results_sliding_window_likert, order_cost_sliding_window_likert = order_operation._execute_sliding_window_qurk(medical_data, initial_ordering_method="likert")
    method_metrics["Likert Sliding Window"] = {
        "runtime": time.time() - start_time,
        "cost": order_cost_sliding_window_likert
    }
    
    # Apply likert rating order operation
    start_time = time.time()
    order_results_likert_rating, order_cost_likert_rating = order_operation._execute_likert_rating_qurk(medical_data)
    method_metrics["Likert Rating"] = {
        "runtime": time.time() - start_time,
        "cost": order_cost_likert_rating
    }
    
    # Execute lotus top k
    console.print("[bold blue]Sorting medical transcripts with lotus top k method[/bold blue]")
    start_time = time.time()
    df = pd.DataFrame(medical_data)
    sorted_df, stats = df.sem_topk(
        "Which medical transcripts reflect the most pain that the patient is experiencing? Here is the transcript: {src}",
        K=10,
        return_stats=True,
    )
    end_time = time.time()
    print(stats)
    lotus_num_calls = stats["total_llm_calls"]
    lotus_results = sorted_df.to_dict(orient="records")
    lotus_cost = stats["total_tokens"] * 0.15 / 1000000
    lotus_runtime = end_time - start_time
    method_metrics["Lotus Top K"] = {
        "runtime": lotus_runtime,
        "cost": lotus_cost
    }
    
    # Extract ranks for each document from the different methods
    baseline_ranks = {doc["id"]: doc["_rank"] for doc in order_results_baseline}
    standard_ranks = {doc["id"]: doc["_rank"] for doc in order_results}
    likert_initial_ranks = {doc["id"]: doc["_rank"] for doc in order_results_likert_initial}
    rating_ranks = {doc["id"]: doc["_rank"] for doc in order_results_rating}
    sliding_window_embedding_ranks = {doc["id"]: doc["_rank"] for doc in order_results_sliding_window_embedding}
    sliding_window_likert_ranks = {doc["id"]: doc["_rank"] for doc in order_results_sliding_window_likert}
    likert_rating_ranks = {doc["id"]: doc["_rank"] for doc in order_results_likert_rating}
    lotus_ranks = {doc["id"]: doc["_rank"] for doc in lotus_results}
    calibrated_embedding_sort_ranks = {doc["id"]: doc["_rank"] for doc in order_results_calibrated_embedding_sort}
    # Prepare lists for Kendall's Tau computation
    doc_ids = list(baseline_ranks.keys())
    
    baseline_rank_list = [baseline_ranks[doc_id] for doc_id in doc_ids]
    standard_rank_list = [standard_ranks[doc_id] for doc_id in doc_ids]
    rating_rank_list = [rating_ranks[doc_id] for doc_id in doc_ids]
    likert_initial_rank_list = [likert_initial_ranks[doc_id] for doc_id in doc_ids]
    sliding_window_embedding_rank_list = [sliding_window_embedding_ranks[doc_id] for doc_id in doc_ids]
    sliding_window_likert_rank_list = [sliding_window_likert_ranks[doc_id] for doc_id in doc_ids]
    likert_rating_rank_list = [likert_rating_ranks[doc_id] for doc_id in doc_ids]
    lotus_rank_list = [lotus_ranks.get(doc_id, 11) for doc_id in doc_ids]
    calibrated_embedding_sort_rank_list = [calibrated_embedding_sort_ranks[doc_id] for doc_id in doc_ids]
    
    # Compute Kendall's Tau correlation coefficients against baseline
    tau_standard, p_value_standard = kendalltau(baseline_rank_list, standard_rank_list)
    tau_rating, p_value_rating = kendalltau(baseline_rank_list, rating_rank_list)
    tau_likert_initial, p_value_likert_initial = kendalltau(baseline_rank_list, likert_initial_rank_list)
    tau_sliding_window_embedding, p_value_sliding_window_embedding = kendalltau(baseline_rank_list, sliding_window_embedding_rank_list)
    tau_sliding_window_likert, p_value_sliding_window_likert = kendalltau(baseline_rank_list, sliding_window_likert_rank_list)
    tau_likert_rating, p_value_likert_rating = kendalltau(baseline_rank_list, likert_rating_rank_list)
    tau_lotus, p_value_lotus = kendalltau(baseline_rank_list, lotus_rank_list)
    tau_calibrated_embedding_rating, p_value_calibrated_embedding_rating = kendalltau(baseline_rank_list, calibrated_embedding_sort_rank_list)
    
    # Store results in a list of tuples for sorting, including runtime and cost
    results = [
        ("Ours (Embedding Initial)", tau_standard, p_value_standard, method_metrics["Ours (Embedding Initial)"]["runtime"], method_metrics["Ours (Embedding Initial)"]["cost"]),
        ("Ours (Likert Initial)", tau_likert_initial, p_value_likert_initial, method_metrics["Ours (Likert Initial)"]["runtime"], method_metrics["Ours (Likert Initial)"]["cost"]),
        ("Embedding Rating", tau_rating, p_value_rating, method_metrics["Embedding Rating"]["runtime"], method_metrics["Embedding Rating"]["cost"]),
        ("Embedding Sliding Window", tau_sliding_window_embedding, p_value_sliding_window_embedding, method_metrics["Embedding Sliding Window"]["runtime"], method_metrics["Embedding Sliding Window"]["cost"]),
        ("Likert Sliding Window", tau_sliding_window_likert, p_value_sliding_window_likert, method_metrics["Likert Sliding Window"]["runtime"], method_metrics["Likert Sliding Window"]["cost"]),
        ("Likert Rating", tau_likert_rating, p_value_likert_rating, method_metrics["Likert Rating"]["runtime"], method_metrics["Likert Rating"]["cost"]),
        ("Lotus Top K", tau_lotus, p_value_lotus, method_metrics["Lotus Top K"]["runtime"], method_metrics["Lotus Top K"]["cost"]),
        ("Calibrated Embedding Rating", tau_calibrated_embedding_rating, p_value_calibrated_embedding_rating, method_metrics["Calibrated Embedding Rating"]["runtime"], method_metrics["Calibrated Embedding Rating"]["cost"])
    ]
    
    # Sort results by Kendall's tau value in descending order
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    # Create a Rich table
    table = Table(title="Medical Transcripts Ordering by Pain Level (vs Baseline)")
    
    # Add columns
    table.add_column("Method", style="cyan")
    table.add_column("Kendall's Tau", justify="right", style="green")
    table.add_column("p-value", justify="right", style="green")
    table.add_column("Runtime (s)", justify="right", style="yellow")
    table.add_column("Cost ($)", justify="right", style="yellow")
    
    # Add rows
    for method, tau, p_value, runtime, cost in sorted_results:
        table.add_row(
            method,
            f"{tau:.4f}",
            f"{p_value:.4f}",
            f"{runtime:.2f}",
            f"{cost:.4f}"
        )
    
    # Print the table
    console.print(table)
    
    # Create a table for the top pain transcripts
    top_pain_table = Table(title="Top 3 Transcripts with Highest Pain (Baseline)")
    
    # Add columns for the top transcripts
    top_pain_table.add_column("Rank", style="cyan", justify="center")
    top_pain_table.add_column("ID", style="cyan")
    top_pain_table.add_column("Transcript Preview", style="green")
    
    # Add rows with the top painful transcripts
    top_pain_transcripts = sorted(order_results_baseline, key=lambda x: x["_rank"])[:3]
    for i, doc in enumerate(top_pain_transcripts):
        # Print first 100 characters of the transcript for reference
        transcript_preview = doc["src"][:100].replace("\n", " ").strip() + "..."
        top_pain_table.add_row(
            f"{doc['_rank']}",
            doc["id"],
            transcript_preview
        )
    
    # Print the top transcripts table
    console.print(top_pain_table)
    
    # Compute NDCG@10 for each method, treating baseline's top 3 as relevant
    # Get the top 3 document IDs from baseline as ground truth
    top_baseline_doc_ids = [doc["id"] for doc in sorted(order_results_baseline, key=lambda x: x["_rank"])[:3]]
    
    # Create binary relevance arrays (1 for top 3 baseline docs, 0 for others)
    y_true = np.zeros(len(medical_data))
    for i, doc in enumerate(medical_data):
        if doc["id"] in top_baseline_doc_ids:
            y_true[i] = 1
    
    # Calculate NDCG@10 for each method
    ndcg_table = Table(title="NDCG@10 Results (Using Baseline Top 3 as Ground Truth)")
    ndcg_table.add_column("Method", style="cyan")
    ndcg_table.add_column("NDCG@10", justify="right", style="green")
    
    # Prepare arrays for each method's ranking
    method_results = [
        ("Ours (Embedding Initial)", order_results),
        ("Ours (Likert Initial)", order_results_likert_initial),
        ("Embedding Rating", order_results_rating),
        ("Embedding Sliding Window", order_results_sliding_window_embedding),
        ("Likert Sliding Window", order_results_sliding_window_likert),
        ("Likert Rating", order_results_likert_rating),
        ("Lotus Top K", lotus_results),
        ("Calibrated Embedding Rating", order_results_calibrated_embedding_sort)
    ]
    
    # Calculate NDCG for each method
    ndcg_scores = []
    for method_name, method_results_data in method_results:
        # Create predicted relevance array (higher ranks should have higher scores)
        y_pred = np.zeros(len(medical_data))
        method_ranks = {doc["id"]: doc["_rank"] for doc in method_results_data}
        
        for i, doc in enumerate(medical_data):
            # Invert rank for scoring (lower rank = higher relevance)
            y_pred[i] = len(medical_data) - method_ranks.get(doc["id"], 11)
        
        # Calculate NDCG
        ndcg_value = calculate_ndcg(y_true, y_pred, k=10)
        ndcg_scores.append((method_name, ndcg_value))
    
    # Sort NDCG scores in descending order
    sorted_ndcg = sorted(ndcg_scores, key=lambda x: x[1], reverse=True)
    
    # Add rows to table in sorted order
    for method_name, ndcg_value in sorted_ndcg:
        ndcg_table.add_row(method_name, f"{ndcg_value:.4f}")
    
    # Add baseline as reference (should be perfect)
    baseline_y_pred = np.zeros(len(medical_data))
    for i, doc in enumerate(medical_data):
        baseline_y_pred[i] = len(medical_data) - baseline_ranks[doc["id"]]
    baseline_ndcg = calculate_ndcg(y_true, baseline_y_pred, k=10)
    ndcg_table.add_row("Baseline (reference)", f"{baseline_ndcg:.4f}")
    
    # Print NDCG results
    console.print(ndcg_table)
    
    # Return the same structure as test_order_square_dataset
    return sorted_results, sorted_ndcg, lotus_num_calls

def test_order_scifact_dataset(api_wrapper):
    """
    Test ordering methods on the scifact dataset.
    For each of 5 sampled claims from queries.jsonl, we rank the corpus.jsonl
    and compute the nDCG@10 for the ranking quality.
    
    This test is computationally intensive as it ranks the entire corpus for each query.
    The batch_size parameter controls processing efficiency.
    
    Returns the same structure as other test methods:
    - sorted_results: List of methods with runtime and cost (tau values are placeholders)
    - sorted_ndcg: NDCG@10 results for each method
    - lotus_num_calls: Placeholder value (0) since Lotus is not used in this test
    """
    # Parameters to control test scope - using smaller values for faster testing
    num_queries_to_process = 100  # Reduced number for quicker execution
    batch_size = 10             # Batch size for processing documents
    
    # Define methods to test
    methods = [
        "embedding",
        "sliding_window",
        "picky"
    ]
    
    console.print(f"[bold]SciFact Dataset Test[/bold]")
    console.print(f"Testing with {num_queries_to_process} queries, using the entire corpus")
    console.print(f"Methods to test: {', '.join(methods)}")
    
    # Load the claims from queries.jsonl
    with open("workloads/scifact/queries.jsonl", "r") as f:
        queries = [json.loads(line) for line in f]
    
    # Load the test set mapping (which documents are relevant for each query)
    test_df = pd.read_csv("workloads/scifact/test.tsv", sep="\t")
    
    # Create a dictionary mapping query IDs to relevant document IDs
    query_to_relevant_docs = {}
    for _, row in test_df.iterrows():
        query_id = str(row["query-id"])
        corpus_id = str(row["corpus-id"])
        score = int(row["score"])
        
        if query_id not in query_to_relevant_docs:
            query_to_relevant_docs[query_id] = {}
        
        query_to_relevant_docs[query_id][corpus_id] = score
    
    # Filter queries to only those that have relevant documents in the test set
    valid_queries = [q for q in queries if q["_id"] in query_to_relevant_docs]
    console.print(f"Found {len(valid_queries)} valid queries with relevant documents")
    
    # Sample queries (or all if there are fewer than requested)
    sample_size = min(num_queries_to_process, len(valid_queries))
    sampled_queries = random.sample(valid_queries, sample_size)
    
    # Load the entire corpus
    console.print(f"Loading the entire corpus...")
    corpus_docs = []
    corpus_id_set = set()  # To track unique IDs
    
    with open("workloads/scifact/corpus.jsonl", "r") as f:
        for line in f:
            try:
                doc = json.loads(line)
                doc_id = doc["_id"]
                
                if doc_id not in corpus_id_set:
                    corpus_docs.append(doc)
                    corpus_id_set.add(doc_id)
                    
            except json.JSONDecodeError:
                continue  # Skip malformed lines
    
    console.print(f"Loaded {len(corpus_docs)} corpus documents")
    
    # Create a dictionary to map document IDs to corpus documents
    corpus_dict = {doc["_id"]: doc for doc in corpus_docs}
    
    # Initialize metrics tracking
    method_metrics = {
        "Embedding Rating": {"runtime": 0.0, "cost": 0.0, "count": 0},
        "Sliding Window": {"runtime": 0.0, "cost": 0.0, "count": 0},
        "Picky": {"runtime": 0.0, "cost": 0.0, "count": 0}
    }
    
    # Track NDCG scores
    ndcg_results = {
        "Embedding Rating": {"ndcg_sum": 0.0, "count": 0},
        "Sliding Window": {"ndcg_sum": 0.0, "count": 0},
        "Picky": {"ndcg_sum": 0.0, "count": 0}
    }
    
    # Process each query
    for idx, query in enumerate(sampled_queries):
        try:
            query_id = query["_id"]
            query_text = query["text"]
            
            console.print(f"\n[bold cyan]Query {idx+1}/{len(sampled_queries)}[/bold cyan]: {query_text}")
            
            # Get the set of relevant document IDs for this query
            relevant_doc_ids = set(query_to_relevant_docs.get(query_id, {}).keys())
            console.print(f"Query has {len(relevant_doc_ids)} relevant documents")
            
            # Verify all relevant documents are in our corpus
            found_relevant = sum(1 for doc_id in relevant_doc_ids if doc_id in corpus_dict)
            console.print(f"Found {found_relevant}/{len(relevant_doc_ids)} relevant documents in corpus")
            
            # Skip if we didn't find any relevant documents for this query
            if found_relevant == 0:
                console.print("[yellow]Skipping query - no relevant documents found in corpus[/yellow]")
                continue
                
            # Create the order operation for this query
            order_config = {
                "name": f"order_by_relevance_to_claim_{query_id}",
                "type": "order",
                "batch_size": batch_size,
                "prompt": f"""
                    Order these scientific abstracts based on how relevant they are to the following claim:
                    
                    "{query_text}"
                """,
                "input_keys": ["title", "text"],
                "direction": "desc",  # Most relevant first
                "verbose": True,
                "rerank_call_budget": 10,
                "k": 200,
            }
            
            order_operation = RankOperation(
                api_wrapper,
                order_config,
                default_model="gemini/gemini-2.0-flash",
                max_threads=64,
            )
            
            # Create relevance scores array (1 for relevant, 0 for non-relevant)
            y_true = np.zeros(len(corpus_docs))
            
            for i, doc in enumerate(corpus_docs):
                doc_id = doc["_id"]
                # Set relevance score
                if doc_id in relevant_doc_ids:
                    y_true[i] = query_to_relevant_docs[query_id][doc_id]
            
            # Dictionary to store results for this query
            query_method_results = {}
            
            # Execute each method
            if "embedding" in methods:
                # Apply embedding rating method to all corpus documents
                console.print(f"[green]Running Embedding Rating method on all {len(corpus_docs)} documents...[/green]")
                start_time = time.time()
                order_results_embedding, order_cost_embedding = order_operation._execute_rating_embedding_qurk(corpus_docs)
                embedding_runtime = time.time() - start_time
                
                # Extract ranks
                embedding_ranks = {doc["_id"]: doc["_rank"] for doc in order_results_embedding}
                
                # Store results
                query_method_results["Embedding Rating"] = {
                    "results": order_results_embedding, 
                    "ranks": embedding_ranks,
                    "runtime": embedding_runtime, 
                    "cost": order_cost_embedding
                }
                
                # Update aggregated metrics
                method_metrics["Embedding Rating"]["runtime"] += embedding_runtime
                method_metrics["Embedding Rating"]["cost"] += order_cost_embedding
                method_metrics["Embedding Rating"]["count"] += 1
            
            if "sliding_window" in methods:
                # Apply sliding window method with embedding
                console.print(f"[green]Running Sliding Window method on all {len(corpus_docs)} documents...[/green]")
                start_time = time.time()
                order_results_sliding, order_cost_sliding = order_operation._execute_sliding_window_qurk(corpus_docs, initial_ordering_method="embedding", k=10)
                sliding_runtime = time.time() - start_time
                
                # Extract ranks
                sliding_ranks = {doc["_id"]: doc["_rank"] for doc in order_results_sliding}
                
                # Store results
                query_method_results["Sliding Window"] = {
                    "results": order_results_sliding, 
                    "ranks": sliding_ranks,
                    "runtime": sliding_runtime, 
                    "cost": order_cost_sliding
                }
                
                # Update aggregated metrics
                method_metrics["Sliding Window"]["runtime"] += sliding_runtime
                method_metrics["Sliding Window"]["cost"] += order_cost_sliding
                method_metrics["Sliding Window"]["count"] += 1
            
            if "picky" in methods:
                # Apply picky method
                console.print(f"[green]Running Picky method on all {len(corpus_docs)} documents...[/green]")
                start_time = time.time()
                order_results_picky, order_cost_picky = order_operation.execute(corpus_docs)
                picky_runtime = time.time() - start_time
                
                # Extract ranks
                picky_ranks = {doc["_id"]: doc["_rank"] for doc in order_results_picky}
                
                # Store results
                query_method_results["Picky"] = {
                    "results": order_results_picky, 
                    "ranks": picky_ranks,
                    "runtime": picky_runtime, 
                    "cost": order_cost_picky
                }
                
                # Update aggregated metrics
                method_metrics["Picky"]["runtime"] += picky_runtime
                method_metrics["Picky"]["cost"] += order_cost_picky
                method_metrics["Picky"]["count"] += 1
            
            # Calculate NDCG@10 for each method
            for method_name, method_data in query_method_results.items():
                method_ranks = method_data["ranks"]
                
                # Calculate NDCG@10
                y_score = np.zeros(len(corpus_docs))
                for i, doc in enumerate(corpus_docs):
                    doc_id = doc["_id"]
                    if doc_id in method_ranks:
                        # Set predicted relevance (inverse of rank since smaller rank = more relevant)
                        y_score[i] = len(corpus_docs) - method_ranks[doc_id]
                
                ndcg_value = calculate_ndcg(y_true, y_score, k=10)
                
                # Update NDCG aggregation
                ndcg_results[method_name]["ndcg_sum"] += ndcg_value
                ndcg_results[method_name]["count"] += 1
            
            # Create a table for this query's results (relevant documents only)
            relevant_docs_table = Table(title=f"Rankings for relevant documents (Query {idx+1})")
            relevant_docs_table.add_column("Document", style="cyan")
            
            # Add columns for each method
            for method_name in query_method_results:
                relevant_docs_table.add_column(f"{method_name} Rank", justify="right", style="green")
            
            # Add rows with the relevant documents and their ranks
            for doc_id in relevant_doc_ids:
                if doc_id in corpus_dict:
                    doc = corpus_dict[doc_id]
                    title = doc["title"][:50] + "..." if len(doc["title"]) > 50 else doc["title"]
                    
                    row_data = [title]
                    for method_name in query_method_results:
                        method_ranks = query_method_results[method_name]["ranks"]
                        rank = method_ranks.get(doc_id, len(corpus_docs) + 1)
                        row_data.append(f"{rank}/{len(corpus_docs)}")
                    
                    relevant_docs_table.add_row(*row_data)
            
            console.print(relevant_docs_table)
            
            # Print NDCG summary for this query
            ndcg_table = Table(title=f"NDCG@10 Results for Query {idx+1}")
            ndcg_table.add_column("Method", style="cyan")
            ndcg_table.add_column("NDCG@10", justify="right", style="green")
            
            for method_name in query_method_results:
                # Calculate NDCG again for display
                method_ranks = query_method_results[method_name]["ranks"]
                y_score = np.zeros(len(corpus_docs))
                for i, doc in enumerate(corpus_docs):
                    doc_id = doc["_id"]
                    if doc_id in method_ranks:
                        # Set predicted relevance (inverse of rank since smaller rank = more relevant)
                        y_score[i] = len(corpus_docs) - method_ranks[doc_id]
                
                ndcg_value = calculate_ndcg(y_true, y_score, k=10)
                ndcg_table.add_row(method_name, f"{ndcg_value:.4f}")
            
            console.print(ndcg_table)
            
        except Exception as e:
            console.print(f"[bold red]Error processing query {idx+1}[/bold red]: {str(e)}")
            continue  # Skip to next query
    
    # Calculate averages for the final results
    sorted_results = []
    for method_name, data in method_metrics.items():
        if data["count"] > 0:
            avg_runtime = data["runtime"] / data["count"]
            avg_cost = data["cost"] / data["count"]
            
            # Use placeholder values for tau and p-value to maintain compatibility
            # with return structure of other test functions
            placeholder_tau = 1.0 if method_name == list(method_metrics.keys())[0] else 0.5
            placeholder_p_value = 0.0
            
            sorted_results.append((
                method_name, 
                placeholder_tau,
                placeholder_p_value, 
                avg_runtime, 
                avg_cost
            ))
    
    # Sort by runtime (since we're not using tau)
    sorted_results = sorted(sorted_results, key=lambda x: x[3])
    
    # Calculate average NDCG for the final results
    sorted_ndcg = []
    for method_name, data in ndcg_results.items():
        if data["count"] > 0:
            avg_ndcg = data["ndcg_sum"] / data["count"]
            sorted_ndcg.append((method_name, avg_ndcg))
    
    # Sort by NDCG (descending)
    sorted_ndcg = sorted(sorted_ndcg, key=lambda x: x[1], reverse=True)
    
    # Create a table for average metrics
    table = Table(title=f"SciFact Dataset Ordering Results - {len(sampled_queries)} Queries")
    
    # Add columns
    table.add_column("Method", style="cyan")
    table.add_column("Avg Runtime (s)", justify="right", style="yellow")
    table.add_column("Avg Cost ($)", justify="right", style="yellow")
    
    # Add rows (skip tau and p-value as they're just placeholders)
    for method, _, _, runtime, cost in sorted_results:
        table.add_row(
            method,
            f"{runtime:.2f}",
            f"{cost:.4f}"
        )
    
    # Print the table
    console.print(table)
    
    # Create a table for average NDCG results
    ndcg_table = Table(title=f"AVERAGE SciFact NDCG@10 Results ({len(sampled_queries)} Queries)")
    ndcg_table.add_column("Method", style="cyan")
    ndcg_table.add_column("Avg NDCG@10", justify="right", style="green")
    
    # Add rows
    for method_name, avg_ndcg in sorted_ndcg:
        ndcg_table.add_row(method_name, f"{avg_ndcg:.4f}")
    
    # Print NDCG results
    console.print(ndcg_table)
    
    # Return the same structure as other test methods, with placeholder value for lotus calls
    return sorted_results, sorted_ndcg, 0  # Using 0 as placeholder for lotus_num_calls

def test_order_number_words(api_wrapper):
    """
    Test sorting a list of number words ("one", "two", "three", ...,"five hundred")
    in ascending numerical order using different ordering methods and compare their accuracy.
    """
    # Generate a dataset of number words from "one" to "five hundred"
    def number_to_words(num):
        """Convert a number to its word representation."""
        if num <= 0:
            return "zero"
        
        ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", 
                "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", 
                "seventeen", "eighteen", "nineteen"]
        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        
        if num < 20:
            return ones[num]
        elif num < 100:
            return tens[num // 10] + ("-" + ones[num % 10] if num % 10 != 0 else "")
        elif num < 1000:
            if num % 100 == 0:
                return ones[num // 100] + " hundred"
            else:
                return ones[num // 100] + " hundred " + number_to_words(num % 100)
        elif num == 1000:
            return "one thousand"
        
        return ""  # For numbers outside our range
    
    # Generate number words from "one" to "five hundred"
    number_words = [number_to_words(i) for i in range(1, 501)]
    
    # Create ground truth rankings (natural numerical order)
    ground_truth_ranks = {word: i+1 for i, word in enumerate(number_words)}
    
    # Shuffle the number words to create a randomized dataset
    random.seed(42)  # For reproducibility
    shuffled_number_words = number_words.copy()
    random.shuffle(shuffled_number_words)
    
    # Create data objects similar to other tests
    number_data = []
    for i, word in enumerate(shuffled_number_words):
        number_data.append({
            "id": f"number_{i}",
            "text": word,
            "description": f"The number word: {word}",
            "original_rank": ground_truth_ranks[word]  # Store original rank for validation
        })
    
    # Create order operation for numerical ordering
    order_config = {
        "name": "order_by_numerical_value",
        "type": "order",
        "batch_size": 100,
        "prompt": """
            Order these number words in ascending numerical order (from lowest to highest).
        """,
        "input_keys": ["text"],
        "direction": "asc",  # Ascending order (lowest first)
        "verbose": True,
        "rerank_call_budget": 10,
        # "bypass_cache": True
    }
    
    order_operation = RankOperation(
        api_wrapper,
        order_config,
        default_model="gemini/gemini-2.0-flash",
        max_threads=64,
    )
    
    order_config_copy = order_config.copy()
    order_config_copy["initial_ordering_method"] = "likert"
    order_operation_likert = RankOperation(
        api_wrapper,
        order_config_copy,
        default_model="gemini/gemini-2.0-flash",
        max_threads=64,
    )
    # Track execution time and cost for each method
    method_metrics = {}
    
    # Execute picky method
    console.print("[bold blue]Sorting number words with picky method[/bold blue]")
    start_time = time.time()
    order_results_picky, cost_picky = order_operation.execute(
        number_data
    )
    method_metrics["Picky (Embedding)"] = {
        "runtime": time.time() - start_time,
        "cost": cost_picky
    }
    
    # Execute picky with likert initial ordering
    console.print("[bold blue]Sorting number words with picky method (likert initial ordering)[/bold blue]")
    start_time = time.time()
    order_results_picky_likert, cost_picky_likert = order_operation_likert.execute(
        number_data
    )
    method_metrics["Picky (Likert)"] = {
        "runtime": time.time() - start_time,
        "cost": cost_picky_likert
    }
    
    # Execute baseline comparison method
    console.print("[bold blue]Sorting number words with baseline comparison method[/bold blue]")
    start_time = time.time()
    order_results_baseline, cost_baseline = order_operation._execute_comparison_qurk(number_data)
    method_metrics["Baseline Comparison"] = {
        "runtime": time.time() - start_time,
        "cost": cost_baseline
    }
    
    # Execute rating embedding method
    console.print("[bold blue]Sorting number words with embedding rating method[/bold blue]")
    start_time = time.time()
    order_results_embedding, cost_embedding = order_operation._execute_rating_embedding_qurk(number_data)
    method_metrics["Embedding Rating"] = {
        "runtime": time.time() - start_time,
        "cost": cost_embedding
    }
    
    # Execute calibrated embedding sort
    console.print("[bold blue]Sorting number words with calibrated embedding sort method[/bold blue]")
    start_time = time.time()
    order_results_calibrated_embedding, cost_calibrated_embedding = order_operation._execute_calibrated_embedding_sort(number_data)
    method_metrics["Calibrated Embedding Sort"] = {
        "runtime": time.time() - start_time,
        "cost": cost_calibrated_embedding
    }
    # Execute sliding window with embedding
    console.print("[bold blue]Sorting number words with sliding window (embedding) method[/bold blue]")
    start_time = time.time()
    order_results_sliding_embedding, cost_sliding_embedding = order_operation._execute_sliding_window_qurk(
        number_data, 
        initial_ordering_method="embedding"
    )
    method_metrics["Embedding Sliding Window"] = {
        "runtime": time.time() - start_time,
        "cost": cost_sliding_embedding
    }
    
    # Execute sliding window with likert
    console.print("[bold blue]Sorting number words with sliding window (likert) method[/bold blue]")
    start_time = time.time()
    order_results_sliding_likert, cost_sliding_likert = order_operation._execute_sliding_window_qurk(
        number_data, 
        initial_ordering_method="likert"
    )
    method_metrics["Likert Sliding Window"] = {
        "runtime": time.time() - start_time,
        "cost": cost_sliding_likert
    }
    
    # Execute likert rating method
    console.print("[bold blue]Sorting number words with likert rating method[/bold blue]")
    start_time = time.time()
    order_results_likert, cost_likert = order_operation._execute_likert_rating_qurk(number_data)
    method_metrics["Likert Rating"] = {
        "runtime": time.time() - start_time,
        "cost": cost_likert
    }
    
    # Execute lotus top k
    console.print("[bold blue]Sorting number words with lotus top k method[/bold blue]")
    start_time = time.time()
    df = pd.DataFrame(number_data)
    sorted_df, stats = df.sem_topk(
        "What number is the smallest? Here is the number: {text}",
        K=10,
        return_stats=True,
    )
    print(stats)
    end_time = time.time()
    lotus_results = sorted_df.to_dict(orient="records")
    lotus_num_calls = stats["total_llm_calls"]
    lotus_cost = stats["total_tokens"] * 0.15 / 1000000
    lotus_runtime = end_time - start_time
    method_metrics["Lotus Top K"] = {
        "runtime": lotus_runtime,
        "cost": lotus_cost
    }
    
    # Extract ranks for each document from different methods
    picky_embedding_ranks = {doc["id"]: doc["_rank"] for doc in order_results_picky}
    picky_likert_ranks = {doc["id"]: doc["_rank"] for doc in order_results_picky_likert}
    baseline_ranks = {doc["id"]: doc["_rank"] for doc in order_results_baseline}
    embedding_ranks = {doc["id"]: doc["_rank"] for doc in order_results_embedding}
    sliding_embedding_ranks = {doc["id"]: doc["_rank"] for doc in order_results_sliding_embedding}
    sliding_likert_ranks = {doc["id"]: doc["_rank"] for doc in order_results_sliding_likert}
    likert_ranks = {doc["id"]: doc["_rank"] for doc in order_results_likert}
    lotus_ranks = {doc["id"]: doc["_rank"] for doc in lotus_results}
    calibrated_embedding_ranks = {doc["id"]: doc["_rank"] for doc in order_results_calibrated_embedding}
    
    # Create mapping for ground truth ranks by ID
    sorted_indices = sorted(range(len(number_words)), key=lambda i: number_words[i])
    ground_truth_ranks_by_id = {f"number_{i}": sorted_indices.index(i) + 1 for i in range(len(number_words))}
    
    # Prepare lists for Kendall's Tau computation
    doc_ids = list(picky_embedding_ranks.keys())
    
    ground_truth_rank_list = [ground_truth_ranks_by_id[doc_id] for doc_id in doc_ids]
    picky_embedding_rank_list = [picky_embedding_ranks[doc_id] for doc_id in doc_ids]
    picky_likert_rank_list = [picky_likert_ranks[doc_id] for doc_id in doc_ids]
    baseline_rank_list = [baseline_ranks[doc_id] for doc_id in doc_ids]
    embedding_rank_list = [embedding_ranks[doc_id] for doc_id in doc_ids]
    sliding_embedding_rank_list = [sliding_embedding_ranks[doc_id] for doc_id in doc_ids]
    sliding_likert_rank_list = [sliding_likert_ranks[doc_id] for doc_id in doc_ids]
    likert_rating_rank_list = [likert_ranks[doc_id] for doc_id in doc_ids]
    lotus_rank_list = [lotus_ranks.get(doc_id, len(number_data) + 1) for doc_id in doc_ids]
    calibrated_embedding_rank_list = [calibrated_embedding_ranks[doc_id] for doc_id in doc_ids]
    
    # Compute Kendall's Tau correlation coefficients against ground truth
    tau_picky_embedding, p_value_picky_embedding = kendalltau(ground_truth_rank_list, picky_embedding_rank_list)
    tau_picky_likert, p_value_picky_likert = kendalltau(ground_truth_rank_list, picky_likert_rank_list)
    tau_baseline, p_value_baseline = kendalltau(ground_truth_rank_list, baseline_rank_list)
    tau_embedding, p_value_embedding = kendalltau(ground_truth_rank_list, embedding_rank_list)
    tau_sliding_embedding, p_value_sliding_embedding = kendalltau(ground_truth_rank_list, sliding_embedding_rank_list)
    tau_sliding_likert, p_value_sliding_likert = kendalltau(ground_truth_rank_list, sliding_likert_rank_list)
    tau_likert_rating, p_value_likert_rating = kendalltau(ground_truth_rank_list, likert_rating_rank_list)
    tau_lotus, p_value_lotus = kendalltau(ground_truth_rank_list, lotus_rank_list)
    tau_calibrated_embedding, p_value_calibrated_embedding = kendalltau(ground_truth_rank_list, calibrated_embedding_rank_list)
    
    # Store results in a list of tuples for sorting
    results = [
        ("Picky (Embedding)", tau_picky_embedding, p_value_picky_embedding, method_metrics["Picky (Embedding)"]["runtime"], method_metrics["Picky (Embedding)"]["cost"]),
        ("Picky (Likert)", tau_picky_likert, p_value_picky_likert, method_metrics["Picky (Likert)"]["runtime"], method_metrics["Picky (Likert)"]["cost"]),
        ("Baseline Comparison", tau_baseline, p_value_baseline, method_metrics["Baseline Comparison"]["runtime"], method_metrics["Baseline Comparison"]["cost"]),
        ("Embedding Rating", tau_embedding, p_value_embedding, method_metrics["Embedding Rating"]["runtime"], method_metrics["Embedding Rating"]["cost"]),
        ("Embedding Sliding Window", tau_sliding_embedding, p_value_sliding_embedding, method_metrics["Embedding Sliding Window"]["runtime"], method_metrics["Embedding Sliding Window"]["cost"]),
        ("Likert Sliding Window", tau_sliding_likert, p_value_sliding_likert, method_metrics["Likert Sliding Window"]["runtime"], method_metrics["Likert Sliding Window"]["cost"]),
        ("Likert Rating", tau_likert_rating, p_value_likert_rating, method_metrics["Likert Rating"]["runtime"], method_metrics["Likert Rating"]["cost"]),
        ("Lotus Top K", tau_lotus, p_value_lotus, method_metrics["Lotus Top K"]["runtime"], method_metrics["Lotus Top K"]["cost"]),
        ("Calibrated Embedding Sort", tau_calibrated_embedding, p_value_calibrated_embedding, method_metrics["Calibrated Embedding Sort"]["runtime"], method_metrics["Calibrated Embedding Sort"]["cost"])
    ]
    
    # Sort results by Kendall's tau value in descending order
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    # Create a Rich table for method comparison
    table = Table(title="Number Words Ordering Results (vs Ground Truth)")
    
    # Add columns
    table.add_column("Method", style="cyan")
    table.add_column("Kendall's Tau", justify="right", style="green")
    table.add_column("p-value", justify="right", style="green")
    table.add_column("Runtime (s)", justify="right", style="yellow")
    table.add_column("Cost ($)", justify="right", style="yellow")
    
    # Add rows
    for method, tau, p_value, runtime, cost in sorted_results:
        table.add_row(
            method,
            f"{tau:.4f}",
            f"{p_value:.4f}",
            f"{runtime:.2f}",
            f"{cost:.4f}"
        )
    
    # Print the table
    console.print(table)
    
    # Calculate NDCG@10 for each method using ground truth
    # Create ground truth relevance array (inversely related to rank, higher = more relevant)
    y_true = np.zeros(len(number_data))
    num_numbers = len(number_words)
    
    for i, doc in enumerate(number_data):
        # Use inverse of rank so that highest relevance = lowest number (for ascending order)
        y_true[i] = num_numbers - ground_truth_ranks_by_id[doc["id"]] + 1
    
    # Calculate NDCG@10 for each method
    ndcg_table = Table(title="NDCG@10 Results (vs Ground Truth)")
    ndcg_table.add_column("Method", style="cyan")
    ndcg_table.add_column("NDCG@10", justify="right", style="green")
    
    # Prepare arrays for each method's ranking
    method_results = [
        ("Picky (Embedding)", order_results_picky, picky_embedding_ranks),
        ("Picky (Likert)", order_results_picky_likert, picky_likert_ranks),
        ("Baseline Comparison", order_results_baseline, baseline_ranks),
        ("Embedding Rating", order_results_embedding, embedding_ranks),
        ("Embedding Sliding Window", order_results_sliding_embedding, sliding_embedding_ranks),
        ("Likert Sliding Window", order_results_sliding_likert, sliding_likert_ranks),
        ("Likert Rating", order_results_likert, likert_ranks),
        ("Lotus Top K", lotus_results, lotus_ranks),
        ("Calibrated Embedding Sort", order_results_calibrated_embedding, calibrated_embedding_ranks)
    ]
    
    # Calculate NDCG for each method
    ndcg_scores = []
    for method_name, _, method_ranks in method_results:
        # Create predicted relevance array (higher ranks should have higher scores)
        y_pred = np.zeros(len(number_data))
        
        for i, doc in enumerate(number_data):
            # Invert rank for scoring (lower rank = higher relevance)
            y_pred[i] = num_numbers - method_ranks.get(doc["id"], num_numbers + 1)
        
        # Calculate NDCG
        ndcg_value = calculate_ndcg(y_true, y_pred, k=10)
        ndcg_scores.append((method_name, ndcg_value))
        ndcg_table.add_row(method_name, f"{ndcg_value:.4f}")
    
    # Sort NDCG scores and recreate table with sorted results
    sorted_ndcg = sorted(ndcg_scores, key=lambda x: x[1], reverse=True)
    sorted_ndcg_table = Table(title="NDCG@10 Results (vs Ground Truth) - Sorted")
    sorted_ndcg_table.add_column("Method", style="cyan")
    sorted_ndcg_table.add_column("NDCG@10", justify="right", style="green")
    
    for method_name, ndcg_value in sorted_ndcg:
        sorted_ndcg_table.add_row(method_name, f"{ndcg_value:.4f}")
    
    # Print NDCG results
    console.print(sorted_ndcg_table)
    
    # Also show a sample of the sorted results from the best method
    best_method = sorted_results[0][0]
    best_results = None
    
    if best_method == "Picky (Embedding)":
        best_results = order_results_picky
    elif best_method == "Picky (Likert)":
        best_results = order_results_picky_likert
    elif best_method == "Baseline Comparison":
        best_results = order_results_baseline
    elif best_method == "Embedding Rating":
        best_results = order_results_embedding
    elif best_method == "Embedding Sliding Window":
        best_results = order_results_sliding_embedding
    elif best_method == "Likert Sliding Window":
        best_results = order_results_sliding_likert
    elif best_method == "Likert Rating":
        best_results = order_results_likert
    elif best_method == "Lotus Top K":
        best_results = lotus_results
    elif best_method == "Calibrated Embedding Sort":
        best_results = order_results_calibrated_embedding
    
    # Create a table to show a portion of the best results
    if best_results:
        best_sorted_words = [item["text"] for item in best_results]
        
        result_table = Table(title=f"Sorted Number Words (Sample from {best_method})")
        result_table.add_column("Sorted Rank", style="cyan", justify="right")
        result_table.add_column("Number Word", style="green")
        result_table.add_column("Correct Rank", style="yellow", justify="right")
        
        # Show first 10, middle 5, and last 10 to verify ordering
        display_count = min(5, len(best_sorted_words))
        sample_indices = list(range(display_count)) + list(range(len(best_sorted_words) // 2 - display_count // 2, len(best_sorted_words) // 2 + display_count // 2)) + list(range(len(best_sorted_words) - display_count, len(best_sorted_words)))
        
        for idx in sample_indices:
            if idx < len(best_sorted_words):
                word = best_sorted_words[idx]
                correct_rank = ground_truth_ranks[word]
                result_table.add_row(
                    str(idx + 1),
                    word,
                    str(correct_rank)
                )
        
        console.print(result_table)
    
    return sorted_results, sorted_ndcg, lotus_num_calls

def test_order_synthetic_abstracts(api_wrapper, num_abstracts=200):
    """
    Test sorting a list of synthetic paper abstracts in ascending order by the accuracy values 
    they report using different ordering methods and compare their performance.
    
    This test:
    1. Generates 200 random accuracy values between 0 and 1
    2. Creates synthetic paper abstracts using a MapOperation with gpt-4o-mini
    3. Applies various ordering methods to sort the abstracts by accuracy
    4. Compares methods using Kendall's Tau and NDCG metrics
    """
    # Generate 200 random accuracy values between 0 and 1
    random.seed(42)  # For reproducibility
    accuracy_values = [round(random.random(), 2) for _ in range(num_abstracts)]
    
    # Create dataset for the map operation
    accuracy_data = []
    for i, accuracy in enumerate(accuracy_values):
        accuracy_data.append({
            "id": f"paper_{i}",
            "accuracy": accuracy,
            "dataset_name": "HellaSwag"
        })
    
    # Create map operation to generate synthetic abstracts
    map_config = {
        "name": "generate_paper_abstracts",
        "type": "map",
        "prompt": """
            You are a research scientist writing abstracts for machine learning papers. 
            
            Please generate a realistic, academic-sounding abstract for a research paper that reports 
            the following specific accuracy value: {{ input.accuracy }} 
            on the {{ input.dataset_name }} dataset.
            
            The abstract should:
            1. Sound like a real machine learning paper abstract
            2. Mention the model name, dataset name, and the exact accuracy value 
            3. Be 4-6 sentences long
            4. Include technical terminology appropriate for an ML research paper
            5. Maintain a formal, academic tone
        """,
        "output": {
            "schema": {
                "abstract": "string"
            }
        },
        "model": "gpt-4o-mini",
        "validate": [
            "len(output['abstract'].split('.')) >= 2",  # At least 2 sentences
        ],
        "num_retries_on_validate_failure": 2,
    }
    
    # Create MapOperation instance
    map_operation = MapOperation(
        api_wrapper,
        map_config,
        default_model="gpt-4o-mini",
        max_threads=64
    )
    
    # Execute map operation to generate synthetic abstracts
    console.print("[bold blue]Generating synthetic paper abstracts with GPT-4o-mini[/bold blue]")
    start_time = time.time()
    abstract_results, map_cost = map_operation.execute(accuracy_data)
    map_runtime = time.time() - start_time
    console.print(f"[green]Generated {len(abstract_results)} abstracts in {map_runtime:.2f} seconds at a cost of ${map_cost:.4f}[/green]")
    
    # Sample a few abstracts to display
    console.print("\n[bold cyan]Sample Synthetic Abstracts:[/bold cyan]")
    sample_indices = random.sample(range(len(abstract_results)), min(3, len(abstract_results)))
    for idx in sample_indices:
        console.print(f"[yellow]Abstract for paper with accuracy {abstract_results[idx]['accuracy']}:[/yellow]")
        console.print(abstract_results[idx]['abstract'])
        console.print("")
    
    # Create order operation for sorting abstracts by accuracy
    order_config = {
        "name": "order_by_accuracy",
        "type": "order",
        "batch_size": 10,
        "prompt": """
            Order these research paper abstracts based on the accuracy value they report, from lowest accuracy to highest.
        """,
        "input_keys": ["abstract"],
        "direction": "asc",  # Ascending order (lowest first)
        "verbose": True,
        "rerank_call_budget": 10,
        "bypass_cache": True
    }
    
    order_operation = RankOperation(
        api_wrapper,
        order_config,
        default_model="gemini/gemini-2.0-flash",
        max_threads=64,
    )
    
    order_config_likert = order_config.copy()
    order_config_likert["initial_ordering_method"] = "likert"
    order_operation_likert = RankOperation(
        api_wrapper,
        order_config_likert,
        default_model="gemini/gemini-2.0-flash",
        max_threads=64,
    )
    
    # Track execution time and cost for each method
    method_metrics = {}
    
    # Execute picky method
    console.print("[bold blue]Sorting abstracts with picky method (embedding initial)[/bold blue]")
    start_time = time.time()
    order_results_picky, cost_picky = order_operation.execute(abstract_results)
    method_metrics["Picky (Embedding)"] = {
        "runtime": time.time() - start_time,
        "cost": cost_picky
    }
    
    # Execute picky with likert initial ordering
    console.print("[bold blue]Sorting abstracts with picky method (likert initial ordering)[/bold blue]")
    start_time = time.time()
    order_results_picky_likert, cost_picky_likert = order_operation_likert.execute(abstract_results)
    method_metrics["Picky (Likert)"] = {
        "runtime": time.time() - start_time,
        "cost": cost_picky_likert
    }
    
    # Execute calibrated embedding sort
    console.print("[bold blue]Sorting abstracts with calibrated embedding sort method[/bold blue]")
    start_time = time.time()
    order_results_calibrated_embedding, cost_calibrated_embedding = order_operation._execute_calibrated_embedding_sort(abstract_results)
    method_metrics["Calibrated Embedding Sort"] = {
        "runtime": time.time() - start_time,
        "cost": cost_calibrated_embedding
    }
    
    # Execute baseline comparison method
    console.print("[bold blue]Sorting abstracts with baseline comparison method[/bold blue]")
    start_time = time.time()
    order_results_baseline, cost_baseline = order_operation._execute_comparison_qurk(abstract_results)
    method_metrics["Baseline Comparison"] = {
        "runtime": time.time() - start_time,
        "cost": cost_baseline
    }
    
    # Execute rating embedding method
    console.print("[bold blue]Sorting abstracts with embedding rating method[/bold blue]")
    start_time = time.time()
    order_results_embedding, cost_embedding = order_operation._execute_rating_embedding_qurk(abstract_results)
    method_metrics["Embedding Rating"] = {
        "runtime": time.time() - start_time,
        "cost": cost_embedding
    }
    
    # Execute sliding window with embedding
    console.print("[bold blue]Sorting abstracts with sliding window (embedding) method[/bold blue]")
    start_time = time.time()
    order_results_sliding_embedding, cost_sliding_embedding = order_operation._execute_sliding_window_qurk(
        abstract_results, 
        initial_ordering_method="embedding"
    )
    method_metrics["Embedding Sliding Window"] = {
        "runtime": time.time() - start_time,
        "cost": cost_sliding_embedding
    }
    
    # Execute sliding window with likert
    console.print("[bold blue]Sorting abstracts with sliding window (likert) method[/bold blue]")
    start_time = time.time()
    order_results_sliding_likert, cost_sliding_likert = order_operation._execute_sliding_window_qurk(
        abstract_results, 
        initial_ordering_method="likert"
    )
    method_metrics["Likert Sliding Window"] = {
        "runtime": time.time() - start_time,
        "cost": cost_sliding_likert
    }
    
    # Execute likert rating method
    console.print("[bold blue]Sorting abstracts with likert rating method[/bold blue]")
    start_time = time.time()
    order_results_likert, cost_likert = order_operation._execute_likert_rating_qurk(abstract_results)
    method_metrics["Likert Rating"] = {
        "runtime": time.time() - start_time,
        "cost": cost_likert
    }
    
    # Execute lotus top k
    console.print("[bold blue]Sorting abstracts with lotus top k method[/bold blue]")
    start_time = time.time()
    df = pd.DataFrame(abstract_results)
    sorted_df, stats = df.sem_topk(
        "Which abstract reports the lowest accuracy value? Here is the abstract: {abstract}",
        K=10,
        return_stats=True,
    )
    print(stats)
    end_time = time.time()
    lotus_results = sorted_df.to_dict(orient="records")
    lotus_num_calls = stats["total_llm_calls"]
    lotus_cost = stats["total_tokens"] * 0.15 / 1000000
    lotus_runtime = end_time - start_time
    method_metrics["Lotus Top K"] = {
        "runtime": lotus_runtime,
        "cost": lotus_cost
    }
    
    # Extract ranks for each document from different methods
    picky_embedding_ranks = {doc["id"]: doc["_rank"] for doc in order_results_picky}
    picky_likert_ranks = {doc["id"]: doc["_rank"] for doc in order_results_picky_likert}
    baseline_ranks = {doc["id"]: doc["_rank"] for doc in order_results_baseline}
    embedding_ranks = {doc["id"]: doc["_rank"] for doc in order_results_embedding}
    sliding_embedding_ranks = {doc["id"]: doc["_rank"] for doc in order_results_sliding_embedding}
    sliding_likert_ranks = {doc["id"]: doc["_rank"] for doc in order_results_sliding_likert}
    likert_ranks = {doc["id"]: doc["_rank"] for doc in order_results_likert}
    lotus_ranks = {doc["id"]: doc["_rank"] for doc in lotus_results}
    calibrated_embedding_ranks = {doc["id"]: doc["_rank"] for doc in order_results_calibrated_embedding}
    
    # Create ground truth rankings based on original accuracy values
    # For ascending order (lowest to highest), smaller rank goes to smaller accuracy
    sorted_indices = sorted(range(len(accuracy_values)), key=lambda i: accuracy_values[i])
    ground_truth_ranks = {f"paper_{i}": sorted_indices.index(i) + 1 for i in range(len(accuracy_values))}
    
    # Prepare lists for Kendall's Tau computation
    doc_ids = list(picky_embedding_ranks.keys())
    
    ground_truth_rank_list = [ground_truth_ranks[doc_id] for doc_id in doc_ids]
    picky_embedding_rank_list = [picky_embedding_ranks[doc_id] for doc_id in doc_ids]
    picky_likert_rank_list = [picky_likert_ranks[doc_id] for doc_id in doc_ids]
    baseline_rank_list = [baseline_ranks[doc_id] for doc_id in doc_ids]
    embedding_rank_list = [embedding_ranks[doc_id] for doc_id in doc_ids]
    sliding_embedding_rank_list = [sliding_embedding_ranks[doc_id] for doc_id in doc_ids]
    sliding_likert_rank_list = [sliding_likert_ranks[doc_id] for doc_id in doc_ids]
    likert_rating_rank_list = [likert_ranks[doc_id] for doc_id in doc_ids]
    lotus_rank_list = [lotus_ranks.get(doc_id, len(abstract_results) + 1) for doc_id in doc_ids]
    calibrated_embedding_rank_list = [calibrated_embedding_ranks[doc_id] for doc_id in doc_ids]
    
    # Compute Kendall's Tau correlation coefficients against ground truth
    tau_picky_embedding, p_value_picky_embedding = kendalltau(ground_truth_rank_list, picky_embedding_rank_list)
    tau_picky_likert, p_value_picky_likert = kendalltau(ground_truth_rank_list, picky_likert_rank_list)
    tau_baseline, p_value_baseline = kendalltau(ground_truth_rank_list, baseline_rank_list)
    tau_embedding, p_value_embedding = kendalltau(ground_truth_rank_list, embedding_rank_list)
    tau_sliding_embedding, p_value_sliding_embedding = kendalltau(ground_truth_rank_list, sliding_embedding_rank_list)
    tau_sliding_likert, p_value_sliding_likert = kendalltau(ground_truth_rank_list, sliding_likert_rank_list)
    tau_likert_rating, p_value_likert_rating = kendalltau(ground_truth_rank_list, likert_rating_rank_list)
    tau_lotus, p_value_lotus = kendalltau(ground_truth_rank_list, lotus_rank_list)
    tau_calibrated_embedding, p_value_calibrated_embedding = kendalltau(ground_truth_rank_list, calibrated_embedding_rank_list)
    
    # Store results in a list of tuples for sorting
    results = [
        ("Picky (Embedding)", tau_picky_embedding, p_value_picky_embedding, method_metrics["Picky (Embedding)"]["runtime"], method_metrics["Picky (Embedding)"]["cost"]),
        ("Picky (Likert)", tau_picky_likert, p_value_picky_likert, method_metrics["Picky (Likert)"]["runtime"], method_metrics["Picky (Likert)"]["cost"]),
        ("Baseline Comparison", tau_baseline, p_value_baseline, method_metrics["Baseline Comparison"]["runtime"], method_metrics["Baseline Comparison"]["cost"]),
        ("Embedding Rating", tau_embedding, p_value_embedding, method_metrics["Embedding Rating"]["runtime"], method_metrics["Embedding Rating"]["cost"]),
        ("Embedding Sliding Window", tau_sliding_embedding, p_value_sliding_embedding, method_metrics["Embedding Sliding Window"]["runtime"], method_metrics["Embedding Sliding Window"]["cost"]),
        ("Likert Sliding Window", tau_sliding_likert, p_value_sliding_likert, method_metrics["Likert Sliding Window"]["runtime"], method_metrics["Likert Sliding Window"]["cost"]),
        ("Likert Rating", tau_likert_rating, p_value_likert_rating, method_metrics["Likert Rating"]["runtime"], method_metrics["Likert Rating"]["cost"]),
        ("Lotus Top K", tau_lotus, p_value_lotus, method_metrics["Lotus Top K"]["runtime"], method_metrics["Lotus Top K"]["cost"]),
        ("Calibrated Embedding Sort", tau_calibrated_embedding, p_value_calibrated_embedding, method_metrics["Calibrated Embedding Sort"]["runtime"], method_metrics["Calibrated Embedding Sort"]["cost"])
    ]
    
    # Sort results by Kendall's tau value in descending order
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    # Create a Rich table for method comparison
    table = Table(title="Synthetic Abstract Ordering Results (vs Ground Truth)")
    
    # Add columns
    table.add_column("Method", style="cyan")
    table.add_column("Kendall's Tau", justify="right", style="green")
    table.add_column("p-value", justify="right", style="green")
    table.add_column("Runtime (s)", justify="right", style="yellow")
    table.add_column("Cost ($)", justify="right", style="yellow")
    
    # Add rows
    for method, tau, p_value, runtime, cost in sorted_results:
        table.add_row(
            method,
            f"{tau:.4f}",
            f"{p_value:.4f}",
            f"{runtime:.2f}",
            f"{cost:.4f}"
        )
    
    # Print the table
    console.print(table)
    
    # Calculate NDCG@10 for each method using ground truth
    # Create ground truth relevance array (inversely related to rank, higher = more relevant)
    y_true = np.zeros(len(abstract_results))
    num_abstracts = len(abstract_results)
    
    for i, doc in enumerate(abstract_results):
        # Use inverse of rank so that highest relevance = lowest number (for ascending order)
        y_true[i] = num_abstracts - ground_truth_ranks[doc["id"]] + 1
    
    # Calculate NDCG@10 for each method
    ndcg_table = Table(title="NDCG@10 Results (vs Ground Truth)")
    ndcg_table.add_column("Method", style="cyan")
    ndcg_table.add_column("NDCG@10", justify="right", style="green")
    
    # Prepare arrays for each method's ranking
    method_results = [
        ("Picky (Embedding)", order_results_picky, picky_embedding_ranks),
        ("Picky (Likert)", order_results_picky_likert, picky_likert_ranks),
        ("Baseline Comparison", order_results_baseline, baseline_ranks),
        ("Embedding Rating", order_results_embedding, embedding_ranks),
        ("Embedding Sliding Window", order_results_sliding_embedding, sliding_embedding_ranks),
        ("Likert Sliding Window", order_results_sliding_likert, sliding_likert_ranks),
        ("Likert Rating", order_results_likert, likert_ranks),
        ("Lotus Top K", lotus_results, lotus_ranks),
        ("Calibrated Embedding Sort", order_results_calibrated_embedding, calibrated_embedding_ranks)
    ]
    
    # Calculate NDCG for each method
    ndcg_scores = []
    for method_name, _, method_ranks in method_results:
        # Create predicted relevance array (higher ranks should have higher scores)
        y_pred = np.zeros(len(abstract_results))
        
        for i, doc in enumerate(abstract_results):
            # Invert rank for scoring (lower rank = higher relevance)
            y_pred[i] = num_abstracts - method_ranks.get(doc["id"], num_abstracts + 1)
        
        # Calculate NDCG
        ndcg_value = calculate_ndcg(y_true, y_pred, k=10)
        ndcg_scores.append((method_name, ndcg_value))
        ndcg_table.add_row(method_name, f"{ndcg_value:.4f}")
    
    # Sort NDCG scores and recreate table with sorted results
    sorted_ndcg = sorted(ndcg_scores, key=lambda x: x[1], reverse=True)
    sorted_ndcg_table = Table(title="NDCG@10 Results (vs Ground Truth) - Sorted")
    sorted_ndcg_table.add_column("Method", style="cyan")
    sorted_ndcg_table.add_column("NDCG@10", justify="right", style="green")
    
    for method_name, ndcg_value in sorted_ndcg:
        sorted_ndcg_table.add_row(method_name, f"{ndcg_value:.4f}")
    
    # Print NDCG results
    console.print(sorted_ndcg_table)
    
    # Also show a sample of the sorted results from the best method
    best_method = sorted_results[0][0]
    best_results = None
    
    if best_method == "Picky (Embedding)":
        best_results = order_results_picky
    elif best_method == "Picky (Likert)":
        best_results = order_results_picky_likert
    elif best_method == "Baseline Comparison":
        best_results = order_results_baseline
    elif best_method == "Embedding Rating":
        best_results = order_results_embedding
    elif best_method == "Embedding Sliding Window":
        best_results = order_results_sliding_embedding
    elif best_method == "Likert Sliding Window":
        best_results = order_results_sliding_likert
    elif best_method == "Likert Rating":
        best_results = order_results_likert
    elif best_method == "Lotus Top K":
        best_results = lotus_results
    elif best_method == "Calibrated Embedding Sort":
        best_results = order_results_calibrated_embedding
    
    # Create a table to show a portion of the best results
    if best_results:
        best_sorted_abstracts = [(doc["id"], doc["accuracy"]) for doc in best_results]
        
        result_table = Table(title=f"Sorted Abstracts (Sample from {best_method})")
        result_table.add_column("Sorted Rank", style="cyan", justify="right")
        result_table.add_column("Paper ID", style="yellow")
        result_table.add_column("Accuracy", style="green", justify="right")
        result_table.add_column("True Rank", style="magenta", justify="right")
        
        # Show first, middle, and last samples to verify ordering
        display_count = min(5, len(best_sorted_abstracts))
        sample_indices = list(range(display_count)) + list(range(len(best_sorted_abstracts) // 2 - display_count // 2, len(best_sorted_abstracts) // 2 + display_count // 2)) + list(range(len(best_sorted_abstracts) - display_count, len(best_sorted_abstracts)))
        
        for idx in sample_indices:
            if idx < len(best_sorted_abstracts):
                paper_id, accuracy = best_sorted_abstracts[idx]
                true_rank = ground_truth_ranks[paper_id]
                result_table.add_row(
                    str(idx + 1),
                    paper_id,
                    f"{accuracy:.4f}",
                    str(true_rank)
                )
        
        console.print(result_table)
    
    return sorted_results, sorted_ndcg, lotus_num_calls

def scifact():
    # Initialize lists to collect results from each trial for scifact
    scifact_results = []
    scifact_ndcg_results = []
    n_trials = 3
    
    # Run the scifact test multiple times and compute the averages
    for trial in range(n_trials):
        # Sleep for 10 seconds to avoid rate limiting
        if trial > 0:
            time.sleep(10)
        console.print(f"\n[bold magenta]===== Running SciFact Trial {trial+1}/{n_trials} =====[/bold magenta]\n")
        
        # Run the scifact test
        console.print(f"\n[bold cyan]Testing SciFact Dataset (Trial {trial+1})[/bold cyan]\n")
        trial_scifact_results, trial_scifact_ndcg, _ = test_order_scifact_dataset(api_wrapper)
        scifact_results.append(trial_scifact_results)
        scifact_ndcg_results.append(trial_scifact_ndcg)
    
    # Process results for SciFact
    console.print("\n[bold green]===== SCIFACT DATASET AGGREGATE RESULTS =====[/bold green]")
    
    # Compute averages for each method across all trials
    methods = set()
    # First collect all unique method names
    for trial_result in scifact_results:
        for method, _, _, _, _ in trial_result:
            methods.add(method)
    
    # Initialize dictionary to store aggregated metrics
    average_metrics = {
        "runtime": {},
        "cost": {}
    }
    
    for method in methods:
        average_metrics["runtime"][method] = []
        average_metrics["cost"][method] = []
    
    # Sum up metrics for each method across trials
    for trial_result in scifact_results:
        for method, _, _, runtime, cost in trial_result:
            average_metrics["runtime"][method].append(runtime)
            average_metrics["cost"][method].append(cost)
    
    # Create a table for the averaged results
    avg_table = Table(title=f"AVERAGE SciFact Dataset Results ({n_trials} Trials)")
    avg_table.add_column("Method", style="cyan")
    avg_table.add_column("Avg Runtime (s)", justify="right", style="yellow")
    avg_table.add_column("Avg Cost ($)", justify="right", style="yellow")
    
    # Calculate and add rows
    for method in methods:
        if average_metrics["runtime"][method]:
            avg_runtime = sum(average_metrics["runtime"][method]) / len(average_metrics["runtime"][method])
            avg_cost = sum(average_metrics["cost"][method]) / len(average_metrics["cost"][method])
            avg_table.add_row(
                method,
                f"{avg_runtime:.2f}",
                f"{avg_cost:.4f}"
            )
    
    # Print the average table
    console.print(avg_table)
    
    # Calculate average NDCG scores
    # Initialize dictionary to store NDCG sums for each method
    ndcg_averages = {}
    
    # Process NDCG results across trials
    for trial_ndcg in scifact_ndcg_results:
        for method_name, ndcg_value in trial_ndcg:
            if method_name not in ndcg_averages:
                ndcg_averages[method_name] = {"sum": 0.0, "count": 0}
            ndcg_averages[method_name]["sum"] += ndcg_value
            ndcg_averages[method_name]["count"] += 1
    
    # Create a table for average NDCG results
    avg_ndcg_table = Table(title=f"AVERAGE SciFact NDCG@10 Results ({n_trials} Trials)")
    avg_ndcg_table.add_column("Method", style="cyan")
    avg_ndcg_table.add_column("Avg NDCG@10", justify="right", style="green")
    
    # Calculate average NDCG for each method
    avg_ndcg_results = []
    for method_name, data in ndcg_averages.items():
        if data["count"] > 0:
            avg_ndcg = data["sum"] / data["count"]
            avg_ndcg_results.append((method_name, avg_ndcg))
    
    # Sort by average NDCG in descending order
    sorted_avg_ndcg = sorted(avg_ndcg_results, key=lambda x: x[1], reverse=True)
    
    # Add rows to table
    for method_name, avg_ndcg in sorted_avg_ndcg:
        avg_ndcg_table.add_row(method_name, f"{avg_ndcg:.4f}")
    
    # Print the average NDCG table
    console.print(avg_ndcg_table)
    
def run_abstract_tests():
    # Test configurations
    abstract_counts = [200]
    n_trials = 3
    
    # For each abstract count
    for num_abstracts in abstract_counts:
        console.print(f"\n[bold green]===== TESTING WITH {num_abstracts} ABSTRACTS =====[/bold green]")
        
        # Initialize collection lists for this abstract count
        results_collection = []
        ndcg_collection = []
        lotus_calls_collection = []
        
        # Run the test multiple times for this abstract count
        for trial in range(n_trials):
            console.print(f"\n[bold magenta]Running Trial {trial+1}/{n_trials} with {num_abstracts} abstracts[/bold magenta]\n")
            
            # Execute the test
            trial_results, trial_ndcg, lotus_calls = test_order_synthetic_abstracts(api_wrapper, num_abstracts=num_abstracts)
            
            # Store the results
            results_collection.append(trial_results)
            ndcg_collection.append(trial_ndcg)
            lotus_calls_collection.append(lotus_calls)
            
            # Add a short delay between trials to avoid rate limits
            if trial < n_trials - 1:
                console.print("Waiting 10 seconds before next trial...")
                time.sleep(10)
        
        # Process results for this abstract count
        console.print(f"\n[bold cyan]===== AGGREGATE RESULTS FOR {num_abstracts} ABSTRACTS ({n_trials} TRIALS) =====[/bold cyan]")
        
        # Collect all unique method names
        methods = set()
        for trial_result in results_collection:
            for method, _, _, _, _ in trial_result:
                methods.add(method)
        
        # Initialize dictionaries to store aggregate metrics
        avg_metrics = {
            "tau": {},      # Kendall's Tau
            "p_value": {},  # p-value
            "runtime": {},  # Runtime in seconds
            "cost": {}      # Cost in dollars
        }
        
        for method in methods:
            for metric in avg_metrics:
                avg_metrics[metric][method] = []
        
        # Collect metrics for each method across all trials
        for trial_result in results_collection:
            for method, tau, p_value, runtime, cost in trial_result:
                avg_metrics["tau"][method].append(tau)
                avg_metrics["p_value"][method].append(p_value)
                avg_metrics["runtime"][method].append(runtime)
                avg_metrics["cost"][method].append(cost)
        
        # Create table for average Kendall's Tau results
        tau_table = Table(title=f"Average Kendall's Tau Results ({num_abstracts} abstracts, {n_trials} trials)")
        tau_table.add_column("Method", style="cyan")
        tau_table.add_column("Avg Tau", justify="right", style="green")
        tau_table.add_column("Avg p-value", justify="right", style="green")
        tau_table.add_column("Avg Runtime (s)", justify="right", style="yellow")
        tau_table.add_column("Avg Cost ($)", justify="right", style="yellow")
        
        # Calculate averages and populate the table
        method_avgs = []
        for method in methods:
            if avg_metrics["tau"][method]:
                avg_tau = sum(avg_metrics["tau"][method]) / len(avg_metrics["tau"][method])
                avg_p = sum(avg_metrics["p_value"][method]) / len(avg_metrics["p_value"][method])
                avg_runtime = sum(avg_metrics["runtime"][method]) / len(avg_metrics["runtime"][method])
                avg_cost = sum(avg_metrics["cost"][method]) / len(avg_metrics["cost"][method])
                
                method_avgs.append((method, avg_tau, avg_p, avg_runtime, avg_cost))
        
        # Sort by average Tau in descending order
        sorted_avgs = sorted(method_avgs, key=lambda x: x[1], reverse=True)
        
        # Add rows to the table
        for method, avg_tau, avg_p, avg_runtime, avg_cost in sorted_avgs:
            tau_table.add_row(
                method,
                f"{avg_tau:.4f}",
                f"{avg_p:.4f}",
                f"{avg_runtime:.2f}",
                f"{avg_cost:.4f}"
            )
        
        # Print the table
        console.print(tau_table)
        
        # Calculate average NDCG scores
        ndcg_averages = {}
        
        # Process NDCG results across all trials
        for trial_ndcg in ndcg_collection:
            for method_name, ndcg_value in trial_ndcg:
                if method_name not in ndcg_averages:
                    ndcg_averages[method_name] = {"sum": 0.0, "count": 0}
                ndcg_averages[method_name]["sum"] += ndcg_value
                ndcg_averages[method_name]["count"] += 1
        
        # Create table for average NDCG results
        ndcg_table = Table(title=f"Average NDCG@10 Results ({num_abstracts} abstracts, {n_trials} trials)")
        ndcg_table.add_column("Method", style="cyan")
        ndcg_table.add_column("Avg NDCG@10", justify="right", style="green")
        
        # Calculate and sort average NDCG values
        avg_ndcg_results = []
        for method_name, data in ndcg_averages.items():
            if data["count"] > 0:
                avg_ndcg = data["sum"] / data["count"]
                avg_ndcg_results.append((method_name, avg_ndcg))
        
        # Sort by NDCG in descending order
        sorted_ndcg = sorted(avg_ndcg_results, key=lambda x: x[1], reverse=True)
        
        # Add rows to the table
        for method_name, avg_ndcg in sorted_ndcg:
            ndcg_table.add_row(method_name, f"{avg_ndcg:.4f}")
        
        # Print the table
        console.print(ndcg_table)
        
        # Calculate average Lotus calls
        if lotus_calls_collection:
            avg_lotus_calls = sum(lotus_calls_collection) / len(lotus_calls_collection)
            console.print(f"[bold yellow]Average Lotus LLM calls: {avg_lotus_calls:.2f}[/bold yellow]")
    
    console.print("\n[bold green]All tests completed![/bold green]")
    

    


def test_chat_harmfulness(api_wrapper, num_samples=200):
    """
    Test ranking chat transcripts by harmlessness.
    Lower min_harmlessness_score_transcript values indicate more harmful content,
    but we want to rank by harmlessness descending (less harmful at top).
    """
    from scipy.stats import kendalltau
    import numpy as np
    
    console.print("[bold blue]Testing ranking chat transcripts by harmlessness...[/bold blue]")
    
    # Load the data
    transcripts = []
    file_path = "tests/data/anthropic-red-team-attempts.jsonl"
    try:
        # Use pandas to efficiently read jsonl files
        import pandas as pd
        df = pd.read_json(path_or_buf=file_path)
        # Convert to list of dictionaries and limit to num_samples
        transcripts = df.head(num_samples).to_dict(orient="records")
    except FileNotFoundError:
        console.print(f"[bold red]Error: File {file_path} not found[/bold red]")
        return
    
    console.print(f"Loaded {len(transcripts)} transcripts")
    
    # Format transcripts for ordering
    chat_data = []
    for i, transcript in enumerate(transcripts):
        chat_data.append({
            "id": f"transcript_{i}",
            "transcript": transcript["transcript"],
            "min_harmlessness_score": transcript["min_harmlessness_score_transcript"]
        })
    
    # Create the order operation config
    order_config = {
        "name": "order_by_harmlessness",
        "type": "order",
        "batch_size": 10,
        "prompt": """
            Rank these chat transcripts by how harmless the assistant is in its responses. Place the most harmless, most appropriate assistant ones at the top.
        """,
        "input_keys": ["transcript"],
        "direction": "desc",  # Descending puts least harmful at top
        "verbose": True,
        "rerank_call_budget": 10,
        "bypass_cache": True
    }
    
    from docetl.operations.rank import RankOperation
    
    # Create order operation instance
    order_operation = RankOperation(
        api_wrapper,
        order_config,
        default_model="gemini/gemini-2.0-flash",
        max_threads=64,
    )
    
    # Track method metrics
    method_metrics = {}
    
    # Execute picky method
    console.print("[bold blue]Sorting transcripts with picky method (embedding initial)[/bold blue]")
    start_time = time.time()
    order_results_picky, cost_picky = order_operation.execute(chat_data)
    method_metrics["Picky (Embedding)"] = {
        "runtime": time.time() - start_time,
        "cost": cost_picky
    }
    
    # Execute with different initial ordering method
    order_config_likert = order_config.copy()
    order_config_likert["initial_ordering_method"] = "likert"
    order_operation_likert = RankOperation(
        api_wrapper,
        order_config_likert,
        default_model="gemini/gemini-2.0-flash",
        max_threads=64,
    )
    
    console.print("[bold blue]Sorting transcripts with picky method (likert initial)[/bold blue]")
    start_time = time.time()
    order_results_picky_likert, cost_picky_likert = order_operation_likert.execute(chat_data)
    method_metrics["Picky (Likert)"] = {
        "runtime": time.time() - start_time,
        "cost": cost_picky_likert
    }
    
    # Execute baseline comparison
    console.print("[bold blue]Sorting transcripts with baseline comparison method[/bold blue]")
    start_time = time.time()
    order_results_baseline, cost_baseline = order_operation._execute_comparison_qurk(chat_data)
    method_metrics["Baseline Comparison"] = {
        "runtime": time.time() - start_time,
        "cost": cost_baseline
    }
    
    # Execute rating embedding
    console.print("[bold blue]Sorting transcripts with embedding rating method[/bold blue]")
    start_time = time.time()
    order_results_embedding, cost_embedding = order_operation._execute_rating_embedding_qurk(chat_data)
    method_metrics["Embedding Rating"] = {
        "runtime": time.time() - start_time,
        "cost": cost_embedding
    }
    
    # Execute sliding window with embedding
    console.print("[bold blue]Sorting transcripts with sliding window (embedding) method[/bold blue]")
    start_time = time.time()
    order_results_sliding_embedding, cost_sliding_embedding = order_operation._execute_sliding_window_qurk(
        chat_data, 
        initial_ordering_method="embedding"
    )
    method_metrics["Embedding Sliding Window"] = {
        "runtime": time.time() - start_time,
        "cost": cost_sliding_embedding
    }
    
    # Execute sliding window with likert
    console.print("[bold blue]Sorting transcripts with sliding window (likert) method[/bold blue]")
    start_time = time.time()
    order_results_sliding_likert, cost_sliding_likert = order_operation._execute_sliding_window_qurk(
        chat_data, 
        initial_ordering_method="likert"
    )
    method_metrics["Likert Sliding Window"] = {
        "runtime": time.time() - start_time,
        "cost": cost_sliding_likert
    }
    
    # Execute likert rating method
    console.print("[bold blue]Sorting transcripts with likert rating method[/bold blue]")
    start_time = time.time()
    order_results_likert, cost_likert = order_operation._execute_likert_rating_qurk(chat_data)
    method_metrics["Likert Rating"] = {
        "runtime": time.time() - start_time,
        "cost": cost_likert
    }
    
    # Execute calibrated embedding sort
    console.print("[bold blue]Sorting transcripts with calibrated embedding sort method[/bold blue]")
    start_time = time.time()
    order_results_calibrated_embedding, cost_calibrated_embedding = order_operation._execute_calibrated_embedding_sort(chat_data)
    method_metrics["Calibrated Embedding Sort"] = {
        "runtime": time.time() - start_time,
        "cost": cost_calibrated_embedding
    }
    
    # Execute lotus top k
    console.print("[bold blue]Sorting transcripts with Lotus top k method[/bold blue]")
    start_time = time.time()
    df = pd.DataFrame(chat_data)
    sorted_df, stats = df.sem_topk(
        "Which of the following chat transcripts has the most harmless assistant responses? Here is the chat transcript: {transcript}",
        K=10,
        return_stats=True,
    )
    print(stats)
    end_time = time.time()
    lotus_results = sorted_df.to_dict(orient="records")
    lotus_num_calls = stats["total_llm_calls"]
    lotus_cost = stats["total_tokens"] * 0.15 / 1000000
    lotus_runtime = end_time - start_time
    method_metrics["Lotus Top K"] = {
        "runtime": lotus_runtime,
        "cost": lotus_cost
    }
    
    # Create ground truth rankings based on harmlessness scores
    # Sort in descending order (most harmless at top)
    sorted_indices = sorted(range(len(chat_data)), key=lambda i: chat_data[i]["min_harmlessness_score"], reverse=True)
    ground_truth_ranks = {chat_data[i]["id"]: sorted_indices.index(i) + 1 for i in range(len(chat_data))}
    
    # Extract method rankings
    picky_ranks = {doc["id"]: doc["_rank"] for doc in order_results_picky}
    picky_likert_ranks = {doc["id"]: doc["_rank"] for doc in order_results_picky_likert}
    baseline_ranks = {doc["id"]: doc["_rank"] for doc in order_results_baseline}
    embedding_ranks = {doc["id"]: doc["_rank"] for doc in order_results_embedding}
    sliding_embedding_ranks = {doc["id"]: doc["_rank"] for doc in order_results_sliding_embedding}
    sliding_likert_ranks = {doc["id"]: doc["_rank"] for doc in order_results_sliding_likert}
    likert_ranks = {doc["id"]: doc["_rank"] for doc in order_results_likert}
    lotus_ranks = {doc["id"]: doc["_rank"] for doc in lotus_results}
    calibrated_embedding_ranks = {doc["id"]: doc["_rank"] for doc in order_results_calibrated_embedding}
    
    # Prepare lists for Kendall's Tau computation
    doc_ids = list(ground_truth_ranks.keys())
    
    ground_truth_rank_list = [ground_truth_ranks[doc_id] for doc_id in doc_ids]
    picky_rank_list = [picky_ranks[doc_id] for doc_id in doc_ids]
    picky_likert_rank_list = [picky_likert_ranks[doc_id] for doc_id in doc_ids]
    baseline_rank_list = [baseline_ranks[doc_id] for doc_id in doc_ids]
    embedding_rank_list = [embedding_ranks[doc_id] for doc_id in doc_ids]
    sliding_embedding_rank_list = [sliding_embedding_ranks[doc_id] for doc_id in doc_ids]
    sliding_likert_rank_list = [sliding_likert_ranks[doc_id] for doc_id in doc_ids]
    likert_rating_rank_list = [likert_ranks[doc_id] for doc_id in doc_ids]
    lotus_rank_list = [lotus_ranks.get(doc_id, len(chat_data) + 1) for doc_id in doc_ids]
    calibrated_embedding_rank_list = [calibrated_embedding_ranks[doc_id] for doc_id in doc_ids]
    
    # Compute Kendall's Tau correlation coefficients against ground truth
    tau_picky, p_value_picky = kendalltau(ground_truth_rank_list, picky_rank_list)
    tau_picky_likert, p_value_picky_likert = kendalltau(ground_truth_rank_list, picky_likert_rank_list)
    tau_baseline, p_value_baseline = kendalltau(ground_truth_rank_list, baseline_rank_list)
    tau_embedding, p_value_embedding = kendalltau(ground_truth_rank_list, embedding_rank_list)
    tau_sliding_embedding, p_value_sliding_embedding = kendalltau(ground_truth_rank_list, sliding_embedding_rank_list)
    tau_sliding_likert, p_value_sliding_likert = kendalltau(ground_truth_rank_list, sliding_likert_rank_list)
    tau_likert_rating, p_value_likert_rating = kendalltau(ground_truth_rank_list, likert_rating_rank_list)
    tau_lotus, p_value_lotus = kendalltau(ground_truth_rank_list, lotus_rank_list)
    tau_calibrated_embedding, p_value_calibrated_embedding = kendalltau(ground_truth_rank_list, calibrated_embedding_rank_list)
    
    # Store results in a list of tuples for sorting
    results = [
        ("Picky (Embedding)", tau_picky, p_value_picky, method_metrics["Picky (Embedding)"]["runtime"], method_metrics["Picky (Embedding)"]["cost"]),
        ("Picky (Likert)", tau_picky_likert, p_value_picky_likert, method_metrics["Picky (Likert)"]["runtime"], method_metrics["Picky (Likert)"]["cost"]),
        ("Baseline Comparison", tau_baseline, p_value_baseline, method_metrics["Baseline Comparison"]["runtime"], method_metrics["Baseline Comparison"]["cost"]),
        ("Embedding Rating", tau_embedding, p_value_embedding, method_metrics["Embedding Rating"]["runtime"], method_metrics["Embedding Rating"]["cost"]),
        ("Embedding Sliding Window", tau_sliding_embedding, p_value_sliding_embedding, method_metrics["Embedding Sliding Window"]["runtime"], method_metrics["Embedding Sliding Window"]["cost"]),
        ("Likert Sliding Window", tau_sliding_likert, p_value_sliding_likert, method_metrics["Likert Sliding Window"]["runtime"], method_metrics["Likert Sliding Window"]["cost"]),
        ("Likert Rating", tau_likert_rating, p_value_likert_rating, method_metrics["Likert Rating"]["runtime"], method_metrics["Likert Rating"]["cost"]),
        ("Lotus Top K", tau_lotus, p_value_lotus, method_metrics["Lotus Top K"]["runtime"], method_metrics["Lotus Top K"]["cost"]),
        ("Calibrated Embedding Sort", tau_calibrated_embedding, p_value_calibrated_embedding, method_metrics["Calibrated Embedding Sort"]["runtime"], method_metrics["Calibrated Embedding Sort"]["cost"])
    ]
    
    # Sort results by Kendall's tau value in descending order
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    # Create a Rich table for method comparison
    table = Table(title="Harmfulness Ranking Results")
    table.add_column("Method", style="cyan")
    table.add_column("Kendall's Tau", justify="right", style="green")
    table.add_column("p-value", justify="right", style="yellow")
    table.add_column("Runtime (s)", justify="right", style="yellow")
    table.add_column("Cost ($)", justify="right", style="yellow")
    
    # Add rows for each method
    method_results = [
        ("Picky (Embedding)", tau_picky, p_value_picky, method_metrics["Picky (Embedding)"]["runtime"], method_metrics["Picky (Embedding)"]["cost"]),
        ("Picky (Likert)", tau_picky_likert, p_value_picky_likert, method_metrics["Picky (Likert)"]["runtime"], method_metrics["Picky (Likert)"]["cost"]),
        ("Baseline Comparison", tau_baseline, p_value_baseline, method_metrics["Baseline Comparison"]["runtime"], method_metrics["Baseline Comparison"]["cost"]),
        ("Embedding Rating", tau_embedding, p_value_embedding, method_metrics["Embedding Rating"]["runtime"], method_metrics["Embedding Rating"]["cost"]),
        ("Embedding Sliding Window", tau_sliding_embedding, p_value_sliding_embedding, method_metrics["Embedding Sliding Window"]["runtime"], method_metrics["Embedding Sliding Window"]["cost"]),
        ("Likert Sliding Window", tau_sliding_likert, p_value_sliding_likert, method_metrics["Likert Sliding Window"]["runtime"], method_metrics["Likert Sliding Window"]["cost"]),
        ("Likert Rating", tau_likert_rating, p_value_likert_rating, method_metrics["Likert Rating"]["runtime"], method_metrics["Likert Rating"]["cost"]),
        ("Lotus Top K", tau_lotus, p_value_lotus, method_metrics["Lotus Top K"]["runtime"], method_metrics["Lotus Top K"]["cost"]),
        ("Calibrated Embedding Sort", tau_calibrated_embedding, p_value_calibrated_embedding, method_metrics["Calibrated Embedding Sort"]["runtime"], method_metrics["Calibrated Embedding Sort"]["cost"])
    ]
    
    # Sort by Kendall's Tau
    sorted_results = sorted(method_results, key=lambda x: x[1], reverse=True)
    
    for method, tau, p_value, runtime, cost in sorted_results:
        table.add_row(
            method,
            f"{tau:.4f}",
            f"{p_value:.4f}",
            f"{runtime:.2f}",
            f"{cost:.4f}"
        )
    
    console.print(table)
    
    # Calculate NDCG
    y_true = np.zeros(len(chat_data))
    for i, doc in enumerate(chat_data):
        # higher ranks are higher scores, so we invert the rank
        doc_id = doc["id"]
        y_true[i] = len(chat_data) - ground_truth_ranks[doc_id] + 1
    
    # Calculate NDCG for each method
    ndcg_table = Table(title="NDCG@10 Results")
    ndcg_table.add_column("Method", style="cyan") 
    ndcg_table.add_column("NDCG@10", justify="right", style="green")
    
    ndcg_scores = []
    
    for method_name, _, _, _, _ in sorted_results:
        # Get the ranks for this method
        if method_name == "Picky (Embedding)":
            method_ranks = picky_ranks
        elif method_name == "Picky (Likert)":
            method_ranks = picky_likert_ranks
        elif method_name == "Baseline Comparison":
            method_ranks = baseline_ranks
        elif method_name == "Embedding Rating":
            method_ranks = embedding_ranks
        elif method_name == "Embedding Sliding Window":
            method_ranks = sliding_embedding_ranks
        elif method_name == "Likert Sliding Window":
            method_ranks = sliding_likert_ranks
        elif method_name == "Likert Rating":
            method_ranks = likert_ranks
        elif method_name == "Lotus Top K":
            method_ranks = lotus_ranks
        elif method_name == "Calibrated Embedding Sort":
            method_ranks = calibrated_embedding_ranks
        
        # Calculate predicted relevance scores (higher for items ranked higher)
        y_pred = np.zeros(len(chat_data))
        for i, doc in enumerate(chat_data):
            doc_id = doc["id"]
            # Inverted rank (higher rank = higher relevance)
            y_pred[i] = len(chat_data) - method_ranks.get(doc_id, len(chat_data)) + 1
        
        # Calculate NDCG
        ndcg = calculate_ndcg(y_true, y_pred, k=10)
        ndcg_scores.append((method_name, ndcg))
    
    # Sort by NDCG
    sorted_ndcg = sorted(ndcg_scores, key=lambda x: x[1], reverse=True)
    
    # Add rows to the NDCG table
    for method_name, ndcg in sorted_ndcg:
        ndcg_table.add_row(method_name, f"{ndcg:.4f}")
    
    console.print(ndcg_table)
    
    # Show top 5 results from the best method
    best_method_name = sorted_results[0][0]
    console.print(f"\n[bold]Top 5 transcripts from best method ({best_method_name}):[/bold]")
    
    if best_method_name == "Picky (Embedding)":
        best_results = order_results_picky
    elif best_method_name == "Picky (Likert)":
        best_results = order_results_picky_likert
    elif best_method_name == "Baseline Comparison":
        best_results = order_results_baseline
    elif best_method_name == "Embedding Rating":
        best_results = order_results_embedding
    elif best_method_name == "Embedding Sliding Window":
        best_results = order_results_sliding_embedding
    elif best_method_name == "Likert Sliding Window":
        best_results = order_results_sliding_likert
    elif best_method_name == "Likert Rating":
        best_results = order_results_likert
    elif best_method_name == "Lotus Top K":
        best_results = lotus_results
    elif best_method_name == "Calibrated Embedding Sort":
        best_results = order_results_calibrated_embedding
    
    for i, doc in enumerate(best_results[:5]):
        doc_id = doc["id"]
        original_index = int(doc_id.split("_")[1])
        harmfulness_score = chat_data[original_index]["min_harmlessness_score"]
        ground_rank = ground_truth_ranks[doc_id]
        
        console.print(f"{i+1}. {doc_id} (Ground truth rank: {ground_rank}, Harmfulness score: {harmfulness_score:.4f})")
        # Show a short preview of the transcript
        transcript_preview = doc["transcript"][:100].replace("\n", " ") + "..."
        console.print(f"   Preview: {transcript_preview}")
    
    console.print("\n[bold green]Harmfulness ranking test completed![/bold green]")
    return sorted_results, sorted_ndcg, lotus_num_calls

def run_harmlessness_tests():
    """
    Run the chat harmfulness ranking test multiple times and compute average metrics.
    """
    n_trials = 3
    sample_sizes = [200]  # Number of transcripts to test with
    
    # For each sample size
    for num_samples in sample_sizes:
        console.print(f"\n[bold green]===== TESTING WITH {num_samples} TRANSCRIPTS =====[/bold green]")
        
        # Initialize collection lists
        results_collection = []
        ndcg_collection = []
        lotus_calls_collection = []
        
        # Run the test multiple times
        for trial in range(n_trials):
            console.print(f"\n[bold magenta]Running Trial {trial+1}/{n_trials} with {num_samples} transcripts[/bold magenta]\n")
            
            # Execute the test
            trial_results, trial_ndcg, lotus_calls = test_chat_harmfulness(api_wrapper, num_samples=num_samples)
            
            # Store the results
            results_collection.append(trial_results)
            ndcg_collection.append(trial_ndcg)
            lotus_calls_collection.append(lotus_calls)
            
            # Add a short delay between trials to avoid rate limits
            if trial < n_trials - 1:
                console.print("Waiting 10 seconds before next trial...")
                time.sleep(10)
        
        # Process results
        console.print(f"\n[bold cyan]===== AGGREGATE RESULTS FOR {num_samples} TRANSCRIPTS ({n_trials} TRIALS) =====[/bold cyan]")
        
        # Collect all unique method names
        methods = set()
        for trial_result in results_collection:
            for method, _, _, _, _ in trial_result:
                methods.add(method)
        
        # Initialize dictionaries to store aggregate metrics
        avg_metrics = {
            "tau": {},      # Kendall's Tau
            "p_value": {},  # p-value
            "runtime": {},  # Runtime in seconds
            "cost": {}      # Cost in dollars
        }
        
        for method in methods:
            for metric in avg_metrics:
                avg_metrics[metric][method] = []
        
        # Collect metrics for each method across all trials
        for trial_result in results_collection:
            for method, tau, p_value, runtime, cost in trial_result:
                avg_metrics["tau"][method].append(tau)
                avg_metrics["p_value"][method].append(p_value)
                avg_metrics["runtime"][method].append(runtime)
                avg_metrics["cost"][method].append(cost)
        
        # Create table for average Kendall's Tau results
        tau_table = Table(title=f"Average Harmfulness Ranking Results ({num_samples} transcripts, {n_trials} trials)")
        tau_table.add_column("Method", style="cyan")
        tau_table.add_column("Avg Tau", justify="right", style="green")
        tau_table.add_column("Avg p-value", justify="right", style="green")
        tau_table.add_column("Avg Runtime (s)", justify="right", style="yellow")
        tau_table.add_column("Avg Cost ($)", justify="right", style="yellow")
        
        # Calculate averages and populate the table
        method_avgs = []
        for method in methods:
            if avg_metrics["tau"][method]:
                avg_tau = sum(avg_metrics["tau"][method]) / len(avg_metrics["tau"][method])
                avg_p = sum(avg_metrics["p_value"][method]) / len(avg_metrics["p_value"][method])
                avg_runtime = sum(avg_metrics["runtime"][method]) / len(avg_metrics["runtime"][method])
                avg_cost = sum(avg_metrics["cost"][method]) / len(avg_metrics["cost"][method])
                
                method_avgs.append((method, avg_tau, avg_p, avg_runtime, avg_cost))
        
        # Sort by average Tau in descending order
        sorted_avgs = sorted(method_avgs, key=lambda x: x[1], reverse=True)
        
        # Add rows to the table
        for method, avg_tau, avg_p, avg_runtime, avg_cost in sorted_avgs:
            tau_table.add_row(
                method,
                f"{avg_tau:.4f}",
                f"{avg_p:.4f}",
                f"{avg_runtime:.2f}",
                f"{avg_cost:.4f}"
            )
        
        # Print the table
        console.print(tau_table)
        
        # Calculate average NDCG scores
        ndcg_averages = {}
        
        # Process NDCG results across all trials
        for trial_ndcg in ndcg_collection:
            for method_name, ndcg_value in trial_ndcg:
                if method_name not in ndcg_averages:
                    ndcg_averages[method_name] = {"sum": 0.0, "count": 0}
                ndcg_averages[method_name]["sum"] += ndcg_value
                ndcg_averages[method_name]["count"] += 1
        
        # Create table for average NDCG results
        ndcg_table = Table(title=f"Average NDCG@10 Results ({num_samples} transcripts, {n_trials} trials)")
        ndcg_table.add_column("Method", style="cyan")
        ndcg_table.add_column("Avg NDCG@10", justify="right", style="green")
        
        # Calculate and sort average NDCG values
        avg_ndcg_results = []
        for method_name, data in ndcg_averages.items():
            if data["count"] > 0:
                avg_ndcg = data["sum"] / data["count"]
                avg_ndcg_results.append((method_name, avg_ndcg))
        
        # Sort by NDCG in descending order
        sorted_ndcg = sorted(avg_ndcg_results, key=lambda x: x[1], reverse=True)
        
        # Add rows to the table
        for method_name, avg_ndcg in sorted_ndcg:
            ndcg_table.add_row(method_name, f"{avg_ndcg:.4f}")
        
        # Print the table
        console.print(ndcg_table)
        
        # Calculate average Lotus calls
        if lotus_calls_collection:
            avg_lotus_calls = sum(lotus_calls_collection) / len(lotus_calls_collection)
            console.print(f"[bold yellow]Average Lotus LLM calls: {avg_lotus_calls:.2f}[/bold yellow]")
    
    console.print("\n[bold green]All harmfulness tests completed![/bold green]")


if __name__ == "__main__":
    # Uncomment the function you want to run:
    test_chat_harmfulness(api_wrapper)
    # run_harmlessness_tests()