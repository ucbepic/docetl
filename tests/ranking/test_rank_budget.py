"""
Test budget variations for different ranking methods.

This file tests three ranking methods (Likert Rating, Likert Sliding Window, and Picky Window with Likert initial)
with varying budget parameters (batch_size and rerank_call_budget) and evaluates their performance.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from sklearn.metrics import ndcg_score
from rich.console import Console
from rich.table import Table
import json
import random

from docetl.runner import DSLRunner

api_wrapper = DSLRunner(
    {
        "default_model": "gpt-4o-mini",
        "operations": [],
        "pipeline": {"steps": [], "output": {"path": "/tmp/testingdocetl.json"}},
    },
    max_threads=64,
)

# Initialize console for rich output
console = Console()

def calculate_ndcg(y_true, y_pred, k=50):
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

# Dataset-specific configurations

def get_dataset_config(dataset_name):
    """
    Get the appropriate configuration for a specific dataset.
    
    Args:
        dataset_name (str): Name of the dataset
        
    Returns:
        dict: Configuration dictionary for the dataset
    """
    configs = {
        "harmfulness": {
            "name": "order_by_harmlessness",
            "type": "order",
            "prompt": """
                Rank these chat transcripts by how harmless the assistant is in its responses. Place the most harmless, most appropriate assistant ones at the top.
            """,
            "input_keys": ["transcript"],
            "direction": "desc",  # Descending puts least harmful at top
            "verbose": True,
            "bypass_cache": True,
            "litellm_completion_kwargs": {
                "temperature": 0.0,
            }
        },
        "synthetic_abstracts": {
            "name": "order_by_accuracy",
            "type": "order",
            "prompt": """
                Order these research paper abstracts based on the accuracy value they report, from lowest accuracy to highest.
            """,
            "input_keys": ["abstract"],
            "direction": "asc",  # Ascending order (lowest first)
            "verbose": True,
            "bypass_cache": True,
            "litellm_completion_kwargs": {
                "temperature": 0.0,
            }
        },
        "medical_pain": {
            "name": "order_by_pain_level",
            "type": "order",
            "prompt": """
                Order these medical transcripts based on how much pain the patient is experiencing or reporting, from most pain to least pain.
            """,
            "input_keys": ["src"],
            "direction": "desc",  # Highest pain first
            "verbose": True,
            "bypass_cache": True,
            "litellm_completion_kwargs": {
                "temperature": 0.0,
            }
        }
    }
    
    if dataset_name not in configs:
        raise ValueError(f"No configuration available for dataset '{dataset_name}'")
    
    return configs[dataset_name]

# Data loading functions

def load_harmfulness_data(num_samples=200):
    """
    Load chat transcripts from the Anthropic Red Team dataset.
    
    Args:
        num_samples (int): Number of transcripts to use from the dataset
        
    Returns:
        tuple: (chat_data, ground_truth_ranks) where:
            - chat_data is a list of dictionaries with transcript data
            - ground_truth_ranks is a dictionary mapping document IDs to ground truth ranks
    """
    console.print("[bold blue]Loading harmfulness dataset...[/bold blue]")
    
    # Load the data
    transcripts = []
    file_path = "tests/data/anthropic-red-team-attempts.jsonl"
    try:
        # Use pandas to efficiently read jsonl files
        df = pd.read_json(path_or_buf=file_path)
        # Convert to list of dictionaries and limit to num_samples
        transcripts = df.head(num_samples).to_dict(orient="records")
    except FileNotFoundError:
        console.print(f"[bold red]Error: File {file_path} not found[/bold red]")
        return [], {}
    
    console.print(f"Loaded {len(transcripts)} transcripts")
    
    # Format transcripts for ordering
    chat_data = []
    for i, transcript in enumerate(transcripts):
        chat_data.append({
            "id": f"transcript_{i}",
            "transcript": transcript["transcript"],
            "min_harmlessness_score": transcript["min_harmlessness_score_transcript"]
        })
    
    # Create ground truth rankings based on harmlessness scores
    # Sort in descending order (most harmless at top)
    sorted_indices = sorted(range(len(chat_data)), key=lambda i: chat_data[i]["min_harmlessness_score"], reverse=True)
    ground_truth_ranks = {chat_data[i]["id"]: sorted_indices.index(i) + 1 for i in range(len(chat_data))}
    
    return chat_data, ground_truth_ranks

def load_synthetic_abstracts_data(num_abstracts=200):
    """
    Load or generate synthetic paper abstracts with accuracy values.
    
    Args:
        num_abstracts (int): Number of synthetic abstracts to generate
        
    Returns:
        tuple: (abstract_data, ground_truth_ranks)
    """
    from docetl.operations.map import MapOperation
    
    console.print("[bold blue]Generating synthetic abstracts dataset...[/bold blue]")
    
    # Generate random accuracy values between 0 and 1
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
    
    # Create MapOperation to generate synthetic abstracts
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
    map_operation = MapOperation(
        api_wrapper,
        map_config,
        default_model="gpt-4o-mini",
        max_threads=64,
    )
    
    # Apply the map operation
    console.print(f"Generating {num_abstracts} synthetic paper abstracts...")
    abstract_results, _ = map_operation.execute(accuracy_data)
    
    
    # Create ground truth rankings based on original accuracy values
    # For ascending order (lowest to highest), smaller rank goes to smaller accuracy
    sorted_indices = sorted(range(len(accuracy_values)), key=lambda i: accuracy_values[i])
    ground_truth_ranks = {f"paper_{i}": sorted_indices.index(i) + 1 for i in range(len(accuracy_values))}
    
    return abstract_results, ground_truth_ranks

def load_medical_data(num_samples=200):
    """
    Load medical transcripts from workloads/medical/raw.json.
    
    Args:
        num_samples (int): Number of transcripts to use from the dataset
        
    Returns:
        tuple: (medical_data, ground_truth_ranks) where:
            - medical_data is a list of dictionaries with transcript data
            - ground_truth_ranks is a dictionary mapping document IDs to ground truth ranks
            generated by the comparison QUrk method with batch_size=4
    """
    from docetl.operations.rank import RankOperation
    
    console.print("[bold blue]Loading medical transcripts dataset...[/bold blue]")
    
    # Load medical transcripts
    try:
        with open("workloads/medical/raw.json", "r") as f:
            medical_data = json.load(f)
        
        # Limit to num_samples
        medical_data = medical_data[:num_samples]
        
        # Add IDs to data if not present
        for i, doc in enumerate(medical_data):
            if "id" not in doc:
                doc["id"] = f"medical_{i}"
        
        console.print(f"Loaded {len(medical_data)} medical transcripts")
    except FileNotFoundError:
        console.print(f"[bold red]Error: File workloads/medical/raw.json not found[/bold red]")
        return [], {}
    
    # Create order operation for pain level assessment with batch_size=4
    order_config = {
        "name": "order_by_pain_level",
        "type": "order",
        "batch_size": 4,  # Using batch size 4 as requested
        "prompt": """
            Order these medical transcripts based on how much pain the patient is experiencing or reporting, from most pain to least pain.
        """,
        "input_keys": ["src"],
        "direction": "desc",  # Highest pain first
        "verbose": True,
    }
    
    operation = RankOperation(
        api_wrapper,
        order_config,
        default_model="gpt-4o-mini",
        max_threads=64,
    )
    
    # Use comparison QUrk to establish ground truth
    console.print("[bold blue]Creating ground truth rankings using comparison QUrk with batch_size=4...[/bold blue]")
    baseline_results, _ = operation._execute_comparison_qurk(medical_data)
    
    # Extract ground truth ranks
    ground_truth_ranks = {doc["id"]: doc["_rank"] for doc in baseline_results}
    
    return medical_data, ground_truth_ranks

def test_ranking_with_budget_variation(api_wrapper, dataset, ground_truth_ranks, dataset_name):
    """
    Test ranking with varying budget parameters.
    
    This test evaluates four methods:
    - Likert Rating (varying batch_size)
    - Likert Sliding Window (varying batch_size)
    - Picky Window with Likert initial (varying rerank_call_budget)
    - Picky Window with Calibrated Embedding initial (varying rerank_call_budget)
    
    Args:
        api_wrapper: API wrapper for LLM calls
        dataset: List of documents to rank
        ground_truth_ranks: Dictionary mapping document IDs to ground truth ranks
        dataset_name: Name of the dataset being used
        
    Returns:
        dict: Results for each method and parameter configuration
    """
    from scipy.stats import kendalltau
    import numpy as np
    from docetl.operations.rank import RankOperation
    
    console.print("[bold blue]Testing ranking methods with varying budget parameters...[/bold blue]")
    console.print(f"Testing with {len(dataset)} documents")
    
    # Parameters to vary
    batch_sizes = [8, 16, 32, 64]  # For Likert Rating and Sliding Window
    rerank_call_budgets = [10, 20, 40, 80]  # For Picky Window
    
    # Get dataset-specific configuration
    base_config = get_dataset_config(dataset_name)
    
    # Store results for each configuration
    results = {
        "Likert Rating": [],
        "Likert Sliding Window": [],
        "Picky Window (Likert)": [],
        "Picky Window (Calibrated Embedding)": []
    }
    
    # Test Likert Rating with different batch sizes
    for batch_size in batch_sizes:
        console.print(f"[bold blue]Testing Likert Rating with batch_size={batch_size}[/bold blue]")
        
        config = base_config.copy()
        config["batch_size"] = batch_size
        
        operation = RankOperation(
            api_wrapper,
            config,
            default_model="gpt-4o-mini",
            max_threads=64,
        )
        
        start_time = time.time()
        order_results, cost = operation._execute_likert_rating_qurk(dataset)
        runtime = time.time() - start_time
        
        # Extract ranks and calculate metrics
        ranks = {doc["id"]: doc["_rank"] for doc in order_results}
        doc_ids = list(ground_truth_ranks.keys())
        
        ground_truth_rank_list = [ground_truth_ranks[doc_id] for doc_id in doc_ids]
        method_rank_list = [ranks[doc_id] for doc_id in doc_ids]
        
        # Calculate Kendall's Tau
        tau, p_value = kendalltau(ground_truth_rank_list, method_rank_list)
        
        # Calculate NDCG@50
        y_true = np.zeros(len(dataset))
        y_pred = np.zeros(len(dataset))
        
        for i, doc in enumerate(dataset):
            doc_id = doc["id"]
            # Inverted rank (higher rank = higher relevance)
            y_true[i] = len(dataset) - ground_truth_ranks[doc_id] + 1
            y_pred[i] = len(dataset) - ranks[doc_id] + 1
        
        ndcg50 = calculate_ndcg(y_true, y_pred, k=50)
        ndcg10 = calculate_ndcg(y_true, y_pred, k=10)
        
        results["Likert Rating"].append({
            "batch_size": batch_size,
            "tau": tau,
            "p_value": p_value,
            "ndcg": ndcg50,
            "ndcg10": ndcg10,
            "runtime": runtime,
            "cost": cost
        })
        
        console.print(f"Batch Size: {batch_size}, Tau: {tau:.4f}, NDCG@50: {ndcg50:.4f}, NDCG@10: {ndcg10:.4f}, Cost: ${cost:.4f}")
    
    # Test Likert Sliding Window with different batch sizes
    for batch_size in batch_sizes:
        console.print(f"[bold blue]Testing Likert Sliding Window with batch_size={batch_size}[/bold blue]")
        
        config = base_config.copy()
        config["batch_size"] = batch_size
        
        operation = RankOperation(
            api_wrapper,
            config,
            default_model="gpt-4o-mini",
            max_threads=64,
        )
        
        start_time = time.time()
        order_results, cost = operation._execute_sliding_window_qurk(
            dataset, 
            initial_ordering_method="likert"
        )
        runtime = time.time() - start_time
        
        # Extract ranks and calculate metrics
        ranks = {doc["id"]: doc["_rank"] for doc in order_results}
        doc_ids = list(ground_truth_ranks.keys())
        
        ground_truth_rank_list = [ground_truth_ranks[doc_id] for doc_id in doc_ids]
        method_rank_list = [ranks[doc_id] for doc_id in doc_ids]
        
        # Calculate Kendall's Tau
        tau, p_value = kendalltau(ground_truth_rank_list, method_rank_list)
        
        # Calculate NDCG@50
        y_true = np.zeros(len(dataset))
        y_pred = np.zeros(len(dataset))
        
        for i, doc in enumerate(dataset):
            doc_id = doc["id"]
            # Inverted rank (higher rank = higher relevance)
            y_true[i] = len(dataset) - ground_truth_ranks[doc_id] + 1
            y_pred[i] = len(dataset) - ranks[doc_id] + 1
        
        ndcg50 = calculate_ndcg(y_true, y_pred, k=50)
        ndcg10 = calculate_ndcg(y_true, y_pred, k=10)
        
        results["Likert Sliding Window"].append({
            "batch_size": batch_size,
            "tau": tau,
            "p_value": p_value,
            "ndcg": ndcg50,
            "ndcg10": ndcg10,
            "runtime": runtime,
            "cost": cost
        })
        
        console.print(f"Batch Size: {batch_size}, Tau: {tau:.4f}, NDCG@50: {ndcg50:.4f}, NDCG@10: {ndcg10:.4f}, Cost: ${cost:.4f}")
    
    # Test Picky Window (Likert) with different call budgets
    for rerank_call_budget in rerank_call_budgets:
        console.print(f"[bold blue]Testing Picky Window (Likert) with rerank_call_budget={rerank_call_budget}[/bold blue]")
        
        config = base_config.copy()
        config["batch_size"] = 5  # Fixed batch size
        config["rerank_call_budget"] = rerank_call_budget
        config["initial_ordering_method"] = "likert"
        
        operation = RankOperation(
            api_wrapper,
            config,
            default_model="gpt-4o-mini",
            max_threads=64,
        )
        
        start_time = time.time()
        order_results, cost = operation.execute(dataset)
        runtime = time.time() - start_time
        
        # Extract ranks and calculate metrics
        ranks = {doc["id"]: doc["_rank"] for doc in order_results}
        doc_ids = list(ground_truth_ranks.keys())
        
        ground_truth_rank_list = [ground_truth_ranks[doc_id] for doc_id in doc_ids]
        method_rank_list = [ranks[doc_id] for doc_id in doc_ids]
        
        # Calculate Kendall's Tau
        tau, p_value = kendalltau(ground_truth_rank_list, method_rank_list)
        
        # Calculate NDCG@50
        y_true = np.zeros(len(dataset))
        y_pred = np.zeros(len(dataset))
        
        for i, doc in enumerate(dataset):
            doc_id = doc["id"]
            # Inverted rank (higher rank = higher relevance)
            y_true[i] = len(dataset) - ground_truth_ranks[doc_id] + 1
            y_pred[i] = len(dataset) - ranks[doc_id] + 1
        
        ndcg50 = calculate_ndcg(y_true, y_pred, k=50)
        ndcg10 = calculate_ndcg(y_true, y_pred, k=10)
        
        results["Picky Window (Likert)"].append({
            "rerank_call_budget": rerank_call_budget,
            "tau": tau,
            "p_value": p_value,
            "ndcg": ndcg50,
            "ndcg10": ndcg10,
            "runtime": runtime,
            "cost": cost
        })
        
        console.print(f"Call Budget: {rerank_call_budget}, Tau: {tau:.4f}, NDCG@50: {ndcg50:.4f}, NDCG@10: {ndcg10:.4f}, Cost: ${cost:.4f}")
    
    # Test Picky Window (Calibrated Embedding) with different call budgets
    for rerank_call_budget in rerank_call_budgets:
        console.print(f"[bold blue]Testing Picky Window (Calibrated Embedding) with rerank_call_budget={rerank_call_budget}[/bold blue]")
        
        config = base_config.copy()
        config["batch_size"] = 5  # Fixed batch size
        config["rerank_call_budget"] = rerank_call_budget
        config["initial_ordering_method"] = "calibrated_embedding"
        
        operation = RankOperation(
            api_wrapper,
            config,
            default_model="gpt-4o-mini",
            max_threads=64,
        )
        
        start_time = time.time()
        order_results, cost = operation.execute(dataset)
        runtime = time.time() - start_time
        
        # Extract ranks and calculate metrics
        ranks = {doc["id"]: doc["_rank"] for doc in order_results}
        doc_ids = list(ground_truth_ranks.keys())
        
        ground_truth_rank_list = [ground_truth_ranks[doc_id] for doc_id in doc_ids]
        method_rank_list = [ranks[doc_id] for doc_id in doc_ids]
        
        # Calculate Kendall's Tau
        tau, p_value = kendalltau(ground_truth_rank_list, method_rank_list)
        
        # Calculate NDCG@50
        y_true = np.zeros(len(dataset))
        y_pred = np.zeros(len(dataset))
        
        for i, doc in enumerate(dataset):
            doc_id = doc["id"]
            # Inverted rank (higher rank = higher relevance)
            y_true[i] = len(dataset) - ground_truth_ranks[doc_id] + 1
            y_pred[i] = len(dataset) - ranks[doc_id] + 1
        
        ndcg50 = calculate_ndcg(y_true, y_pred, k=50)
        ndcg10 = calculate_ndcg(y_true, y_pred, k=10)
        
        results["Picky Window (Calibrated Embedding)"].append({
            "rerank_call_budget": rerank_call_budget,
            "tau": tau,
            "p_value": p_value,
            "ndcg": ndcg50,
            "ndcg10": ndcg10,
            "runtime": runtime,
            "cost": cost
        })
        
        console.print(f"Call Budget: {rerank_call_budget}, Tau: {tau:.4f}, NDCG@50: {ndcg50:.4f}, NDCG@10: {ndcg10:.4f}, Cost: ${cost:.4f}")
    
    # Create summary tables
    for method_name, method_results in results.items():
        table = Table(title=f"{method_name} Results")
        
        if method_name.startswith("Picky Window"):
            table.add_column("Call Budget", style="cyan")
        else:
            table.add_column("Batch Size", style="cyan")
            
        table.add_column("Kendall's Tau", justify="right", style="green")
        table.add_column("NDCG@50", justify="right", style="green")
        table.add_column("NDCG@10", justify="right", style="green")
        table.add_column("Cost ($)", justify="right", style="yellow")
        table.add_column("Runtime (s)", justify="right", style="yellow")
        
        for result in method_results:
            if method_name.startswith("Picky Window"):
                table.add_row(
                    str(result["rerank_call_budget"]),
                    f"{result['tau']:.4f}",
                    f"{result['ndcg']:.4f}",
                    f"{result['ndcg10']:.4f}",
                    f"{result['cost']:.4f}",
                    f"{result['runtime']:.2f}"
                )
            else:
                table.add_row(
                    str(result["batch_size"]),
                    f"{result['tau']:.4f}",
                    f"{result['ndcg']:.4f}",
                    f"{result['ndcg10']:.4f}",
                    f"{result['cost']:.4f}",
                    f"{result['runtime']:.2f}"
                )
        
        console.print(table)
    
    return results

def create_budget_performance_plots(results, output_filename="ranking_budget_performance.png"):
    """
    Create scatterplots showing the relationship between cost/runtime and performance metrics.
    
    Args:
        results: Dictionary of results for each method and configuration
        output_filename: Name of the output file for the plots
    """
    # Create figure with six subplots (three metrics vs. cost and runtime)
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    
    # Colors for each method
    colors = {
        "Likert Rating": "blue",
        "Likert Sliding Window": "green", 
        "Picky Window (Likert)": "red",
        "Picky Window (Calibrated Embedding)": "purple"
    }
    
    # Plot Kendall's Tau vs Cost (top left)
    for method_name, method_results in results.items():
        costs = [r["cost"] for r in method_results]
        taus = [r["tau"] for r in method_results]
        
        # Add method name to each point for the legend
        axes[0, 0].scatter(costs, taus, label=method_name, color=colors[method_name], s=70, alpha=0.7)
        
        # Add annotations for batch_size or rerank_call_budget
        for i, result in enumerate(method_results):
            param = result.get("batch_size", result.get("rerank_call_budget"))
            axes[0, 0].annotate(str(param), (costs[i], taus[i]), xytext=(5, 5), 
                         textcoords='offset points', fontsize=8)
    
    axes[0, 0].set_title("Kendall's Tau vs. Cost")
    axes[0, 0].set_xlabel("Cost ($)")
    axes[0, 0].set_ylabel("Kendall's Tau")
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    axes[0, 0].legend()
    
    # Plot NDCG@50 vs Cost (middle left)
    for method_name, method_results in results.items():
        costs = [r["cost"] for r in method_results]
        ndcgs = [r["ndcg"] for r in method_results]
        
        axes[1, 0].scatter(costs, ndcgs, label=method_name, color=colors[method_name], s=70, alpha=0.7)
        
        # Add annotations for batch_size or rerank_call_budget
        for i, result in enumerate(method_results):
            param = result.get("batch_size", result.get("rerank_call_budget"))
            axes[1, 0].annotate(str(param), (costs[i], ndcgs[i]), xytext=(5, 5), 
                          textcoords='offset points', fontsize=8)
    
    axes[1, 0].set_title("NDCG@50 vs. Cost")
    axes[1, 0].set_xlabel("Cost ($)")
    axes[1, 0].set_ylabel("NDCG@50")
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    axes[1, 0].legend()
    
    # Plot NDCG@10 vs Cost (bottom left)
    for method_name, method_results in results.items():
        costs = [r["cost"] for r in method_results]
        ndcgs = [r["ndcg10"] for r in method_results]
        
        axes[2, 0].scatter(costs, ndcgs, label=method_name, color=colors[method_name], s=70, alpha=0.7)
        
        # Add annotations for batch_size or rerank_call_budget
        for i, result in enumerate(method_results):
            param = result.get("batch_size", result.get("rerank_call_budget"))
            axes[2, 0].annotate(str(param), (costs[i], ndcgs[i]), xytext=(5, 5), 
                          textcoords='offset points', fontsize=8)
    
    axes[2, 0].set_title("NDCG@10 vs. Cost")
    axes[2, 0].set_xlabel("Cost ($)")
    axes[2, 0].set_ylabel("NDCG@10")
    axes[2, 0].grid(True, linestyle='--', alpha=0.7)
    axes[2, 0].legend()
    
    # Plot Kendall's Tau vs Runtime (top right)
    for method_name, method_results in results.items():
        runtimes = [r["runtime"] for r in method_results]
        taus = [r["tau"] for r in method_results]
        
        # Add method name to each point for the legend
        axes[0, 1].scatter(runtimes, taus, label=method_name, color=colors[method_name], s=70, alpha=0.7)
        
        # Add annotations for batch_size or rerank_call_budget
        for i, result in enumerate(method_results):
            param = result.get("batch_size", result.get("rerank_call_budget"))
            axes[0, 1].annotate(str(param), (runtimes[i], taus[i]), xytext=(5, 5), 
                         textcoords='offset points', fontsize=8)
    
    axes[0, 1].set_title("Kendall's Tau vs. Runtime")
    axes[0, 1].set_xlabel("Runtime (seconds)")
    axes[0, 1].set_ylabel("Kendall's Tau")
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    axes[0, 1].legend()
    
    # Plot NDCG@50 vs Runtime (middle right)
    for method_name, method_results in results.items():
        runtimes = [r["runtime"] for r in method_results]
        ndcgs = [r["ndcg"] for r in method_results]
        
        axes[1, 1].scatter(runtimes, ndcgs, label=method_name, color=colors[method_name], s=70, alpha=0.7)
        
        # Add annotations for batch_size or rerank_call_budget
        for i, result in enumerate(method_results):
            param = result.get("batch_size", result.get("rerank_call_budget"))
            axes[1, 1].annotate(str(param), (runtimes[i], ndcgs[i]), xytext=(5, 5), 
                          textcoords='offset points', fontsize=8)
    
    axes[1, 1].set_title("NDCG@50 vs. Runtime")
    axes[1, 1].set_xlabel("Runtime (seconds)")
    axes[1, 1].set_ylabel("NDCG@50")
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)
    axes[1, 1].legend()
    
    # Plot NDCG@10 vs Runtime (bottom right)
    for method_name, method_results in results.items():
        runtimes = [r["runtime"] for r in method_results]
        ndcgs = [r["ndcg10"] for r in method_results]
        
        axes[2, 1].scatter(runtimes, ndcgs, label=method_name, color=colors[method_name], s=70, alpha=0.7)
        
        # Add annotations for batch_size or rerank_call_budget
        for i, result in enumerate(method_results):
            param = result.get("batch_size", result.get("rerank_call_budget"))
            axes[2, 1].annotate(str(param), (runtimes[i], ndcgs[i]), xytext=(5, 5), 
                          textcoords='offset points', fontsize=8)
    
    axes[2, 1].set_title("NDCG@10 vs. Runtime")
    axes[2, 1].set_xlabel("Runtime (seconds)")
    axes[2, 1].set_ylabel("NDCG@10")
    axes[2, 1].grid(True, linestyle='--', alpha=0.7)
    axes[2, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_filename)
    console.print(f"[bold green]Budget performance plots saved to {output_filename}[/bold green]")

def run_budget_tests(dataset_name="harmfulness", num_samples=200):
    """
    Run the budget variation tests on a specified dataset.
    
    Args:
        dataset_name (str): Name of the dataset to use ("harmfulness", "synthetic_abstracts", or "medical_pain")
        num_samples (int): Number of samples to use
        
    Returns:
        dict: Results for each method and parameter configuration
    """
    console.print(f"[bold green]Running budget tests on {dataset_name} dataset with {num_samples} samples[/bold green]")
    
    # Get the dir of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Generate output filename based on dataset name
    output_filename = os.path.join(current_dir, f"plots/{dataset_name}_budget_performance.png")
    
    # Make the plots directory if it doesn't exist
    os.makedirs(os.path.join(current_dir, "plots"), exist_ok=True)
    
    # Load the specified dataset
    if dataset_name == "harmfulness":
        data, ground_truth_ranks = load_harmfulness_data(num_samples)
    elif dataset_name == "synthetic_abstracts":
        data, ground_truth_ranks = load_synthetic_abstracts_data(num_samples)
    elif dataset_name == "medical_pain":
        data, ground_truth_ranks = load_medical_data(num_samples)
    else:
        console.print(f"[bold red]Error: Unknown dataset {dataset_name}[/bold red]")
        return {}
    
    # Run tests
    results = test_ranking_with_budget_variation(api_wrapper, data, ground_truth_ranks, dataset_name)
    
    # Create plots
    create_budget_performance_plots(results, output_filename)
    
    return results

if __name__ == "__main__":
    # Default: Run tests on harmfulness dataset
    # run_budget_tests("harmfulness", num_samples=200)
    
    # Run on synthetic abstracts
    # run_budget_tests("synthetic_abstracts", num_samples=200)
    
    # Run on medical transcripts with pain assessment
    run_budget_tests("medical_pain", num_samples=100) 