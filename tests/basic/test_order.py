import pytest
from docetl.operations.order import OrderOperation
from docetl.operations.map import MapOperation
from tests.conftest import api_wrapper as api_wrapper
import json
import time
from scipy.stats import kendalltau  # Import for Kendall's Tau calculation
from rich.console import Console
from rich.table import Table

console = Console()

def test_order_comparison_vs_rating(api_wrapper):
    with open("example_data/debates/data.json", "r") as f:
        data = json.load(f)
        
    # First, create a map operation to extract hostile exchanges
    map_config = {
        "name": "extract_hostile_exchanges",
        "type": "map",
        "output": {
            "schema": {
                "meanness_summary": "str"
            }
        },
        "prompt": """
            Analyze the following debate transcript for {{ input.title }} on {{ input.date }}:

            {{ input.content }}

            Extract and summarize exchanges where candidates are mean, hostile, or disrespectful to each other.
            Focus on:
            - Direct personal attacks and insults
            - Interruptions and talking over each other
            - Sarcasm and condescending remarks
            - Character attacks instead of policy discussions
            - Use of inflammatory language
            - Negative characterizations of opponents
            - Dismissive or disrespectful tone
            - Questioning integrity, honesty, or competence

            Your response should include 3-5 direct quotes that best exemplify the meanest exchanges. The answer should only include the quotes, not any other text.
        """
    }
    
    map_operation = MapOperation(
        api_wrapper,
        map_config,
        default_model="gpt-4o-mini",
        max_threads=64,
    )
    
    # Then create the order operation
    order_config = {
        "name": "order_by_meanness",
        "type": "order",
        "batch_size": 10,
        "prompt": """
            Order these debate transcripts based on how mean or hostile the candidates are to each other.
            Focus on the meanness summaries and examples that have been extracted.
            
            Consider:
            - The overall hostility level rating
            - Severity of personal attacks in the key examples
            - Frequency of hostile exchanges
            - How personal or character-based the attacks are versus policy-based
            - Tone and language used
        """,
        "input_keys": ["meanness_summary", "title", "date"],
        "direction": "desc",
        "verbose": True,
        "bypass_cache": True
    }
    
    order_operation = OrderOperation(
        api_wrapper,
        order_config,
        default_model="gpt-4o-mini",
        max_threads=64,
    )
    
    # Apply the map operation to the data
    map_results, _ = map_operation.execute(data)
    
    # Track execution time and cost for each method
    method_metrics = {}
    
    # Apply the standard order operation
    start_time = time.time()
    order_results, order_cost = order_operation.execute(map_results)
    method_metrics["Ours"] = {
        "runtime": time.time() - start_time,
        "cost": order_cost
    }
    
    # Apply baseline comparison method
    start_time = time.time()
    order_results_baseline, order_cost_baseline = order_operation._execute_comparison_qurk(map_results)
    method_metrics["Baseline Comparison"] = {
        "runtime": time.time() - start_time,
        "cost": order_cost_baseline
    }
    
    # Apply rating based order operation
    start_time = time.time()
    order_results_rating, order_cost_rating = order_operation._execute_rating_embedding_qurk(map_results)
    method_metrics["Embedding Rating"] = {
        "runtime": time.time() - start_time,
        "cost": order_cost_rating
    }
    
    # Apply sliding window order operation with embedding
    start_time = time.time()
    order_results_sliding_window_embedding, order_cost_sliding_window_embedding = order_operation._execute_sliding_window_qurk(map_results, initial_ordering_method="embedding")
    method_metrics["Embedding Sliding Window"] = {
        "runtime": time.time() - start_time,
        "cost": order_cost_sliding_window_embedding
    }
    
    # Apply sliding window order operation with likert
    start_time = time.time()
    order_results_sliding_window_likert, order_cost_sliding_window_likert = order_operation._execute_sliding_window_qurk(map_results, initial_ordering_method="likert")
    method_metrics["Likert Sliding Window"] = {
        "runtime": time.time() - start_time,
        "cost": order_cost_sliding_window_likert
    }
    
    # Apply likert rating order operation
    start_time = time.time()
    order_results_likert_rating, order_cost_likert_rating = order_operation._execute_likert_rating_qurk(map_results)
    method_metrics["Likert Rating"] = {
        "runtime": time.time() - start_time,
        "cost": order_cost_likert_rating
    }

    # Extract ranks for each document from the different methods
    # Create dictionaries mapping document IDs to ranks
    baseline_ranks = {doc["id"]: doc["_rank"] for i, doc in enumerate(order_results_baseline)}
    
    # Extract ranks from standard order results and rating-based results
    standard_ranks = {doc["id"]: doc["_rank"] for i, doc in enumerate(order_results)}
    rating_ranks = {doc["id"]: doc["_rank"] for i, doc in enumerate(order_results_rating)}
    sliding_window_embedding_ranks = {doc["id"]: doc["_rank"] for i, doc in enumerate(order_results_sliding_window_embedding)}
    sliding_window_likert_ranks = {doc["id"]: doc["_rank"] for i, doc in enumerate(order_results_sliding_window_likert)}
    likert_rating_ranks = {doc["id"]: doc["_rank"] for i, doc in enumerate(order_results_likert_rating)}
    
    # Prepare lists for Kendall's Tau computation
    # We need to ensure the documents are in the same order for both rankings
    doc_ids = list(baseline_ranks.keys())
    
    baseline_rank_list = [baseline_ranks[doc_id] for doc_id in doc_ids]
    standard_rank_list = [standard_ranks[doc_id] for doc_id in doc_ids]
    rating_rank_list = [rating_ranks[doc_id] for doc_id in doc_ids]
    sliding_window_embedding_rank_list = [sliding_window_embedding_ranks[doc_id] for doc_id in doc_ids]
    sliding_window_likert_rank_list = [sliding_window_likert_ranks[doc_id] for doc_id in doc_ids]
    likert_rating_rank_list = [likert_rating_ranks[doc_id] for doc_id in doc_ids]
    
    # Compute Kendall's Tau correlation coefficients
    tau_standard, p_value_standard = kendalltau(baseline_rank_list, standard_rank_list)
    tau_rating, p_value_rating = kendalltau(baseline_rank_list, rating_rank_list)
    tau_sliding_window_embedding, p_value_sliding_window_embedding = kendalltau(baseline_rank_list, sliding_window_embedding_rank_list)
    tau_sliding_window_likert, p_value_sliding_window_likert = kendalltau(baseline_rank_list, sliding_window_likert_rank_list)
    tau_likert_rating, p_value_likert_rating = kendalltau(baseline_rank_list, likert_rating_rank_list)
    
    # Store results in a list of tuples (method_name, tau, p_value, runtime, cost) for sorting
    results = [
        ("Ours", tau_standard, p_value_standard, method_metrics["Ours"]["runtime"], method_metrics["Ours"]["cost"]),
        ("Embedding Rating", tau_rating, p_value_rating, method_metrics["Embedding Rating"]["runtime"], method_metrics["Embedding Rating"]["cost"]),
        ("Embedding Sliding Window", tau_sliding_window_embedding, p_value_sliding_window_embedding, method_metrics["Embedding Sliding Window"]["runtime"], method_metrics["Embedding Sliding Window"]["cost"]),
        ("Likert Sliding Window", tau_sliding_window_likert, p_value_sliding_window_likert, method_metrics["Likert Sliding Window"]["runtime"], method_metrics["Likert Sliding Window"]["cost"]),
        ("Likert Rating", tau_likert_rating, p_value_likert_rating, method_metrics["Likert Rating"]["runtime"], method_metrics["Likert Rating"]["cost"])
    ]
    
    # Sort results by Kendall's tau value in descending order
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    # Create a Rich table
    table = Table(title="Debate Meanness Ordering Results (vs Baseline)")
    
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
    
    # Assert that the correlations are reasonably high (optional)
    # This threshold can be adjusted based on expected correlation
    assert tau_standard > -1.0, "Ours ordering should have some correlation with baseline"
    assert tau_rating > -1.0, "Rating-based ordering should have some correlation with baseline"
    assert tau_sliding_window_embedding > -1.0, "Embedding sliding window ordering should have some correlation with baseline"
    assert tau_sliding_window_likert > -1.0, "Likert sliding window ordering should have some correlation with baseline"
    assert tau_likert_rating > -1.0, "Likert rating ordering should have some correlation with baseline"
    
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
        size = 20 + (3 * i)
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
        "bypass_cache": True
    }
    
    order_operation = OrderOperation(
        api_wrapper,
        order_config,
        default_model="gpt-4o-mini",
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
    method_metrics["Ours"] = {
        "runtime": time.time() - start_time,
        "cost": order_cost
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
    
    # Extract ranks for each document from the different methods
    standard_ranks = {doc["id"]: doc["_rank"] for doc in order_results}
    baseline_ranks = {doc["id"]: doc["_rank"] for doc in order_results_baseline}
    rating_ranks = {doc["id"]: doc["_rank"] for doc in order_results_rating}
    sliding_window_embedding_ranks = {doc["id"]: doc["_rank"] for doc in order_results_sliding_window_embedding}
    sliding_window_likert_ranks = {doc["id"]: doc["_rank"] for doc in order_results_sliding_window_likert}
    likert_rating_ranks = {doc["id"]: doc["_rank"] for doc in order_results_likert_rating}
    
    # Prepare lists for Kendall's Tau computation
    doc_ids = list(ground_truth_ranks.keys())
    
    ground_truth_rank_list = [ground_truth_ranks[doc_id] for doc_id in doc_ids]
    standard_rank_list = [standard_ranks[doc_id] for doc_id in doc_ids]
    baseline_rank_list = [baseline_ranks[doc_id] for doc_id in doc_ids]
    rating_rank_list = [rating_ranks[doc_id] for doc_id in doc_ids]
    sliding_window_embedding_rank_list = [sliding_window_embedding_ranks[doc_id] for doc_id in doc_ids]
    sliding_window_likert_rank_list = [sliding_window_likert_ranks[doc_id] for doc_id in doc_ids]
    likert_rating_rank_list = [likert_rating_ranks[doc_id] for doc_id in doc_ids]
    
    # Compute Kendall's Tau correlation coefficients against ground truth
    tau_standard, p_value_standard = kendalltau(ground_truth_rank_list, standard_rank_list)
    tau_baseline, p_value_baseline = kendalltau(ground_truth_rank_list, baseline_rank_list)
    tau_rating, p_value_rating = kendalltau(ground_truth_rank_list, rating_rank_list)
    tau_sliding_window_embedding, p_value_sliding_window_embedding = kendalltau(ground_truth_rank_list, sliding_window_embedding_rank_list)
    tau_sliding_window_likert, p_value_sliding_window_likert = kendalltau(ground_truth_rank_list, sliding_window_likert_rank_list)
    tau_likert_rating, p_value_likert_rating = kendalltau(ground_truth_rank_list, likert_rating_rank_list)
    
    # Store results in a list of tuples for sorting, including runtime and cost
    results = [
        ("Ours", tau_standard, p_value_standard, method_metrics["Ours"]["runtime"], method_metrics["Ours"]["cost"]),
        ("Baseline Comparison", tau_baseline, p_value_baseline, method_metrics["Baseline Comparison"]["runtime"], method_metrics["Baseline Comparison"]["cost"]),
        ("Embedding Rating", tau_rating, p_value_rating, method_metrics["Embedding Rating"]["runtime"], method_metrics["Embedding Rating"]["cost"]),
        ("Embedding Sliding Window", tau_sliding_window_embedding, p_value_sliding_window_embedding, method_metrics["Embedding Sliding Window"]["runtime"], method_metrics["Embedding Sliding Window"]["cost"]),
        ("Likert Sliding Window", tau_sliding_window_likert, p_value_sliding_window_likert, method_metrics["Likert Sliding Window"]["runtime"], method_metrics["Likert Sliding Window"]["cost"]),
        ("Likert Rating", tau_likert_rating, p_value_likert_rating, method_metrics["Likert Rating"]["runtime"], method_metrics["Likert Rating"]["cost"])
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
            Order these medical transcripts based on how much pain the patient is experiencing or reporting.
            
            Consider:
            - Explicit mentions of pain levels
            - Description of symptoms that indicate pain
            - Patient's language and tone when describing discomfort
            - Impact of pain on daily activities
            - Frequency and duration of pain episodes
            - Phrases like "it hurts", "in pain", "suffering", "can't bear it"
            - Medications or treatments being prescribed for pain
        """,
        "input_keys": ["src"],
        "direction": "desc",  # Highest pain first
        "verbose": True,
        "bypass_cache": True
    }
    
    order_operation = OrderOperation(
        api_wrapper,
        order_config,
        default_model="gpt-4o-mini",
        max_threads=64,
    )
    
    # Track execution time and cost for each method
    method_metrics = {}
    
    # Apply standard order operation
    start_time = time.time()
    order_results, order_cost = order_operation.execute(medical_data)
    method_metrics["Ours"] = {
        "runtime": time.time() - start_time,
        "cost": order_cost
    }
    
    # Apply baseline comparison method
    start_time = time.time()
    order_results_baseline, order_cost_baseline = order_operation._execute_comparison_qurk(medical_data)
    method_metrics["Baseline Comparison"] = {
        "runtime": time.time() - start_time,
        "cost": order_cost_baseline
    }
    
    # Apply rating based order operation
    start_time = time.time()
    order_results_rating, order_cost_rating = order_operation._execute_rating_embedding_qurk(medical_data)
    method_metrics["Embedding Rating"] = {
        "runtime": time.time() - start_time,
        "cost": order_cost_rating
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
    
    # Extract ranks for each document from the different methods
    baseline_ranks = {doc["id"]: doc["_rank"] for doc in order_results_baseline}
    standard_ranks = {doc["id"]: doc["_rank"] for doc in order_results}
    rating_ranks = {doc["id"]: doc["_rank"] for doc in order_results_rating}
    sliding_window_embedding_ranks = {doc["id"]: doc["_rank"] for doc in order_results_sliding_window_embedding}
    sliding_window_likert_ranks = {doc["id"]: doc["_rank"] for doc in order_results_sliding_window_likert}
    likert_rating_ranks = {doc["id"]: doc["_rank"] for doc in order_results_likert_rating}
    
    # Prepare lists for Kendall's Tau computation
    # We need to ensure the documents are in the same order for both rankings
    doc_ids = list(baseline_ranks.keys())
    
    baseline_rank_list = [baseline_ranks[doc_id] for doc_id in doc_ids]
    standard_rank_list = [standard_ranks[doc_id] for doc_id in doc_ids]
    rating_rank_list = [rating_ranks[doc_id] for doc_id in doc_ids]
    sliding_window_embedding_rank_list = [sliding_window_embedding_ranks[doc_id] for doc_id in doc_ids]
    sliding_window_likert_rank_list = [sliding_window_likert_ranks[doc_id] for doc_id in doc_ids]
    likert_rating_rank_list = [likert_rating_ranks[doc_id] for doc_id in doc_ids]
    
    # Compute Kendall's Tau correlation coefficients against baseline
    tau_standard, p_value_standard = kendalltau(baseline_rank_list, standard_rank_list)
    tau_rating, p_value_rating = kendalltau(baseline_rank_list, rating_rank_list)
    tau_sliding_window_embedding, p_value_sliding_window_embedding = kendalltau(baseline_rank_list, sliding_window_embedding_rank_list)
    tau_sliding_window_likert, p_value_sliding_window_likert = kendalltau(baseline_rank_list, sliding_window_likert_rank_list)
    tau_likert_rating, p_value_likert_rating = kendalltau(baseline_rank_list, likert_rating_rank_list)
    
    # Store results in a list of tuples for sorting, including runtime and cost
    results = [
        ("Ours", tau_standard, p_value_standard, method_metrics["Ours"]["runtime"], method_metrics["Ours"]["cost"]),
        ("Embedding Rating", tau_rating, p_value_rating, method_metrics["Embedding Rating"]["runtime"], method_metrics["Embedding Rating"]["cost"]),
        ("Embedding Sliding Window", tau_sliding_window_embedding, p_value_sliding_window_embedding, method_metrics["Embedding Sliding Window"]["runtime"], method_metrics["Embedding Sliding Window"]["cost"]),
        ("Likert Sliding Window", tau_sliding_window_likert, p_value_sliding_window_likert, method_metrics["Likert Sliding Window"]["runtime"], method_metrics["Likert Sliding Window"]["cost"]),
        ("Likert Rating", tau_likert_rating, p_value_likert_rating, method_metrics["Likert Rating"]["runtime"], method_metrics["Likert Rating"]["cost"])
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
    top_pain_transcripts = sorted(order_results_baseline, key=lambda x: x["_rank"], reverse=True)[:3]
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
    
    # Assert that the correlations are reasonably high
    assert tau_standard > -1.0, "Ours ordering should have some correlation with baseline"
    assert tau_rating > -1.0, "Rating-based ordering should have some correlation with baseline"
    assert tau_sliding_window_embedding > -1.0, "Embedding sliding window ordering should have some correlation with baseline"
    assert tau_sliding_window_likert > -1.0, "Likert sliding window ordering should have some correlation with baseline"
    assert tau_likert_rating > -1.0, "Likert rating ordering should have some correlation with baseline"
