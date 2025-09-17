#!/usr/bin/env python3
"""
Run Pareto frontier plans on test datasets.

This script:
1. Pulls pareto frontier JSON files from Modal volume for each method (simple, baseline, mcts)
2. For each frontier point, gets the corresponding YAML pipeline file
3. Modifies the YAML to use test data instead of train data
4. Runs the pipeline and calculates cost/accuracy
5. Saves results to a test_logs.json file
"""

import json
import yaml
import traceback
from pathlib import Path
from typing import Any, Dict, List
import modal
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import time

from docetl.runner import DSLRunner
from experiments.reasoning.utils import app, volume, VOLUME_MOUNT_PATH, image
from experiments.reasoning.evaluation.utils import dataset_accuracy_metrics, get_evaluate_func


def plot_matrix(matrix_data: Dict[str, Any], matrix_type: str, dataset: str, output_dir: Path):
    """
    Create a beautiful matrix visualization using the same styling as plot_matrix.py
    
    Args:
        matrix_data: Dictionary containing matrix and method_info
        matrix_type: Type of matrix ('best_cost_savings', 'avg_cost_savings', 'coverage')
        dataset: Dataset name
        output_dir: Output directory for saving the plot
    """
    matrix = matrix_data["matrix"]
    methods = list(matrix.keys())
    
    if not methods:
        print(f"No methods found for {matrix_type} matrix")
        return
    
    # Create DataFrame
    df = pd.DataFrame(matrix, index=methods, columns=methods)
    
    # Find ALL numeric values and calculate color intensity
    numeric_positions = []
    numeric_values = []
    for i in range(len(methods)):
        for j in range(len(methods)):
            val = df.iloc[i, j]
            if isinstance(val, (int, float)) and val != '--':
                numeric_positions.append((i, j, val))
                numeric_values.append(abs(val))
    
    # Calculate max absolute value for normalization
    max_abs_value = max(numeric_values) if numeric_values else 1
    
    def get_color_intensity(value, max_val):
        """Calculate color intensity with more vibrant gradient"""
        intensity = abs(value) / max_val
        # Use a moderate curve for balanced colors
        intensity = intensity ** 0.6  # Moderate curve
        # Allow more color intensity while keeping it elegant
        return max(0.2, min(0.8, intensity))
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # More vibrant but still elegant color scheme
    base_colors = {
        'positive': np.array([74, 222, 128]) / 255,      # Fresh green
        'negative': np.array([248, 113, 113]) / 255,    # Warm red
        'diagonal': np.array([249, 250, 251]) / 255,    # Very light gray
        'unable': np.array([243, 244, 246]) / 255,      # Light gray
        'none': np.array([243, 244, 246]) / 255         # Light gray
    }
    
    # Draw each cell individually
    for i in range(len(methods)):
        for j in range(len(methods)):
            val = df.iloc[i, j]
            
            # Determine cell color and intensity
            if isinstance(val, (int, float)):
                intensity = get_color_intensity(val, max_abs_value)
                
                if val > 0:
                    # Vibrant green gradient
                    base_color = base_colors['positive']
                    # Create more vibrant gradient
                    color = base_color * (0.4 + 0.6 * intensity)
                    # Add less white tint for more color
                    color = color + (1 - color) * (1 - intensity) * 0.3
                    text_color = 'black'  # Always black for better readability
                else:
                    # Vibrant red gradient
                    base_color = base_colors['negative']
                    # Create more vibrant gradient
                    color = base_color * (0.4 + 0.6 * intensity)
                    # Add less white tint for more color
                    color = color + (1 - color) * (1 - intensity) * 0.3
                    text_color = 'black'  # Always black for better readability
                text = f'{val:.2f}'
                
            elif val == '--':
                color = base_colors['diagonal']
                text_color = 'gray'
                text = '--'
            elif val == 'Unable':
                color = base_colors['unable']
                text_color = 'gray'
                text = 'Unable'
            elif val == 'None':
                color = base_colors['none']
                text_color = 'gray'
                text = 'None'
            else:
                color = 'white'
                text_color = 'black'
                text = str(val)
            
            # Draw rectangle with subtle border
            rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='white', linewidth=1.5)
            ax.add_patch(rect)
            
            # Add text with larger font
            fontweight = 'bold' if isinstance(val, (int, float)) else 'normal'
            fontsize = 14 if isinstance(val, (int, float)) else 12
            ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', 
                    fontsize=fontsize, fontweight=fontweight, color=text_color)
    
    # Set up the plot
    ax.set_xlim(0, len(methods))
    ax.set_ylim(0, len(methods))
    ax.set_aspect('equal')
    
    # Set ticks and labels with larger font
    ax.set_xticks(np.arange(len(methods)) + 0.5)
    ax.set_yticks(np.arange(len(methods)) + 0.5)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=13)
    ax.set_yticklabels(methods, fontsize=13)
    
    # Invert y-axis to match matrix convention
    ax.invert_yaxis()
    
    # Remove outer frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Title with larger font
    title_map = {
        'best_cost_savings': 'Best Cost Savings Matrix',
        'avg_cost_savings': 'Average Cost Savings Matrix', 
        'coverage': 'Coverage Matrix'
    }
    plt.title(f'{title_map.get(matrix_type, matrix_type)} - {dataset}', 
              fontsize=24, fontweight='bold', pad=30, color='#1f2937')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = output_dir / f"{dataset}_{matrix_type}_matrix.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Matrix plot saved: {plot_path}")
    
    # Summary statistics
    if numeric_values:
        numeric_vals = [val for i, j, val in numeric_positions]
        print(f"   Max value: {max(numeric_vals):.2f}")
        print(f"   Min value: {min(numeric_vals):.2f}")
        print(f"   Mean: {np.mean(numeric_vals):.2f}")

# Dataset configurations
DATASETS = ["cuad", "blackvault", "game_reviews", "sustainability", "biodex", "medec", "facility"]
METHODS = ["simple_baseline", "baseline", "mcts"]

def calculate_coverage_matrix(all_points: Dict[str, List[Dict[str, float]]]) -> Dict[str, Any]:
    """
    Calculate a matrix of coverage between all methods (excluding original).
    Each cell (method A, method B) shows the fraction of plans in Method B that are 
    completely dominated by plans in Method A (ignoring plans with acc <= original).
    Complete domination means lower cost AND higher accuracy.
    
    Args:
        all_points: Dictionary with method names as keys and lists of {cost, accuracy} dicts as values
        
    Returns:
        Dictionary containing the coverage matrix
    """

    print("="*80)
    print(f"Calculating coverage matrix for {all_points}")

    # Get all methods except original
    methods = [m for m in all_points.keys() if m != "original" and all_points[m]]
    
    if len(methods) < 2:
        return {
            "error": "Need at least 2 methods (excluding original) for matrix comparison"
        }
    
    # Get original accuracy for reference
    original_points = all_points.get("original", [])
    original_best_accuracy = None
    if original_points:
        original_valid_points = [p for p in original_points if p["accuracy"] is not None]
        if original_valid_points:
            original_best_accuracy = max(point["accuracy"] for point in original_valid_points)
    
    # Initialize matrix
    matrix = {}
    method_info = {}
    
    # Calculate info for each method
    for method in methods:
        method_points = all_points[method]
        valid_points = [p for p in method_points if p["accuracy"] is not None and p["cost"] is not None]
        
        if not valid_points:
            method_info[method] = {
                "valid_points": 0,
                "points_above_original": 0
            }
            continue
        
        # Count points above original accuracy
        points_above_original = 0
        if original_best_accuracy is not None:
            points_above_original = len([p for p in valid_points if p["accuracy"] > original_best_accuracy])
        
        method_info[method] = {
            "valid_points": len(valid_points),
            "points_above_original": points_above_original
        }
    
    # Calculate matrix values
    for method_a in methods:
        matrix[method_a] = {}
        
        for method_b in methods:
            if method_a == method_b:
                matrix[method_a][method_b] = "--"  # Same method (diagonal)
                continue
            
            # Get method B's points
            method_b_points = all_points[method_b]
            method_b_valid_points = [p for p in method_b_points if p["accuracy"] is not None and p["cost"] is not None]
            
            if not method_b_valid_points:
                matrix[method_a][method_b] = "Unable"
                continue
            
            # Filter method B points to only those above original accuracy
            if original_best_accuracy is not None:
                method_b_above_original = [p for p in method_b_valid_points if p["accuracy"] > original_best_accuracy]
            else:
                method_b_above_original = method_b_valid_points
            
            if not method_b_above_original:
                matrix[method_a][method_b] = "None"  # No plans above original accuracy
                continue
            
            # Get method A's points
            method_a_points = all_points[method_a]
            method_a_valid_points = [p for p in method_a_points if p["accuracy"] is not None and p["cost"] is not None]
            
            if not method_a_valid_points:
                matrix[method_a][method_b] = "Unable"
                continue
            
            # Calculate coverage: fraction of method B plans that are completely dominated by method A
            dominated_count = 0
            
            for method_b_point in method_b_above_original:
                method_b_cost = method_b_point["cost"]
                method_b_accuracy = method_b_point["accuracy"]
                
                # Check if this method B point is completely dominated by any method A point
                is_dominated = False
                for method_a_point in method_a_valid_points:
                    method_a_cost = method_a_point["cost"]
                    method_a_accuracy = method_a_point["accuracy"]
                    
                    # Complete domination: lower cost AND higher accuracy
                    if method_a_cost < method_b_cost and method_a_accuracy > method_b_accuracy:
                        is_dominated = True
                        break
                
                if is_dominated:
                    dominated_count += 1
            
            # Calculate coverage fraction
            coverage_fraction = dominated_count / len(method_b_above_original)
            matrix[method_a][method_b] = round(coverage_fraction, 3)
    
    return {
        "matrix": matrix,
        "methods": methods,
        "method_info": method_info,
        "original_best_accuracy": original_best_accuracy
    }


def calculate_avg_cost_savings_matrix(all_points: Dict[str, List[Dict[str, float]]]) -> Dict[str, Any]:
    """
    Calculate a matrix of average cost savings between all methods (excluding original).
    Each cell (method A, method B) shows the average cost savings when method A achieves 
    the same or higher accuracy as each plan in method B (ignoring plans with acc <= original).
    
    Args:
        all_points: Dictionary with method names as keys and lists of {cost, accuracy} dicts as values
        
    Returns:
        Dictionary containing the average cost savings matrix
    """
    # Get all methods except original
    methods = [m for m in all_points.keys() if m != "original" and all_points[m]]
    
    if len(methods) < 2:
        return {
            "error": "Need at least 2 methods (excluding original) for matrix comparison"
        }
    
    # Get original accuracy for reference
    original_points = all_points.get("original", [])
    original_best_accuracy = None
    if original_points:
        original_valid_points = [p for p in original_points if p["accuracy"] is not None]
        if original_valid_points:
            original_best_accuracy = max(point["accuracy"] for point in original_valid_points)
    
    # Initialize matrix
    matrix = {}
    method_info = {}
    
    # Calculate info for each method
    for method in methods:
        method_points = all_points[method]
        valid_points = [p for p in method_points if p["accuracy"] is not None and p["cost"] is not None]
        
        if not valid_points:
            method_info[method] = {
                "valid_points": 0,
                "points_above_original": 0
            }
            continue
        
        # Count points above original accuracy
        points_above_original = 0
        if original_best_accuracy is not None:
            points_above_original = len([p for p in valid_points if p["accuracy"] > original_best_accuracy])
        
        method_info[method] = {
            "valid_points": len(valid_points),
            "points_above_original": points_above_original
        }
    
    # Calculate matrix values
    for method_a in methods:
        matrix[method_a] = {}
        
        for method_b in methods:
            if method_a == method_b:
                matrix[method_a][method_b] = "--"  # Same method (diagonal)
                continue
            
            # Get method B's points
            method_b_points = all_points[method_b]
            method_b_valid_points = [p for p in method_b_points if p["accuracy"] is not None and p["cost"] is not None]
            
            if not method_b_valid_points:
                matrix[method_a][method_b] = "Unable"
                continue
            
            # Filter method B points to only those above original accuracy
            if original_best_accuracy is not None:
                method_b_above_original = [p for p in method_b_valid_points if p["accuracy"] > original_best_accuracy]
            else:
                method_b_above_original = method_b_valid_points
            
            if not method_b_above_original:
                matrix[method_a][method_b] = "None"  # No plans above original accuracy
                continue
            
            # Get method A's points
            method_a_points = all_points[method_a]
            method_a_valid_points = [p for p in method_a_points if p["accuracy"] is not None and p["cost"] is not None]
            
            if not method_a_valid_points:
                matrix[method_a][method_b] = "Unable"
                continue
            
            # Calculate cost savings for each method B plan
            cost_savings_list = []
            
            for method_b_point in method_b_above_original:
                method_b_accuracy = method_b_point["accuracy"]
                method_b_cost = method_b_point["cost"]
                
                # Find method A's cheapest plan that meets or exceeds this accuracy
                qualifying_method_a_points = [p for p in method_a_valid_points if p["accuracy"] >= method_b_accuracy]
                
                if qualifying_method_a_points:
                    method_a_cheapest = min(qualifying_method_a_points, key=lambda x: x["cost"])
                    savings = method_b_cost - method_a_cheapest["cost"]
                    cost_savings_list.append(savings)
            
            if not cost_savings_list:
                matrix[method_a][method_b] = "Unable"
                continue
            
            # Calculate average cost savings
            avg_cost_savings = sum(cost_savings_list) / len(cost_savings_list)
            matrix[method_a][method_b] = round(avg_cost_savings, 2)
    
    return {
        "matrix": matrix,
        "methods": methods,
        "method_info": method_info,
        "original_best_accuracy": original_best_accuracy
    }


def calculate_best_cost_savings_matrix(all_points: Dict[str, List[Dict[str, float]]]) -> Dict[str, Any]:
    """
    Calculate a matrix of cost savings between all methods (excluding original).
    Each cell (method A, method B) shows how much cost method A saves for achieving 
    or surpassing the highest accuracy of method B.
    
    Args:
        all_points: Dictionary with method names as keys and lists of {cost, accuracy} dicts as values
        
    Returns:
        Dictionary containing the cost savings matrix
    """
    
    print("="*80)
    print(f"Calculating best cost savings matrix for {all_points}")
    # Get all methods except original
    methods = [m for m in all_points.keys() if m != "original" and all_points[m]]
    
    if len(methods) < 2:
        return {
            "error": "Need at least 2 methods (excluding original) for matrix comparison"
        }
    
    # Get original accuracy for reference
    original_points = all_points.get("original", [])
    original_best_accuracy = None
    if original_points:
        original_valid_points = [p for p in original_points if p["accuracy"] is not None]
        if original_valid_points:
            original_best_accuracy = max(point["accuracy"] for point in original_valid_points)
    
    # Initialize matrix
    matrix = {}
    method_info = {}
    
    # Calculate best accuracy for each method
    for method in methods:
        method_points = all_points[method]
        valid_points = [p for p in method_points if p["accuracy"] is not None]
        
        if not valid_points:
            method_info[method] = {
                "best_accuracy": None,
                "best_cost": None,
                "valid_points": 0,
                "points_above_original": 0
            }
            continue
        
        best_accuracy = max(point["accuracy"] for point in valid_points)
        # Find the cheapest plan at best accuracy
        best_accuracy_points = [p for p in valid_points if p["accuracy"] == best_accuracy and p["cost"] is not None]
        best_cost = min(point["cost"] for point in best_accuracy_points) if best_accuracy_points else None
        
        # Calculate points above original accuracy
        points_above_original = 0
        if original_best_accuracy is not None:
            points_above_original = len([p for p in valid_points if p["accuracy"] > original_best_accuracy])
        
        method_info[method] = {
            "best_accuracy": best_accuracy,
            "best_cost": best_cost,
            "valid_points": len(valid_points),
            "points_above_original": points_above_original
        }
    
    # Calculate matrix values
    for method_a in methods:
        matrix[method_a] = {}
        
        for method_b in methods:
            if method_a == method_b:
                matrix[method_a][method_b] = "--"  # Same method (diagonal)
                continue
            
            # Get method B's best accuracy
            method_b_info = method_info[method_b]
            method_b_best_accuracy = method_b_info["best_accuracy"]
            
            if method_b_best_accuracy is None:
                matrix[method_a][method_b] = None
                continue
            
            # Check if method B's best accuracy is lower than original
            if original_best_accuracy is not None and method_b_best_accuracy < original_best_accuracy:
                matrix[method_a][method_b] = "None"
                continue
            
            # Get method A's points
            method_a_points = all_points[method_a]
            method_a_valid_points = [p for p in method_a_points if p["accuracy"] is not None and p["cost"] is not None]
            
            if not method_a_valid_points:
                matrix[method_a][method_b] = "Unable"
                continue
            
            # Find method A's cheapest plan that meets or exceeds method B's best accuracy
            qualifying_points = [p for p in method_a_valid_points if p["accuracy"] >= method_b_best_accuracy]
            
            if not qualifying_points:
                matrix[method_a][method_b] = "Unable"
                continue
            
            method_a_cheapest = min(qualifying_points, key=lambda x: x["cost"])
            method_b_best_cost = method_b_info["best_cost"]
            
            if method_b_best_cost is None:
                matrix[method_a][method_b] = "Unable"
                continue
            
            # Calculate cost savings
            cost_savings = method_b_best_cost - method_a_cheapest["cost"]
            matrix[method_a][method_b] = round(cost_savings, 2)
    
    return {
        "matrix": matrix,
        "methods": methods,
        "method_info": method_info,
        "original_best_accuracy": original_best_accuracy
    }


def calculate_comparison_metrics(all_points: Dict[str, List[Dict[str, float]]]) -> Dict[str, Any]:
    """
    Calculate three key comparison metrics between MCTS and each other method individually.
    
    Args:
        all_points: Dictionary with method names as keys and lists of {cost, accuracy} dicts as values
        
    Returns:
        Dictionary containing pairwise comparison metrics for each method vs MCTS
    """
    metrics = {}
    
    # Define our method vs other systems
    our_method = "mcts"  # Only MCTS is our method
    other_systems = ["original", "simple_baseline", "baseline", "lotus", "pz_direct", "pz_retrieval"]
    
    # Filter out methods with no data
    if our_method not in all_points or not all_points[our_method]:
        return {
            "error": "No MCTS data available for comparison"
        }
    
    other_systems = [m for m in other_systems if m in all_points and all_points[m]]
    
    if not other_systems:
        return {
            "error": "No other systems data available for comparison"
        }
    
    # Get our points
    our_points = all_points[our_method]
    our_valid_points = [p for p in our_points if p["accuracy"] is not None]
    
    if not our_valid_points:
        return {
            "error": "No valid MCTS accuracy data found"
        }
    
    our_best_accuracy = max(point["accuracy"] for point in our_valid_points)
    
    # Calculate pairwise metrics for each other system
    pairwise_metrics = {}
    
    for other_method in other_systems:
        other_points = all_points[other_method]
        other_valid_points = [p for p in other_points if p["accuracy"] is not None]
        
        if not other_valid_points:
            pairwise_metrics[other_method] = {
                "error": f"No valid accuracy data found for {other_method}"
            }
            continue
        
        other_best_accuracy = max(point["accuracy"] for point in other_valid_points)
        
        # 1. Calculate accuracy improvement: how much more accurate our best is vs their best
        accuracy_improvement = our_best_accuracy - other_best_accuracy
        accuracy_improvement_pct = (accuracy_improvement / other_best_accuracy) * 100 if other_best_accuracy > 0 else 0
        
        # 2. Calculate cost savings for plans within 10% accuracy of their best
        target_accuracy = other_best_accuracy * 0.9  # 10% below their best
        
        # Find our cheapest plan that meets or exceeds this accuracy (filter None values)
        qualifying_our_plans = [p for p in our_valid_points if p["accuracy"] >= target_accuracy and p["cost"] is not None]
        
        cost_savings_within_10pct = None
        if qualifying_our_plans:
            our_cheapest_at_target = min(qualifying_our_plans, key=lambda x: x["cost"])
            
            # Find their cheapest plan at or above their best accuracy (filter None values)
            qualifying_other_plans = [p for p in other_valid_points if p["accuracy"] >= other_best_accuracy and p["cost"] is not None]
            if qualifying_other_plans:
                other_cheapest_at_best = min(qualifying_other_plans, key=lambda x: x["cost"])
                
                cost_savings = other_cheapest_at_best["cost"] - our_cheapest_at_target["cost"]
                cost_savings_pct = (cost_savings / other_cheapest_at_best["cost"]) * 100 if other_cheapest_at_best["cost"] > 0 else 0
                
                cost_savings_within_10pct = {
                    "absolute": cost_savings,
                    "percentage": cost_savings_pct,
                    "our_cost": our_cheapest_at_target["cost"],
                    "our_accuracy": our_cheapest_at_target["accuracy"],
                    "other_cost": other_cheapest_at_best["cost"],
                    "other_accuracy": other_cheapest_at_best["accuracy"],
                    "target_accuracy": target_accuracy
                }
        
        # 3. Calculate average cost savings: for each other system plan P, find our cheapest plan 
        # that meets P's accuracy, then calculate average savings
        cost_savings_list = []
        
        for other_point in other_valid_points:
            other_accuracy = other_point["accuracy"]
            other_cost = other_point["cost"]
            
            # Skip if cost is None
            if other_cost is None:
                continue
                
            # Find our cheapest plan that meets or exceeds this accuracy (filter None values)
            qualifying_our_plans = [p for p in our_valid_points if p["accuracy"] >= other_accuracy and p["cost"] is not None]
            
            if qualifying_our_plans:
                our_cheapest = min(qualifying_our_plans, key=lambda x: x["cost"])
                savings = other_cost - our_cheapest["cost"]
                cost_savings_list.append(savings)
        
        average_cost_savings = None
        if cost_savings_list:
            avg_cost_savings = sum(cost_savings_list) / len(cost_savings_list)
            # Calculate percentage based on valid points only
            valid_other_points = [p for p in other_valid_points if p["cost"] is not None]
            avg_cost_savings_pct = sum(savings / valid_other_points[i]["cost"] * 100 
                                     for i, savings in enumerate(cost_savings_list)) / len(cost_savings_list)
            
            average_cost_savings = {
                "absolute": avg_cost_savings,
                "percentage": avg_cost_savings_pct,
                "comparisons_count": len(cost_savings_list),
                "individual_savings": cost_savings_list
            }
        
        # Store pairwise metrics
        pairwise_metrics[other_method] = {
            "accuracy_improvement": {
                "absolute": accuracy_improvement,
                "percentage": accuracy_improvement_pct,
                "our_best": our_best_accuracy,
                "other_best": other_best_accuracy
            },
            "cost_savings_within_10pct": cost_savings_within_10pct,
            "average_cost_savings": average_cost_savings
        }
    
    return {
        "pairwise_comparisons": pairwise_metrics,
        "our_method": our_method,
        "other_methods": other_systems
    }

@app.function(
    image=image, 
    secrets=[modal.Secret.from_dotenv()], 
    volumes={VOLUME_MOUNT_PATH: volume}, 
    timeout=60 * 60
)
def run_test_frontier_remote(dataset: str, method: str) -> Dict[str, Any]:
    """
    Run all Pareto frontier plans for a given dataset and method on test data.
    
    Args:
        dataset: Dataset name (e.g., 'cuad')
        method: Method name ('simple_baseline', 'baseline', or 'mcts')
    
    Returns:
        Dictionary with test results for all frontier points
    """
    try:
        print(f"\n{'='*60}")
        print(f"Running test frontier for {dataset} - {method}")
        print(f"{'='*60}\n")
        
        # Set up paths
        base_output_dir = Path(VOLUME_MOUNT_PATH) / "outputs"
        if method == "mcts": experiment_dir = base_output_dir / f"{dataset}_{method}"
        else: experiment_dir = base_output_dir / f"{dataset}_{method}"
        pareto_file = experiment_dir / f"pareto_frontier_{dataset}.json"
        
        # Check if pareto frontier file exists
        if not pareto_file.exists():
            print(f"❌ Pareto frontier file not found: {pareto_file}")
            return {
                "success": False,
                "dataset": dataset,
                "method": method,
                "error": f"Pareto frontier file not found: {pareto_file}"
            }
        
        # Load pareto frontier
        with open(pareto_file, 'r') as f:
            pareto_data = json.load(f)
        
        frontier_points = pareto_data.get("frontier_points", [])
        print(f"📊 Found {len(frontier_points)} frontier points")
        
        if not frontier_points:
            return {
                "success": False,
                "dataset": dataset,
                "method": method,
                "error": "No frontier points found in pareto file"
            }
        
        # Process each frontier point
        test_results = []
        for i, point in enumerate(frontier_points):
            print(f"\n--- Processing frontier point {i+1}/{len(frontier_points)} ---")
            
            # Extract file name from point
            point_file = point.get("file")
            print(point_file)
            # if point_file == "gpt_config_4.json": continue
            if method != "mcts" and "test_cost" in point and "test_error" not in point: 
                test_results.append({
                    "file": point.get("file"),
                    "cost": point.get("test_cost"),
                    "accuracy": point.get("test_accuracy"),
                    "accuracy_metric": point.get("test_accuracy_metric"),
                })
                continue

            
            if not point_file:
                print("⚠️  No file field in frontier point, skipping")
                continue
            
            # Get the base name without .json extension
            base_name = Path(point_file).stem  # e.g., "cuad_modal_12" or "baseline_output" or "iteration_5_output"
            
            # Determine the correct YAML file name
            if base_name == "baseline_output":
                # For baseline_output.json, use the original pipeline YAML
                yaml_file = Path(f"experiments/reasoning/pipelines/{dataset}.yaml")
            elif base_name.endswith("_output"):
                # For iteration_X_output.json, strip _output to get iteration_X.yaml
                yaml_base = base_name.replace("_output", "")
                yaml_file = experiment_dir / f"{yaml_base}.yaml"
            elif base_name.endswith("_results"):
                # For iteration_X_results.json (baseline method), strip _results to get iteration_X.yaml
                yaml_base = base_name.replace("_results", "")
                yaml_file = experiment_dir / f"{yaml_base}.yaml"
            else:
                # Standard case: cuad_modal_12.json -> cuad_modal_12.yaml
                yaml_file = experiment_dir / f"{base_name}.yaml"
            
            if not yaml_file.exists():
                base_name = base_name.replace("_output", "_config")
                yaml_file = base_output_dir / f"{dataset}" / f"{base_name}.yaml"
                if not yaml_file.exists():
                    print(f"⚠️  YAML file not found: {yaml_file}, skipping")
                    continue
            
            print(f"📄 Processing {base_name}")
            
            # Load and modify YAML
            with open(yaml_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Change dataset path from train to test
            if 'datasets' in config:
                for dataset_name, dataset_config in config['datasets'].items():
                    original_path = dataset_config.get('path', '')
                    # Replace 'train' with 'test' in the path
                    test_path = original_path.replace('/train/', '/test/')
                    config['datasets'][dataset_name]['path'] = test_path
                    print(f"📂 Changed dataset path to: {test_path}")
            
            # Change output path to include test_plans folder with method subdirectory
            if base_name == "baseline_output":
                # Special case for original pipeline
                test_output = f"{VOLUME_MOUNT_PATH}/outputs/{dataset}_{method}/test_plans/{method}/{dataset}_baseline.json"
            else:
                original_output = config.get('pipeline', {}).get('output', {}).get('path', '')
                # Insert test_plans/method before the filename
                output_parts = original_output.rsplit('/', 1)
                if len(output_parts) == 2:
                    test_output = f"{output_parts[0]}/test_plans/{method}/{output_parts[1]}"
                else:
                    test_output = f"test_plans/{method}/{original_output}"
            
            config['pipeline']['output']['path'] = test_output
            print(f"📤 Output path: {test_output}")
            
            # Create test_plans directory if it doesn't exist
            test_plans_dir = Path(test_output).parent
            test_plans_dir.mkdir(parents=True, exist_ok=True)
            
            # Save modified YAML to /tmp for temporary use (we don't need to keep these)
            test_yaml_path = Path("/tmp") / f"{dataset}_{method}_{base_name}_test.yaml"
            with open(test_yaml_path, 'w') as f:
                yaml.dump(config, f, sort_keys=False)
            
            try:
                # Run the pipeline
                print("🚀 Running pipeline...")
                start_time = time.time()
                runner = DSLRunner.from_yaml(str(test_yaml_path))
                runner.load()
                
                if runner.last_op_container:
                    data, _, _ = runner.last_op_container.next()
                    runner.save(data)
                
                total_cost = runner.total_cost
                runner.reset_env()
                end_time = time.time()
                latency = end_time - start_time
                
                print(f"✅ Pipeline completed. Cost: ${total_cost:.6f}, Latency: {latency:.3f}s")
                
                # Evaluate accuracy
                eval_func = get_evaluate_func(dataset)
                if eval_func:
                    # Evaluate results
                    accuracy_results = eval_func(base_name, test_output)
                    
                    # Get the appropriate accuracy metric
                    accuracy_metric = dataset_accuracy_metrics.get(dataset, "accuracy")
                    print("res: ", accuracy_results)
                    print("accuracy_metric: ", accuracy_metric)
                    accuracy = accuracy_results.get(accuracy_metric, 0.0)
                    
                    print(f"📈 Accuracy ({accuracy_metric}): {accuracy:.4f}")
                else:
                    print(f"⚠️  No evaluation function found for {dataset}")
                    accuracy = None
                    accuracy_metric = None
                
                # Store result
                test_results.append({
                    "file": base_name,
                    "cost": total_cost,
                    "accuracy": accuracy,
                    "accuracy_metric": accuracy_metric,
                    "latency": latency,
                })
                
            except Exception as e:
                print(f"❌ Error running pipeline: {e}")
                traceback.print_exc()
                test_results.append({
                    "file": base_name,
                    "error": str(e),
                })
        
        # Save test results back to the pareto frontier file with test results included
        pareto_file = experiment_dir / f"pareto_frontier_{dataset}.json"
        with open(pareto_file, 'r') as f:
            pareto_data = json.load(f)
        
        # Add test results to each frontier point
        for point in pareto_data.get("frontier_points", []):
            point_file = point.get("file")
            if point_file:
                point_base = Path(point_file).stem
                # Find matching test result
                for test_result in test_results:
                    if test_result.get("file") == point_base:
                        point["test_cost"] = test_result.get("cost")
                        point["test_accuracy"] = test_result.get("accuracy")
                        point["test_accuracy_metric"] = test_result.get("accuracy_metric")
                        point["test_latency"] = test_result.get("latency")
                        if "error" in test_result:
                            point["test_error"] = test_result["error"]
                        break
        
        # Update metadata
        pareto_data["test_evaluation"] = {
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }
        
        # Save updated pareto frontier with test results
        with open(pareto_file, 'w') as f:
            json.dump(pareto_data, f, indent=2)
        
        print(f"\n✅ Test results added to: {pareto_file}")
        
        # Commit changes to Modal volume
        volume.commit()
        
        return {
            "success": True,
            "dataset": dataset,
            "method": method,
            "results": test_results
        }
        
    except Exception as e:
        print(f"❌ Error in run_test_frontier_remote: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "dataset": dataset,
            "method": method,
            "error": str(e)
        }


@app.function(
    image=image,
    secrets=[modal.Secret.from_dotenv()],
    volumes={VOLUME_MOUNT_PATH: volume},
    timeout=60 * 30
)
def generate_test_frontier_plot(dataset: str) -> Dict[str, Any]:
    """
    Generate a plot of test frontier results for all methods.
    
    Args:
        dataset: Dataset name
        
    Returns:
        Dictionary with plot generation status
    """
    matplotlib.use('Agg')  # Use non-interactive backend for Modal
    
    try:
        print(f"\n📊 Generating test frontier plot for {dataset}")
        
        base_output_dir = Path(VOLUME_MOUNT_PATH) / "outputs"
        
        # Get the accuracy metric name for the dataset
        accuracy_metric = dataset_accuracy_metrics.get(dataset, "accuracy")
        
        # Collect all test frontier points from each method
        all_points = {
            "original": [],
            "simple_baseline": [],
            "baseline": [],
            "mcts": [],
            "lotus": [],
            "pz_direct": [],
            "pz_retrieval": []
        }
        
        # Method colors
        method_colors = {
            "original": "#ffd700",           # Gold/Yellow
            "simple_baseline": "#2ecc71",    # Green
            "baseline": "#1f77b4",           # Blue (darker blue)
            "mcts": "#0f1b3c",               # Very dark navy blue
            "lotus": "#c27cf3",              # Light purple
            "pz_direct": "#ff0b50",                 # Pink/magenta
            "pz_retrieval": "#ff6b35"        # Orange
        }
        
        # Load test_frontier_summary.json from dataset_original folder
        summary_file = base_output_dir / f"{dataset}_original" / "test_frontier_summary.json"
        
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
            
            # Extract results for each method
            if "results" in summary_data:
                # Handle original baseline
                if "original" in summary_data["results"] and summary_data["results"]["original"].get("success"):
                    original_result = summary_data["results"]["original"]
                    if "cost" in original_result and "accuracy" in original_result:
                        all_points["original"].append({
                            "cost": original_result["cost"],
                            "accuracy": original_result["accuracy"]
                        })
                    print(f"  ✅ Loaded {len(all_points['original'])} test points from original")
                else:
                    print("  ⚠️  No successful results found for original")
                
                # Handle other methods
                for method in METHODS:
                    if method in summary_data["results"] and summary_data["results"][method].get("success"):
                        method_results = summary_data["results"][method].get("results", [])
                        for point in method_results:
                            if "cost" in point and "accuracy" in point:
                                all_points[method].append({
                                    "cost": point["cost"],
                                    "accuracy": point["accuracy"]
                                })
                        print(f"  ✅ Loaded {len(all_points[method])} test points from {method}")
                    else:
                        print(f"  ⚠️  No successful results found for {method}")
        else:
            print(f"  ⚠️  No test_frontier_summary.json found at {summary_file}")
        
        # check local othersystems directory
        local_othersystems = Path("experiments/reasoning/othersystems") / dataset
        if local_othersystems.exists():
            print(f"  📁 Local othersystems/{dataset} contents:")
            for item in local_othersystems.iterdir():
                print(f"     - {item.name}")
        
        # Load LOTUS evaluation data if available (from local filesystem in Modal image)
        lotus_file = Path("experiments/reasoning/othersystems") / dataset / "lotus_evaluation.json"
        
        if lotus_file.exists():
            print(f"  📄 Found LOTUS evaluation at: {lotus_file}")
            with open(lotus_file, 'r') as f:
                lotus_data = json.load(f)
            
            # Extract test results from LOTUS (only lotus_test.json entries)
            for entry in lotus_data:
                if "lotus_test.json" in entry.get("file", "") or "lotus_full_test.json" in entry.get("file", ""):
                    # Find the accuracy metric using containment
                    accuracy_value = None
                    for key in entry.keys():
                        # Check if either the key contains the metric or the metric contains the key
                        if accuracy_metric in key or key in accuracy_metric:
                            accuracy_value = entry[key]
                            break
                    
                    if accuracy_value is None:
                        print(f"  ❌ Could not find accuracy metric '{accuracy_metric}' in LOTUS entry")
                        print(f"     Available keys: {list(entry.keys())}")
                        continue
                    
                    if "cost" in entry:
                        all_points["lotus"].append({
                            "cost": entry["cost"],
                            "accuracy": accuracy_value
                        })
                    elif "total_cost" in entry:
                        all_points["lotus"].append({
                            "cost": entry["total_cost"],
                            "accuracy": accuracy_value
                        })
            
            print(f"  ✅ Loaded {len(all_points['lotus'])} test points from LOTUS")
        else:
            print(f"  ⚠️  No LOTUS evaluation found at {lotus_file}")
        
        # Load PZ evaluation data if available (from local filesystem in Modal image)
        pz_file = Path("experiments/reasoning/othersystems") / dataset / "pz_evaluation.json"
        
        if pz_file.exists():
            print(f"  📄 Found PZ evaluation at: {pz_file}")
            with open(pz_file, 'r') as f:
                pz_data = json.load(f)
            
            # Load PZ direct data
            if "direct" in pz_data:
                print(f"  📊 Loading PZ direct data...")
                for config_name, config_data in pz_data["direct"].items():
                    if isinstance(config_data, dict):
                        # Skip if "metadata" is the config_name
                        if config_name == "metadata":
                            continue
                        
                        # Find the accuracy metric using containment
                        accuracy_value = None
                        for key in config_data.keys():
                            # Check if either the key contains the metric or the metric contains the key
                            if accuracy_metric in key or key in accuracy_metric:
                                accuracy_value = config_data[key]
                                break
                        
                        if accuracy_value is None:
                            print(f"  ❌ Could not find accuracy metric '{accuracy_metric}' in PZ direct config '{config_name}'")
                            print(f"     Available keys: {list(config_data.keys())}")
                            continue
                        
                        if "plan_execution_cost" in config_data:
                            all_points["pz_direct"].append({
                                "cost": config_data["plan_execution_cost"],
                                "accuracy": accuracy_value
                            })
                
                print(f"  ✅ Loaded {len(all_points['pz_direct'])} test points from PZ direct")
            
            # Load PZ retrieval data
            if "retrieval" in pz_data:
                print(f"  📊 Loading PZ retrieval data...")
                for config_name, config_data in pz_data["retrieval"].items():
                    if isinstance(config_data, dict):
                        # Skip if "metadata" is the config_name
                        if config_name == "metadata":
                            continue
                        
                        # Find the accuracy metric using containment
                        accuracy_value = None
                        for key in config_data.keys():
                            # Check if either the key contains the metric or the metric contains the key
                            if accuracy_metric in key or key in accuracy_metric:
                                accuracy_value = config_data[key]
                                break
                        
                        if accuracy_value is None:
                            print(f"  ❌ Could not find accuracy metric '{accuracy_metric}' in PZ retrieval config '{config_name}'")
                            print(f"     Available keys: {list(config_data.keys())}")
                            continue
                        
                        if "plan_execution_cost" in config_data:
                            all_points["pz_retrieval"].append({
                                "cost": config_data["plan_execution_cost"],
                                "accuracy": accuracy_value
                            })
                
                print(f"  ✅ Loaded {len(all_points['pz_retrieval'])} test points from PZ retrieval")
        else:
            print(f"  ⚠️  No PZ evaluation found at {pz_file}")
        
        # Check if we have any data to plot
        total_points = sum(len(points) for points in all_points.values())
        if total_points == 0:
            return {
                "success": False,
                "error": f"No test data found for {dataset}"
            }
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot points for each method
        for method, points in all_points.items():
            if points:
                costs = [p["cost"] for p in points]
                accuracies = [p["accuracy"] for p in points]
                
                if method == "original":
                    # Plot original as yellow stars
                    ax.scatter(costs, accuracies, 
                              color=method_colors[method],
                              label="Original",
                              s=150, marker='*', alpha=0.8, edgecolors='black', linewidth=1)
                else:
                    ax.scatter(costs, accuracies, 
                              color=method_colors[method],
                              label=method.replace("_", " ").title(),
                              s=100, alpha=0.7, edgecolors='black', linewidth=1)
        
        # Set log scale for x-axis (cost)
        ax.set_xscale('log')
        
        # Get the range of costs to set appropriate ticks
        all_costs = []
        for points in all_points.values():
            all_costs.extend([p["cost"] for p in points])
        
        # Filter out infinite and NaN values, and convert to float
        filtered_costs = []
        for cost in all_costs:
            try:
                # Convert to float if it's not already
                cost_float = float(cost)
                if np.isfinite(cost_float):
                    filtered_costs.append(cost_float)
            except (ValueError, TypeError):
                # Skip non-numeric values
                continue
        all_costs = filtered_costs
        
        if all_costs:
            min_cost = min(all_costs)
            max_cost = max(all_costs)
            
            # Handle zero or negative costs for log scale
            if min_cost <= 0:
                # If we have zero or negative costs, use a small positive value for log scale
                min_cost = max(0.01, min_cost)  # Use at least 0.01 for log scale
                print(f"⚠️  Found zero or negative cost, using {min_cost} for log scale")
            
            # Set x-axis limits with some padding
            ax.set_xlim(min_cost * 0.8, max_cost * 1.2)
            
            # Create explicit ticks based on the data range
            # Find the order of magnitude range
            min_order = np.floor(np.log10(min_cost * 0.8))
            max_order = np.ceil(np.log10(max_cost * 1.2))
            
            # Handle infinite values
            if not np.isfinite(min_order) or not np.isfinite(max_order):
                print(f"⚠️  Infinite values detected in log calculation. min_order: {min_order}, max_order: {max_order}")
                # Use a reasonable default range
                min_order = -2  # 0.01
                max_order = 2   # 100
                ax.set_xlim(0.01, 100)
            
            # Create ticks at powers of 10 and some intermediate values
            tick_values = []
            for exp in range(int(min_order), int(max_order) + 1):
                tick_values.extend([10**exp, 2*10**exp, 5*10**exp])
            
            # Filter to only include ticks within our range
            tick_values = [t for t in tick_values if min_cost * 0.8 <= t <= max_cost * 1.2]
            
            # Set the ticks explicitly
            ax.set_xticks(tick_values)
            
            # Format labels - use $ and appropriate decimal places
            def format_func(value, tick_number):
                if value >= 1:
                    return f'${value:.0f}'
                elif value >= 0.1:
                    return f'${value:.1f}'
                else:
                    return f'${value:.2f}'
            
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
        
        # Labels and title
        ax.set_xlabel('Cost ($) - Log Scale', fontsize=12)
        ax.set_ylabel(f'{accuracy_metric.replace("_", " ").title()}', fontsize=12)
        ax.set_title(f'Frontier Plans on Test Set - {dataset.upper()} Dataset', fontsize=14, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.grid(True, which='minor', alpha=0.1, linestyle=':')
        
        # Add legend
        ax.legend(loc='best', frameon=True, shadow=True)
        
        # Tight layout
        plt.tight_layout()
        
        # Save the plot
        plot_path = base_output_dir / f"{dataset}_original" / "test_frontier_plot.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Test frontier plot saved to: {plot_path}")
        
        # Calculate and print comparison metrics
        print("\n📊 Calculating comparison metrics...")
        comparison_metrics = calculate_comparison_metrics(all_points)
        
        # Print metrics
        print("\n" + "="*60)
        print("PAIRWISE COMPARISON METRICS (MCTS vs Each Method)")
        print("="*60)
        
        if comparison_metrics.get("error"):
            print(f"❌ Error calculating metrics: {comparison_metrics['error']}")
        else:
            pairwise_comparisons = comparison_metrics.get("pairwise_comparisons", {})
            
            for other_method, metrics in pairwise_comparisons.items():
                print(f"\n📊 MCTS vs {other_method.upper()}:")
                print("-" * 40)
                
                if metrics.get("error"):
                    print(f"   ❌ {metrics['error']}")
                    continue
                
                # 1. Accuracy improvement
                if metrics["accuracy_improvement"]:
                    acc_imp = metrics["accuracy_improvement"]
                    print(f"1. ACCURACY IMPROVEMENT:")
                    print(f"   MCTS best accuracy: {acc_imp['our_best']:.4f}")
                    print(f"   {other_method} best accuracy: {acc_imp['other_best']:.4f}")
                    print(f"   Improvement: +{acc_imp['absolute']:.4f} ({acc_imp['percentage']:.1f}%)")
                else:
                    print("1. ACCURACY IMPROVEMENT: Could not calculate")
                
                # 2. Cost savings within 10% accuracy
                if metrics["cost_savings_within_10pct"]:
                    cost_sav = metrics["cost_savings_within_10pct"]
                    print(f"\n2. COST SAVINGS (within 10% of {other_method}'s best):")
                    print(f"   MCTS cost: ${cost_sav['our_cost']:.6f} (accuracy: {cost_sav['our_accuracy']:.4f})")
                    print(f"   {other_method} cost: ${cost_sav['other_cost']:.6f} (accuracy: {cost_sav['other_accuracy']:.4f})")
                    print(f"   Savings: ${cost_sav['absolute']:.6f} ({cost_sav['percentage']:.1f}%)")
                else:
                    print(f"\n2. COST SAVINGS (within 10% of {other_method}'s best): Could not calculate")
                
                # 3. Average cost savings
                if metrics["average_cost_savings"]:
                    avg_sav = metrics["average_cost_savings"]
                    print(f"\n3. AVERAGE COST SAVINGS:")
                    print(f"   Average savings: ${avg_sav['absolute']:.6f} ({avg_sav['percentage']:.1f}%)")
                    print(f"   Based on {avg_sav['comparisons_count']} comparisons")
                else:
                    print("\n3. AVERAGE COST SAVINGS: Could not calculate")
        
        print("="*60)
        
        # Commit changes to Modal volume
        volume.commit()
        
        return {
            "success": True,
            "plot_path": str(plot_path),
            "total_points": total_points,
            "points_by_method": {m: len(p) for m, p in all_points.items()},
            "comparison_metrics": comparison_metrics
        }
        
    except Exception as e:
        print(f"❌ Error generating test frontier plot: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


@app.function(
    image=image,
    secrets=[modal.Secret.from_dotenv()],
    volumes={VOLUME_MOUNT_PATH: volume},
    timeout=60 * 60
)
def run_original_baseline_test(dataset: str) -> Dict[str, Any]:
    """
    Run the original baseline pipeline on test data.
    
    Args:
        dataset: Dataset name
        
    Returns:
        Dictionary with test results
    """
    try:
        print(f"\n{'='*60}")
        print(f"Running original baseline test for {dataset}")
        print(f"{'='*60}\n")
        
        # Load original pipeline YAML
        pipeline_path = Path(f"experiments/reasoning/pipelines/{dataset}.yaml")
        if not pipeline_path.exists():
            return {
                "success": False,
                "dataset": dataset,
                "error": f"Pipeline file not found: {pipeline_path}"
            }
        
        with open(pipeline_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Change dataset path from train to test
        if 'datasets' in config:
            for dataset_name, dataset_config in config['datasets'].items():
                original_path = dataset_config.get('path', '')
                # Replace 'train' with 'test' in the path
                test_path = original_path.replace('/train/', '/test/')
                config['datasets'][dataset_name]['path'] = test_path
                print(f"📂 Changed dataset path to: {test_path}")
        
        # Set output path to tests/original folder
        base_output_dir = Path(VOLUME_MOUNT_PATH) / "outputs"
        test_output = str(base_output_dir / f"{dataset}_original" / "tests" / "original" / f"{dataset}_baseline_test.json")
        config['pipeline']['output']['path'] = test_output
        print(f"📤 Output path: {test_output}")
        
        # Check if output file already exists
        # if Path(test_output).exists():
        #     print(f"✅ Output file already exists: {test_output}")
        #     print("⏭️  Skipping execution - loading existing results")
            
        #     # Load existing results and return them
        #     try:
        #         with open(test_output, 'r') as f:
        #             existing_data = json.load(f)
                
        #         # Try to extract cost and accuracy from existing data
        #         # This is a fallback - you might need to adjust based on your data structure
        #         total_cost = 0.0
        #         accuracy = None
        #         accuracy_metric = None
                
        #         # If the existing file contains cost/accuracy info, use it
        #         if isinstance(existing_data, dict):
        #             total_cost = existing_data.get('cost', 0.0)
        #             accuracy = existing_data.get('accuracy')
        #             accuracy_metric = existing_data.get('accuracy_metric')
                
        #         return {
        #             "success": True,
        #             "dataset": dataset,
        #             "cost": total_cost,
        #             "accuracy": accuracy,
        #             "accuracy_metric": accuracy_metric,
        #             "output_path": test_output,
        #             "skipped": True
        #         }
        #     except Exception as e:
        #         print(f"⚠️  Could not load existing results: {e}")
        #         print("🔄 Proceeding with execution...")
        
        # Create output directory
        Path(test_output).parent.mkdir(parents=True, exist_ok=True)
        
        # Save modified YAML to /tmp
        test_yaml_path = Path("/tmp") / f"{dataset}_original_baseline_test.yaml"
        with open(test_yaml_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False)
        
        # Run the pipeline
        print("🚀 Running original baseline pipeline...")
        start_time = time.time()
        runner = DSLRunner.from_yaml(str(test_yaml_path))
        runner.load()
        
        if runner.last_op_container:
            data, _, _ = runner.last_op_container.next()
            runner.save(data)
        
        total_cost = runner.total_cost
        runner.reset_env()
        end_time = time.time()
        latency = end_time - start_time
        
        print(f"✅ Pipeline completed. Cost: ${total_cost:.6f}, Latency: {latency:.3f}s")
        
        # Evaluate accuracy
        eval_func = get_evaluate_func(dataset)
        if eval_func:
            # Evaluate results
            accuracy_results = eval_func("baseline_test", test_output)
            
            # Get the appropriate accuracy metric
            accuracy_metric = dataset_accuracy_metrics.get(dataset, "accuracy")
            accuracy = accuracy_results.get(accuracy_metric, 0.0)
            
            print(f"📈 Accuracy ({accuracy_metric}): {accuracy:.4f}")
        else:
            print(f"⚠️  No evaluation function found for {dataset}")
            accuracy = None
            accuracy_metric = None
        
        return {
            "success": True,
            "dataset": dataset,
            "cost": total_cost,
            "accuracy": accuracy,
            "accuracy_metric": accuracy_metric,
            "latency": latency,
            "output_path": test_output
        }
        
    except Exception as e:
        print(f"❌ Error running original baseline test: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "dataset": dataset,
            "error": str(e)
        }


@app.function(
    image=image,
    secrets=[modal.Secret.from_dotenv()],
    volumes={VOLUME_MOUNT_PATH: volume},
    timeout=60 * 60 * 3  # 3 hours timeout for all methods
)
def run_all_test_frontiers(dataset: str) -> Dict[str, Any]:
    """
    Run test frontier evaluation for all methods (original_baseline, simple_baseline, baseline, mcts) for a dataset.
    
    Args:
        dataset: Dataset name
        
    Returns:
        Combined results for all methods including original baseline
    """
    print(f"\n{'='*70}")
    print(f"RUNNING ALL TEST FRONTIERS FOR {dataset.upper()}")
    print(f"{'='*70}\n")
    
    all_results = {}
    
    # First run the original baseline pipeline
    print("\n📊 Processing original...")
    original_result = run_original_baseline_test.local(dataset)
    all_results["original"] = original_result
    
    if original_result["success"]:
        print("✅ original completed successfully")
    else:
        print(f"❌ original failed: {original_result.get('error', 'Unknown error')}")
    
    # Then run frontier evaluations for all methods
    for method in METHODS:
        print(f"\n📊 Processing {method}...")
        result = run_test_frontier_remote.local(dataset, method)
        all_results[method] = result
        
        if result["success"]:
            print(f"✅ {method} completed successfully")
        else:
            print(f"❌ {method} failed: {result.get('error', 'Unknown error')}")
    
    # Generate summary
    summary = {
        "dataset": dataset,
        "timestamp": datetime.now().isoformat(),
        "methods_processed": list(all_results.keys()),
        "successful_methods": [m for m, r in all_results.items() if r["success"]],
        "failed_methods": [m for m, r in all_results.items() if not r["success"]],
        "results": all_results
    }
    
    # Calculate comparison metrics
    print("\n📊 Calculating comparison metrics...")
    
    # Collect all points for metrics calculation
    all_points = {
        "original": [],
        "simple_baseline": [],
        "baseline": [],
        "mcts": [],
        "lotus": [],
        "pz": [],
        "pz_retrieval": []
    }
    
    # Extract points from our method (MCTS)
    if "mcts" in all_results and all_results["mcts"]["success"]:
        method_results = all_results["mcts"].get("results", [])
        for point in method_results:
            if "cost" in point and "accuracy" in point and point["cost"] is not None and point["accuracy"] is not None:
                all_points["mcts"].append({
                    "cost": point["cost"],
                    "accuracy": point["accuracy"]
                })
    
    # Extract points from other systems (original, simple_baseline, baseline)
    if "original" in all_results and all_results["original"]["success"]:
        if "cost" in all_results["original"] and "accuracy" in all_results["original"] and all_results["original"]["cost"] is not None and all_results["original"]["accuracy"] is not None:
            all_points["original"].append({
                "cost": all_results["original"]["cost"],
                "accuracy": all_results["original"]["accuracy"]
            })
    
    for method in ["simple_baseline", "baseline"]:
        if method in all_results and all_results[method]["success"]:
            method_results = all_results[method].get("results", [])
            for point in method_results:
                if "cost" in point and "accuracy" in point and point["cost"] is not None and point["accuracy"] is not None:
                    all_points[method].append({
                        "cost": point["cost"],
                        "accuracy": point["accuracy"]
                    })
    
    # Load other systems' data (LOTUS and PZ) from local filesystem
    local_othersystems = Path("experiments/reasoning/othersystems") / dataset
    accuracy_metric = dataset_accuracy_metrics.get(dataset, "accuracy")
    
    # Load LOTUS evaluation data
    lotus_file = local_othersystems / "lotus_evaluation.json"
    if lotus_file.exists():
        with open(lotus_file, 'r') as f:
            lotus_data = json.load(f)
        
        for entry in lotus_data:
            if "lotus_test.json" in entry.get("file", "") or "lotus_full_test.json" in entry.get("file", ""):
                accuracy_value = None
                for key in entry.keys():
                    if accuracy_metric in key or key in accuracy_metric:
                        accuracy_value = entry[key]
                        break
                
                if accuracy_value is not None:
                    cost = entry.get("cost") or entry.get("total_cost")
                    if cost is not None:
                        all_points["lotus"].append({
                            "cost": cost,
                            "accuracy": accuracy_value
                        })
    
    # Load PZ evaluation data
    pz_file = local_othersystems / "pz_evaluation.json"
    if pz_file.exists():
        with open(pz_file, 'r') as f:
            pz_data = json.load(f)
        
        # Load PZ direct data
        if "direct" in pz_data:
            for config_name, config_data in pz_data["direct"].items():
                if isinstance(config_data, dict) and config_name != "metadata":
                    accuracy_value = None
                    for key in config_data.keys():
                        if accuracy_metric in key or key in accuracy_metric:
                            accuracy_value = config_data[key]
                            break
                    
                    if accuracy_value is not None and "plan_execution_cost" in config_data and config_data["plan_execution_cost"] is not None:
                        all_points["pz_direct"].append({
                            "cost": config_data["plan_execution_cost"],
                            "accuracy": accuracy_value
                        })
        
        # Load PZ retrieval data
        if "retrieval" in pz_data:
            for config_name, config_data in pz_data["retrieval"].items():
                if isinstance(config_data, dict) and config_name != "metadata":
                    accuracy_value = None
                    for key in config_data.keys():
                        if accuracy_metric in key or key in accuracy_metric:
                            accuracy_value = config_data[key]
                            break
                    
                    if accuracy_value is not None and "plan_execution_cost" in config_data and config_data["plan_execution_cost"] is not None:
                        all_points["pz_retrieval"].append({
                            "cost": config_data["plan_execution_cost"],
                            "accuracy": accuracy_value
                        })
    
    # Calculate metrics
    comparison_metrics = calculate_comparison_metrics(all_points)
    summary["comparison_metrics"] = comparison_metrics
    
    # Print metrics
    print("\n" + "="*60)
    print("PAIRWISE COMPARISON METRICS (MCTS vs Each Method)")
    print("="*60)
    
    if comparison_metrics.get("error"):
        print(f"❌ Error calculating metrics: {comparison_metrics['error']}")
    else:
        pairwise_comparisons = comparison_metrics.get("pairwise_comparisons", {})
        
        for other_method, metrics in pairwise_comparisons.items():
            print(f"\n📊 MCTS vs {other_method.upper()}:")
            print("-" * 40)
            
            if metrics.get("error"):
                print(f"   ❌ {metrics['error']}")
                continue
            
            # 1. Accuracy improvement
            if metrics["accuracy_improvement"]:
                acc_imp = metrics["accuracy_improvement"]
                print(f"1. ACCURACY IMPROVEMENT:")
                print(f"   MCTS best accuracy: {acc_imp['our_best']:.4f}")
                print(f"   {other_method} best accuracy: {acc_imp['other_best']:.4f}")
                print(f"   Improvement: +{acc_imp['absolute']:.4f} ({acc_imp['percentage']:.1f}%)")
            else:
                print("1. ACCURACY IMPROVEMENT: Could not calculate")
            
            # 2. Cost savings within 10% accuracy
            if metrics["cost_savings_within_10pct"]:
                cost_sav = metrics["cost_savings_within_10pct"]
                print(f"\n2. COST SAVINGS (within 10% of {other_method}'s best):")
                print(f"   MCTS cost: ${cost_sav['our_cost']:.6f} (accuracy: {cost_sav['our_accuracy']:.4f})")
                print(f"   {other_method} cost: ${cost_sav['other_cost']:.6f} (accuracy: {cost_sav['other_accuracy']:.4f})")
                print(f"   Savings: ${cost_sav['absolute']:.6f} ({cost_sav['percentage']:.1f}%)")
            else:
                print(f"\n2. COST SAVINGS (within 10% of {other_method}'s best): Could not calculate")
            
            # 3. Average cost savings
            if metrics["average_cost_savings"]:
                avg_sav = metrics["average_cost_savings"]
                print(f"\n3. AVERAGE COST SAVINGS:")
                print(f"   Average savings: ${avg_sav['absolute']:.6f} ({avg_sav['percentage']:.1f}%)")
                print(f"   Based on {avg_sav['comparisons_count']} comparisons")
            else:
                print("\n3. AVERAGE COST SAVINGS: Could not calculate")
    
    print("="*60)
    
    # Save test frontier summary to file
    base_output_dir = Path(VOLUME_MOUNT_PATH) / "outputs"
    summary_path = base_output_dir / f"{dataset}_original" / "test_frontier_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Test evaluation complete for {dataset}")
    print(f"   Test frontier summary saved to: {summary_path}")
    print(f"   Results also integrated into pareto_frontier_{dataset}.json files")
    
    # Generate test frontier plot
    print("\n📊 Generating test frontier plot...")
    plot_result = generate_test_frontier_plot.local(dataset)
    if plot_result["success"]:
        print(f"✅ Plot saved to: {plot_result['plot_path']}")
        summary["plot_path"] = plot_result["plot_path"]
    else:
        print(f"⚠️  Failed to generate plot: {plot_result.get('error', 'Unknown error')}")
    
    # Commit changes
    volume.commit()
    
    return summary


@app.function(
    image=image,
    secrets=[modal.Secret.from_dotenv()],
    volumes={VOLUME_MOUNT_PATH: volume},
    timeout=60 * 30
)
def generate_all_matrices(dataset: str) -> Dict[str, Any]:
    """
    Generate all three matrices (best cost savings, average cost savings, and coverage) 
    for all methods (excluding original) for a dataset.
    
    Args:
        dataset: Dataset name
        
    Returns:
        Dictionary with all matrix results
    """
    try:
        print(f"\n📊 Generating all matrices for {dataset}")
        
        base_output_dir = Path(VOLUME_MOUNT_PATH) / "outputs"
        
        # Get the accuracy metric name for the dataset
        accuracy_metric = dataset_accuracy_metrics.get(dataset, "accuracy")
        
        # Collect all test frontier points from each method
        all_points = {
            "original": [],
            "simple_baseline": [],
            "baseline": [],
            "mcts": [],
            "lotus": [],
            "pz_direct": [],
            "pz_retrieval": []
        }
        
        # Load test_frontier_summary.json from dataset_original folder
        summary_file = base_output_dir / f"{dataset}_original" / "test_frontier_summary.json"
        
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
            
            # Extract results for each method
            if "results" in summary_data:
                # Handle original baseline
                if "original" in summary_data["results"] and summary_data["results"]["original"].get("success"):
                    original_result = summary_data["results"]["original"]
                    if "cost" in original_result and "accuracy" in original_result:
                        all_points["original"].append({
                            "cost": original_result["cost"],
                            "accuracy": original_result["accuracy"]
                        })
                    print(f"  ✅ Loaded {len(all_points['original'])} test points from original")
                else:
                    print("  ⚠️  No successful results found for original")
                
                # Handle other methods
                for method in METHODS:
                    if method in summary_data["results"] and summary_data["results"][method].get("success"):
                        method_results = summary_data["results"][method].get("results", [])
                        for point in method_results:
                            if "cost" in point and "accuracy" in point:
                                all_points[method].append({
                                    "cost": point["cost"],
                                    "accuracy": point["accuracy"]
                                })
                        print(f"  ✅ Loaded {len(all_points[method])} test points from {method}")
                    else:
                        print(f"  ⚠️  No successful results found for {method}")
        else:
            print(f"  ⚠️  No test_frontier_summary.json found at {summary_file}")
        
        # Load LOTUS evaluation data if available
        lotus_file = Path("experiments/reasoning/othersystems") / dataset / "lotus_evaluation.json"
        
        if lotus_file.exists():
            print(f"  📄 Found LOTUS evaluation at: {lotus_file}")
            with open(lotus_file, 'r') as f:
                lotus_data = json.load(f)
            
            # Extract test results from LOTUS
            for entry in lotus_data:
                if "lotus_test.json" in entry.get("file", "") or "lotus_full_test.json" in entry.get("file", ""):
                    accuracy_value = None
                    for key in entry.keys():
                        if accuracy_metric in key or key in accuracy_metric:
                            accuracy_value = entry[key]
                            break
                    
                    if accuracy_value is not None:
                        cost = entry.get("cost") or entry.get("total_cost")
                        if cost is not None:
                            all_points["lotus"].append({
                                "cost": cost,
                                "accuracy": accuracy_value
                            })
            
            print(f"  ✅ Loaded {len(all_points['lotus'])} test points from LOTUS")
        else:
            print(f"  ⚠️  No LOTUS evaluation found at {lotus_file}")
        
        # Load PZ evaluation data if available
        pz_file = Path("experiments/reasoning/othersystems") / dataset / "pz_evaluation.json"
        
        if pz_file.exists():
            print(f"  📄 Found PZ evaluation at: {pz_file}")
            with open(pz_file, 'r') as f:
                pz_data = json.load(f)
            
            # Load PZ direct data
            if "direct" in pz_data:
                for config_name, config_data in pz_data["direct"].items():
                    if isinstance(config_data, dict) and config_name != "metadata":
                        accuracy_value = None
                        for key in config_data.keys():
                            if accuracy_metric in key or key in accuracy_metric:
                                accuracy_value = config_data[key]
                                break
                        
                        if accuracy_value is not None and "plan_execution_cost" in config_data:
                            all_points["pz_direct"].append({
                                "cost": config_data["plan_execution_cost"],
                                "accuracy": accuracy_value
                            })
                
                print(f"  ✅ Loaded {len(all_points['pz_direct'])} test points from PZ direct")
            
            # Load PZ retrieval data
            if "retrieval" in pz_data:
                for config_name, config_data in pz_data["retrieval"].items():
                    if isinstance(config_data, dict) and config_name != "metadata":
                        accuracy_value = None
                        for key in config_data.keys():
                            if accuracy_metric in key or key in accuracy_metric:
                                accuracy_value = config_data[key]
                                break
                        
                        if accuracy_value is not None and "plan_execution_cost" in config_data:
                            all_points["pz_retrieval"].append({
                                "cost": config_data["plan_execution_cost"],
                                "accuracy": accuracy_value
                            })
                
                print(f"  ✅ Loaded {len(all_points['pz_retrieval'])} test points from PZ retrieval")
        else:
            print(f"  ⚠️  No PZ evaluation found at {pz_file}")
        
        # Calculate all three matrices
        print("\n📊 Calculating all matrices...")
        
        # 1. Best Cost Savings Matrix
        print("\n" + "="*80)
        print("1. BEST COST SAVINGS MATRIX")
        print("="*80)
        best_matrix_result = calculate_best_cost_savings_matrix(all_points)
        
        # 2. Average Cost Savings Matrix
        print("\n" + "="*80)
        print("2. AVERAGE COST SAVINGS MATRIX")
        print("="*80)
        avg_matrix_result = calculate_avg_cost_savings_matrix(all_points)
        
        # 3. Coverage Matrix
        print("\n" + "="*80)
        print("3. COVERAGE MATRIX")
        print("="*80)
        coverage_matrix_result = calculate_coverage_matrix(all_points)
        
        # Check for errors
        if best_matrix_result.get("error"):
            return {
                "success": False,
                "error": f"Best matrix error: {best_matrix_result['error']}"
            }
        
        if avg_matrix_result.get("error"):
            return {
                "success": False,
                "error": f"Average matrix error: {avg_matrix_result['error']}"
            }
        
        if coverage_matrix_result.get("error"):
            return {
                "success": False,
                "error": f"Coverage matrix error: {coverage_matrix_result['error']}"
            }
        
        # Print all matrices
        methods = best_matrix_result["methods"]
        original_best_accuracy = best_matrix_result["original_best_accuracy"]
        
        # Print method info
        print("\n" + "="*80)
        print("METHOD INFORMATION")
        print("="*80)
        if original_best_accuracy is not None:
            print(f"Original best accuracy: {original_best_accuracy:.4f}")
        print()
        
        for method in methods:
            best_info = best_matrix_result["method_info"][method]
            avg_info = avg_matrix_result["method_info"][method]
            cov_info = coverage_matrix_result["method_info"][method]
            print(f"{method:15} | Total points: {best_info['valid_points']} | Points above original: {best_info['points_above_original']}")
        print()
        
        # Print matrix headers
        header = f"{'Method':>15}"
        for method in methods:
            header += f"{method:>12}"
        
        # Print Best Cost Savings Matrix
        print("BEST COST SAVINGS MATRIX:")
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        
        for method_a in methods:
            row = f"{method_a:>15}"
            for method_b in methods:
                value = best_matrix_result["matrix"][method_a][method_b]
                if value is None:
                    row += f"{'None':>12}"
                elif value == "--":
                    row += f"{'--':>12}"
                elif value == "Unable":
                    row += f"{'Unable':>12}"
                elif value == "None":
                    row += f"{'None':>12}"
                else:
                    row += f"{value:>12.2f}"
            print(row)
        
        # Print Average Cost Savings Matrix
        print("\nAVERAGE COST SAVINGS MATRIX:")
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        
        for method_a in methods:
            row = f"{method_a:>15}"
            for method_b in methods:
                value = avg_matrix_result["matrix"][method_a][method_b]
                if value is None:
                    row += f"{'None':>12}"
                elif value == "--":
                    row += f"{'--':>12}"
                elif value == "Unable":
                    row += f"{'Unable':>12}"
                elif value == "None":
                    row += f"{'None':>12}"
                else:
                    row += f"{value:>12.2f}"
            print(row)
        
        # Print Coverage Matrix
        print("\nCOVERAGE MATRIX:")
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        
        for method_a in methods:
            row = f"{method_a:>15}"
            for method_b in methods:
                value = coverage_matrix_result["matrix"][method_a][method_b]
                if value is None:
                    row += f"{'None':>12}"
                elif value == "--":
                    row += f"{'--':>12}"
                elif value == "Unable":
                    row += f"{'Unable':>12}"
                elif value == "None":
                    row += f"{'None':>12}"
                else:
                    row += f"{value:>12.3f}"
            print(row)
        
        print("="*80)
        
        # Save all matrices to files
        timestamp = datetime.now().isoformat()
        
        # Save Best Cost Savings Matrix
        best_matrix_path = base_output_dir / f"{dataset}_original" / "best_cost_savings_matrix.json"
        best_matrix_path.parent.mkdir(parents=True, exist_ok=True)
        
        best_matrix_data = {
            "dataset": dataset,
            "timestamp": timestamp,
            "original_best_accuracy": original_best_accuracy,
            "methods": methods,
            "method_info": best_matrix_result["method_info"],
            "matrix": best_matrix_result["matrix"]
        }
        
        with open(best_matrix_path, 'w') as f:
            json.dump(best_matrix_data, f, indent=2)
        
        # Create visualization for Best Cost Savings Matrix
        plot_matrix(best_matrix_result, 'best_cost_savings', dataset, base_output_dir / f"{dataset}_original")
        
        # Save Average Cost Savings Matrix
        avg_matrix_path = base_output_dir / f"{dataset}_original" / "avg_cost_savings_matrix.json"
        
        avg_matrix_data = {
            "dataset": dataset,
            "timestamp": timestamp,
            "original_best_accuracy": original_best_accuracy,
            "methods": methods,
            "method_info": avg_matrix_result["method_info"],
            "matrix": avg_matrix_result["matrix"]
        }
        
        with open(avg_matrix_path, 'w') as f:
            json.dump(avg_matrix_data, f, indent=2)
        
        # Create visualization for Average Cost Savings Matrix
        plot_matrix(avg_matrix_result, 'avg_cost_savings', dataset, base_output_dir / f"{dataset}_original")
        
        # Save Coverage Matrix
        coverage_matrix_path = base_output_dir / f"{dataset}_original" / "coverage_matrix.json"
        
        coverage_matrix_data = {
            "dataset": dataset,
            "timestamp": timestamp,
            "original_best_accuracy": original_best_accuracy,
            "methods": methods,
            "method_info": coverage_matrix_result["method_info"],
            "matrix": coverage_matrix_result["matrix"]
        }
        
        with open(coverage_matrix_path, 'w') as f:
            json.dump(coverage_matrix_data, f, indent=2)
        
        # Create visualization for Coverage Matrix
        plot_matrix(coverage_matrix_result, 'coverage', dataset, base_output_dir / f"{dataset}_original")
        
        print(f"\n✅ All matrices saved and visualized:")
        print(f"   Best cost savings matrix: {best_matrix_path}")
        print(f"   Average cost savings matrix: {avg_matrix_path}")
        print(f"   Coverage matrix: {coverage_matrix_path}")
        print(f"   Matrix plots saved in: {base_output_dir / f'{dataset}_original'}")
        
        # Commit changes to Modal volume
        volume.commit()
        
        return {
            "success": True,
            "best_matrix_path": str(best_matrix_path),
            "avg_matrix_path": str(avg_matrix_path),
            "coverage_matrix_path": str(coverage_matrix_path),
            "methods": methods,
            "original_best_accuracy": original_best_accuracy
        }
        
    except Exception as e:
        print(f"❌ Error generating all matrices: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


@app.local_entrypoint()
def main(dataset: str = "cuad", method: str = "all", plot_only: bool = False, matrix_only: bool = False):
    """
    Main entrypoint for running test frontier evaluation.
    
    Args:
        dataset: Dataset to process ('cuad', 'blackvault', etc., or 'all' for all datasets)
        method: Method to run ('simple_baseline', 'baseline', 'mcts', or 'all' for all methods)
        plot_only: If True, only generate the plot without running evaluations
        matrix_only: If True, only generate all three matrices without running evaluations
    """
    if dataset not in DATASETS + ["all"]:
        print(f"❌ Invalid dataset: {dataset}")
        print(f"   Valid options: {', '.join(DATASETS + ['all'])}")
        return
    
    if matrix_only:
        # Just generate all three matrices for the dataset(s)
        datasets_to_matrix = DATASETS if dataset == "all" else [dataset]
        for dataset_name in datasets_to_matrix:
            print(f"\n📊 Generating all matrices for {dataset_name}...")
            matrix_result = generate_all_matrices.remote(dataset_name)
            if matrix_result["success"]:
                print(f"✅ All matrices saved:")
                print(f"   Best cost savings matrix: {matrix_result['best_matrix_path']}")
                print(f"   Average cost savings matrix: {matrix_result['avg_matrix_path']}")
                print(f"   Coverage matrix: {matrix_result['coverage_matrix_path']}")
                print(f"   Methods: {', '.join(matrix_result['methods'])}")
                if matrix_result['original_best_accuracy'] is not None:
                    print(f"   Original best accuracy: {matrix_result['original_best_accuracy']:.4f}")
            else:
                print(f"❌ Failed to generate matrices: {matrix_result.get('error', 'Unknown error')}")
        return
    
    if plot_only:
        # Just generate the plot for the dataset(s)
        datasets_to_plot = DATASETS if dataset == "all" else [dataset]
        for dataset_name in datasets_to_plot:
            print(f"\n📊 Generating plot for {dataset_name}...")
            plot_result = generate_test_frontier_plot.remote(dataset_name)
            if plot_result["success"]:
                print(f"✅ Plot saved to: {plot_result['plot_path']}")
                print(f"   Total points: {plot_result['total_points']}")
                for m, count in plot_result['points_by_method'].items():
                    if count > 0:
                        print(f"   - {m}: {count} points")
            else:
                print(f"❌ Failed to generate plot: {plot_result.get('error', 'Unknown error')}")
        return
    
    if method not in METHODS + ["all"]:
        print(f"❌ Invalid method: {method}")
        print(f"   Valid options: {', '.join(METHODS + ['all'])}")
        return
    
    datasets_to_process = DATASETS if dataset == "all" else [dataset]
    
    for dataset_name in datasets_to_process:
        if method == "all":
            # Run all methods for this dataset
            result = run_all_test_frontiers.remote(dataset_name)
            print(f"\n{'='*70}")
            print(f"SUMMARY FOR {dataset_name.upper()}")
            print(f"{'='*70}")
            print(f"✅ Successful methods: {', '.join(result['successful_methods'])}")
            if result['failed_methods']:
                print(f"❌ Failed methods: {', '.join(result['failed_methods'])}")
            if "plot_path" in result:
                print(f"📈 Plot saved to: {result['plot_path']}")
        else:
            # Run specific method
            result = run_test_frontier_remote.remote(dataset_name, method)
            if result["success"]:
                print(f"\n✅ Successfully processed {dataset_name} - {method}")
                print("📄 Results saved to pareto frontier file")
                # Generate plot after single method run
                print("\n📊 Generating test frontier plot...")
                plot_result = generate_test_frontier_plot.remote(dataset_name)
                if plot_result["success"]:
                    print(f"📈 Plot saved to: {plot_result['plot_path']}")
            else:
                print(f"\n❌ Failed to process {dataset_name} - {method}")
                print(f"Error: {result.get('error', 'Unknown error')}")