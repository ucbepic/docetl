#!/usr/bin/env python3
"""
Run Pareto frontier plans on test datasets.

This script:
1. Pulls pareto frontier JSON files from Modal volume for each method (simple, mcts)
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


def accuracy_within_tolerance(acc1: float, acc2: float, tolerance_pct: float = 2) -> bool:
    """
    Check if two accuracy values are within the specified tolerance percentage.
    
    Args:
        acc1: First accuracy value
        acc2: Second accuracy value  
        tolerance_pct: Tolerance percentage (default 1.0%)
        
    Returns:
        True if acc1 is within tolerance_pct of acc2, False otherwise
    """
    if acc1 is None or acc2 is None:
        return False
    
    # Calculate absolute difference 
    diff_pct = abs(acc1 - acc2) * 100 
    return diff_pct < tolerance_pct


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
            if isinstance(val, dict) and 'absolute' in val:
                # Use absolute value for color intensity
                abs_val = val['absolute']
                numeric_positions.append((i, j, abs_val))
                numeric_values.append(abs(abs_val))
            elif isinstance(val, (int, float)) and val != '--':
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
            if isinstance(val, dict) and 'absolute' in val:
                # Handle dictionary format with absolute, ratio, and savings percentage
                abs_val = val['absolute']
                ratio_val = val['ratio']
                savings_pct_val = val['savings_pct']
                intensity = get_color_intensity(abs_val, max_abs_value)
                
                if abs_val > 0:
                    # Vibrant green gradient - we save money
                    base_color = base_colors['positive']
                    # Create more vibrant gradient
                    color = base_color * (0.4 + 0.6 * intensity)
                    # Add less white tint for more color
                    color = color + (1 - color) * (1 - intensity) * 0.3
                    text_color = 'black'  # Always black for better readability
                elif abs_val < 0:
                    # Vibrant red gradient - we cost more money
                    base_color = base_colors['negative']
                    # Create more vibrant gradient
                    color = base_color * (0.4 + 0.6 * intensity)
                    # Add less white tint for more color
                    color = color + (1 - color) * (1 - intensity) * 0.3
                    text_color = 'black'  # Always black for better readability
                else:
                    # Neutral color for zero savings
                    color = base_colors['diagonal']
                    text_color = 'gray'
                text = f'{abs_val:.3f}\n({ratio_val:.3f}x, {savings_pct_val:.3f}%)'
            elif isinstance(val, (int, float)):
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
                text = f'{val:.3f}'
                
            elif val == '--':
                color = base_colors['diagonal']
                text_color = 'gray'
                text = '--'
            elif val == 'n/a':
                color = base_colors['unable']
                text_color = 'gray'
                text = 'n/a'
            elif val == '—':
                color = base_colors['none']
                text_color = 'gray'
                text = '—'
            else:
                color = 'white'
                text_color = 'black'
                text = str(val)
            
            # Draw rectangle with subtle border
            rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='white', linewidth=1.5)
            ax.add_patch(rect)
            
            # Add text with larger font
            fontweight = 'bold' if isinstance(val, (int, float)) or (isinstance(val, dict) and 'absolute' in val) else 'normal'
            fontsize = 18 if isinstance(val, (int, float)) or (isinstance(val, dict) and 'absolute' in val) else 16
            ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', 
                    fontsize=fontsize, fontweight=fontweight, color=text_color)
    
    # Set up the plot
    ax.set_xlim(0, len(methods))
    ax.set_ylim(0, len(methods))
    ax.set_aspect('equal')
    
    # Set ticks and labels with larger font
    ax.set_xticks(np.arange(len(methods)) + 0.5)
    ax.set_yticks(np.arange(len(methods)) + 0.5)
    # Custom label mapping for methods
    label_map = {
        "original": "User-specified plan",
        "PZ_direct": "PZ-d",
        "PZ_retrieval": "PZ-r&r", 
        "PZ": "PZ",
        "lotus": "LOTUS",
        "LOTUS_d": "LOTUS-d",
        "LOTUS_r&r": "LOTUS-r&r",
        "mcts": "MOAR",
        "simple_baseline": "Simple agent"
    }
    
    # Apply label mapping
    x_labels = [label_map.get(method, method.replace("_", " ").title()) for method in methods]
    y_labels = [label_map.get(method, method.replace("_", " ").title()) for method in methods]
    
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=28)
    ax.set_yticklabels(y_labels, fontsize=28)
    
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
    # Map dataset names to proper titles
    dataset_titles = {
        'cuad': 'CUAD',
        'game_reviews': 'Game Reviews', 
        'blackvault': 'BlackVault',
        'biodex': 'Biodex',
        'medec': 'Medec',
        'sustainability': 'Sustainability'
    }
    title = dataset_titles.get(dataset, dataset.upper())
    
    plt.title(title, 
              fontsize=32, fontweight='bold', pad=30, color='#1f2937')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = output_dir / f"{dataset}_{matrix_type}_matrix.pdf"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Matrix plot saved: {plot_path}")
    
    # Summary statistics
    if numeric_values:
        numeric_vals = [val for i, j, val in numeric_positions]
        print(f"   Max absolute value: {max(numeric_vals):.3f}")
        print(f"   Min absolute value: {min(numeric_vals):.3f}")
        print(f"   Mean absolute value: {np.mean(numeric_vals):.3f}")

# Dataset configurations
DATASETS = ["cuad", "blackvault", "game_reviews", "sustainability", "biodex", "medec", "facility"]
METHODS = ["simple_baseline", "mcts"]

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

    # Get all methods excluding original and reorder them
    available_methods = [m for m in all_points.keys() if m != "original" and all_points[m]]
    
    # Define the desired order: MCTS, simple_baseline, lotus, PZ variants (no original)
    desired_order = ["mcts", "simple_baseline", "lotus", "LOTUS_d", "LOTUS_r&r", "PZ_direct", "PZ_retrieval", "PZ"]
    
    # Reorder methods according to desired order
    methods = []
    for method in desired_order:
        if method in available_methods:
            methods.append(method)
    
    # Add any remaining methods that weren't in the desired order
    for method in available_methods:
        if method not in methods:
            methods.append(method)
    
    if len(methods) < 2:
        return {
            "error": "Need at least 2 methods for matrix comparison"
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
                matrix[method_a][method_b] = "n/a"
                continue
            
            # Filter method B points to only those above original accuracy
            if original_best_accuracy is not None:
                method_b_above_original = [p for p in method_b_valid_points if p["accuracy"] > original_best_accuracy]
            else:
                method_b_above_original = method_b_valid_points
            
            if not method_b_above_original:
                matrix[method_a][method_b] = "—"  # No plans above original accuracy
                continue
            
            # Get method A's points
            method_a_points = all_points[method_a]
            method_a_valid_points = [p for p in method_a_points if p["accuracy"] is not None and p["cost"] is not None]
            
            if not method_a_valid_points:
                matrix[method_a][method_b] = "n/a"
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
                    
                    # Complete domination: lower cost AND (higher accuracy OR accuracy within 1% tolerance)
                    if method_a_cost < method_b_cost and (method_a_accuracy > method_b_accuracy or accuracy_within_tolerance(method_a_accuracy, method_b_accuracy)):
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
    # Get all methods excluding original and reorder them
    available_methods = [m for m in all_points.keys() if m != "original" and all_points[m]]
    
    # Define the desired order: MCTS, simple_baseline, lotus, PZ variants (no original)
    desired_order = ["mcts", "simple_baseline", "lotus", "LOTUS_d", "LOTUS_r&r", "PZ_direct", "PZ_retrieval", "PZ"]
    
    # Reorder methods according to desired order
    methods = []
    for method in desired_order:
        if method in available_methods:
            methods.append(method)
    
    # Add any remaining methods that weren't in the desired order
    for method in available_methods:
        if method not in methods:
            methods.append(method)
    
    if len(methods) < 2:
        return {
            "error": "Need at least 2 methods for matrix comparison"
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
                matrix[method_a][method_b] = "n/a"
                continue
            
            # Filter method B points to only those above original accuracy
            if original_best_accuracy is not None:
                method_b_above_original = [p for p in method_b_valid_points if p["accuracy"] > original_best_accuracy]
            else:
                method_b_above_original = method_b_valid_points
            
            if not method_b_above_original:
                matrix[method_a][method_b] = "—"  # No plans above original accuracy
                continue
            
            # Get method A's points
            method_a_points = all_points[method_a]
            method_a_valid_points = [p for p in method_a_points if p["accuracy"] is not None and p["cost"] is not None]
            
            if not method_a_valid_points:
                matrix[method_a][method_b] = "n/a"
                continue
            
            # Calculate cost savings for each method B plan
            cost_savings_list = []

            method_b_sum = 0.0
            method_a_sum = 0.0
            num = 0.0
            for method_b_point in method_b_above_original:
                method_b_accuracy = method_b_point["accuracy"]
                method_b_cost = method_b_point["cost"]
                
                
                # Find method A's cheapest plan that meets or exceeds this accuracy (within 1% tolerance)
                qualifying_method_a_points = [p for p in method_a_valid_points if p["accuracy"] >= method_b_accuracy or accuracy_within_tolerance(p["accuracy"], method_b_accuracy)]
                
                if qualifying_method_a_points:
                    method_a_cheapest = min(qualifying_method_a_points, key=lambda x: x["cost"])
                    savings = method_b_cost - method_a_cheapest["cost"]
                    cost_savings_list.append(savings)
                    method_a_sum += method_a_cheapest["cost"]
                    method_b_sum += method_b_cost
                    num += 1
            
            if method_b_sum == 0 or num == 0:
                matrix[method_a][method_b] = "n/a"
                continue

            ratio = float(method_a_sum) / float(method_b_sum)
            
            # Calculate average cost savings
            #avg_cost_savings = sum(cost_savings_list) / len(cost_savings_list)
            avg_cost_savings = (float(method_b_sum) - float(method_a_sum) ) / num

            
            matrix[method_a][method_b] = {
                "absolute": round(avg_cost_savings, 3),
                "ratio": round(ratio, 3)
            }
    
    return {
        "matrix": matrix,
        "methods": methods,
        "method_info": method_info,
        "original_best_accuracy": original_best_accuracy
    }


def calculate_best_cost_savings_matrix(all_points: Dict[str, List[Dict[str, float]]]) -> Dict[str, Any]:
    """
    Calculate a matrix of cost savings between all methods (including original).
    Each cell (method A, method B) shows how much cost method A saves for achieving 
    or surpassing the highest accuracy of method B.
    
    Args:
        all_points: Dictionary with method names as keys and lists of {cost, accuracy} dicts as values
        
    Returns:
        Dictionary containing the cost savings matrix
    """
    
    print("="*80)
    print(f"Calculating best cost savings matrix for {all_points}")
    # Get all methods including original and reorder them
    available_methods = [m for m in all_points.keys() if all_points[m]]
    
    # Define the desired order: MCTS, Original, simple_baseline, lotus, PZ variants
    desired_order = ["mcts", "original", "simple_baseline", "lotus", "LOTUS_d", "LOTUS_r&r", "PZ_direct", "PZ_retrieval", "PZ"]
    
    # Reorder methods according to desired order
    methods = []
    for method in desired_order:
        if method in available_methods:
            methods.append(method)
    
    # Add any remaining methods that weren't in the desired order
    for method in available_methods:
        if method not in methods:
            methods.append(method)
    
    if len(methods) < 2:
        return {
            "error": "Need at least 2 methods for matrix comparison"
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
                matrix[method_a][method_b] = "—"
                continue
            
            # Get method A's points
            method_a_points = all_points[method_a]
            method_a_valid_points = [p for p in method_a_points if p["accuracy"] is not None and p["cost"] is not None]
            
            if not method_a_valid_points:
                matrix[method_a][method_b] = "n/a"
                continue
            
            # Find method A's cheapest plan that meets or exceeds method B's best accuracy (within 1% tolerance)
            qualifying_points = [p for p in method_a_valid_points if p["accuracy"] >= method_b_best_accuracy or accuracy_within_tolerance(p["accuracy"], method_b_best_accuracy)]
            
            if not qualifying_points:
                matrix[method_a][method_b] = "n/a"
                continue
            
            method_a_cheapest = min(qualifying_points, key=lambda x: x["cost"])
            method_b_best_cost = method_b_info["best_cost"]
            
            if method_b_best_cost is None:
                matrix[method_a][method_b] = "n/a"
                continue
            
            # Calculate cost savings
            cost_savings = method_b_best_cost - method_a_cheapest["cost"]
            # Calculate cost ratio: what fraction of method_b's cost we use (savings perspective)
            cost_ratio = method_a_cheapest["cost"] / method_b_best_cost if method_b_best_cost > 0 else 0
            # Calculate savings percentage: what percentage of their cost we save
            savings_pct = (1 - cost_ratio) * 100
            
            matrix[method_a][method_b] = {
                "absolute": round(cost_savings, 3),
                "ratio": round(cost_ratio, 3),
                "savings_pct": round(savings_pct, 3)
            }
    
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
    other_systems = ["original", "simple_baseline", "lotus", "LOTUS_d", "LOTUS_r&r", "PZ_direct", "PZ_retrieval", "PZ"]
    
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
    
    # Find our best plan (highest accuracy)
    our_best_plan = max(our_valid_points, key=lambda x: x["accuracy"])
    our_best_accuracy = our_best_plan["accuracy"]
    our_best_file = our_best_plan.get("file", "unknown")
    
    # Calculate pairwise metrics for each other system
    pairwise_metrics = {}
    
    for other_method in other_systems:
        if other_method not in all_points: continue
        other_points = all_points[other_method]
        other_valid_points = [p for p in other_points if p["accuracy"] is not None]
        
        if not other_valid_points:
            pairwise_metrics[other_method] = {
                "error": f"No valid accuracy data found for {other_method}"
            }
            continue
        
        # Find other method's best plan (highest accuracy)
        other_best_plan = max(other_valid_points, key=lambda x: x["accuracy"])
        other_best_accuracy = other_best_plan["accuracy"]
        other_best_file = other_best_plan.get("file", "unknown")
        
        # 1. Calculate accuracy improvement: how much more accurate our best is vs their best
        accuracy_improvement = our_best_accuracy - other_best_accuracy
        accuracy_improvement_pct = (accuracy_improvement / other_best_accuracy) * 100 if other_best_accuracy > 0 else 0
        
        # 2. Calculate cost savings for plans of their best accuracy
        target_accuracy = other_best_accuracy 
        
        # Find our cheapest plan that meets or exceeds this accuracy (within 1% tolerance, filter None values)
        qualifying_our_plans = [p for p in our_valid_points if (p["accuracy"] >= target_accuracy or accuracy_within_tolerance(p["accuracy"], target_accuracy)) and p["cost"] is not None]
        
        cost_savings_within_10pct = None
        if qualifying_our_plans:
            our_cheapest_at_target = min(qualifying_our_plans, key=lambda x: x["cost"])
            
            # Find their cheapest plan at or above their best accuracy (within 1% tolerance, filter None values)
            qualifying_other_plans = [p for p in other_valid_points if (p["accuracy"] >= other_best_accuracy or accuracy_within_tolerance(p["accuracy"], other_best_accuracy)) and p["cost"] is not None]
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
                
            # Find our cheapest plan that meets or exceeds this accuracy (within 1% tolerance, filter None values)
            qualifying_our_plans = [p for p in our_valid_points if (p["accuracy"] >= other_accuracy or accuracy_within_tolerance(p["accuracy"], other_accuracy)) and p["cost"] is not None]
            
            if qualifying_our_plans:
                our_cheapest = min(qualifying_our_plans, key=lambda x: x["cost"])
                savings = other_cost - our_cheapest["cost"]
                cost_savings_list.append(savings)
        
        average_cost_savings = None
        if cost_savings_list:
            avg_cost_savings = sum(cost_savings_list) / len(cost_savings_list)
            # Calculate percentage based on valid points only (exclude zero costs to avoid division by zero)
            valid_other_points = [p for p in other_valid_points if p["cost"] is not None and p["cost"] > 0]
            # Only calculate percentage for points with non-zero costs
            if valid_other_points and len(valid_other_points) == len(cost_savings_list):
                avg_cost_savings_pct = sum(savings / valid_other_points[i]["cost"] * 100 
                                         for i, savings in enumerate(cost_savings_list)) / len(cost_savings_list)
            else:
                # If we have zero-cost points, skip percentage calculation
                avg_cost_savings_pct = None
            
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
                "our_best_file": our_best_file,
                "other_best": other_best_accuracy,
                "other_best_file": other_best_file
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
        method: Method name ('simple_baseline' or 'mcts')
    
    Returns:
        Dictionary with test results for all frontier points
    """
    try:
        print(f"\n{'='*60}")
        print(f"Running test frontier for {dataset} - {method}")
        print(f"{'='*60}\n")
        
        # Set up paths
        base_output_dir = Path(VOLUME_MOUNT_PATH) / "outputs"
        if method == "mcts": experiment_dir = base_output_dir / f"{dataset}_{method}_final"
        else: experiment_dir = base_output_dir / f"{dataset}_{method}_final"
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
                test_output = f"{VOLUME_MOUNT_PATH}/outputs/{dataset}_{method}_final/test_plans/{method}/{dataset}_baseline.json"
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
            test_yaml_path = Path("/tmp") / f"{dataset}_{method}_final_{base_name}_test.yaml"
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
                
                print(f"✅ Pipeline completed. Cost: ${total_cost:.3f}, Latency: {latency:.3f}s")
                
                # Evaluate accuracy
                eval_func = get_evaluate_func(dataset, mode="test")
                if eval_func:
                    # Evaluate results
                    print("base_name: ", base_name)
                    accuracy_results = eval_func(base_name, test_output)
                    
                    # Get the appropriate accuracy metric
                    accuracy_metric = dataset_accuracy_metrics.get(dataset, "accuracy")
                    print("res: ", accuracy_results)
                    print("accuracy_metric: ", accuracy_metric)
                    accuracy = accuracy_results.get(accuracy_metric, 0.0)
                    
                    print(f"📈 Accuracy ({accuracy_metric}): {accuracy:.3f}")
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
            "mcts": [],
            "lotus": [],
            "PZ_direct": [],
            "PZ_retrieval": [],
            "PZ": [],
            "LOTUS_d": [],
            "LOTUS_r&r": []
        }
        
        # Method colors
        method_colors = {
            "original": "#ffd700",           # Gold/Yellow
            "simple_baseline": "#2ecc71",    # Green
            "mcts": "#0f1b3c",               # Very dark navy blue
            "lotus": "#c27cf3",              # Light purple
            "PZ_direct": "#ff0b50",          # Pink/magenta
            "PZ_retrieval": "#ff0b50",       # Pink/magenta (same as PZ_direct)
            "PZ": "#ff0b50",                 # Pink/magenta
            "LOTUS_d": "#c27cf3",            # Light purple
            "LOTUS_r&r": "#c27cf3"           # Light purple
        }
        
        # Load test_frontier_summary.json from dataset_original folder
        summary_file = base_output_dir / f"{dataset}_original_final" / "test_frontier_summary.json"
        
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
            
            # Extract results for each method
            if "results" in summary_data:
                # Handle original baseline
                if "original" in summary_data["results"] and summary_data["results"]["original"].get("success"):
                    original_result = summary_data["results"]["original"]
                    if "cost" in original_result and "accuracy" in original_result:
                        point_data = {
                            "cost": original_result["cost"],
                            "accuracy": original_result["accuracy"]
                        }
                        # Include file field if available
                        if "file" in original_result:
                            point_data["file"] = original_result["file"]
                        all_points["original"].append(point_data)
                    print(f"  ✅ Loaded {len(all_points['original'])} test points from original")
                else:
                    print("  ⚠️  No successful results found for original")
                
                # Handle other methods
                for method in METHODS:
                    if method in summary_data["results"] and summary_data["results"][method].get("success"):
                        method_results = summary_data["results"][method].get("results", [])
                        for point in method_results:
                            if "cost" in point and "accuracy" in point:
                                point_data = {
                                    "cost": point["cost"],
                                    "accuracy": point["accuracy"]
                                }
                                # Include file field if available (especially for MCTS)
                                if "file" in point:
                                    point_data["file"] = point["file"]
                                all_points[method].append(point_data)
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
                
                # Check the method field to determine which LOTUS variant this is
                method = entry.get("method", "")
                print(f"  📊 Processing LOTUS entry: method={method}, file={entry.get('file', '')}")
                
                # Skip join_only method as requested
                if method == "join_only":
                    print(f"  ⏭️  Skipping join_only method")
                    continue
                
                # Determine the target method based on the method field
                if method == "simple_map":
                    target_method = "LOTUS_d"
                elif method == "join_and_rerank":
                    target_method = "LOTUS_r&r"
                else:
                    # Fallback to original lotus for unknown methods
                    target_method = "lotus"
                
                print(f"  🎯 Target method: {target_method}")
                
                # Find the accuracy metric using containment
                accuracy_value = None
                for key in entry.keys():
                    # Check if either the key contains the metric or the metric contains the key
                    if accuracy_metric in key or key in accuracy_metric:
                        accuracy_value = entry[key]
                        print(f"  ✅ Found accuracy metric '{key}' = {accuracy_value}")
                        break
                
                if accuracy_value is None:
                    print(f"  ❌ Could not find accuracy metric '{accuracy_metric}' in LOTUS entry")
                    print(f"     Available keys: {list(entry.keys())}")
                    continue
                
                if "cost" in entry:
                    print(f"  💰 Adding point: cost={entry['cost']:.3f}, accuracy={accuracy_value:.3f}")
                    all_points[target_method].append({
                        "cost": entry["cost"],
                        "accuracy": accuracy_value
                    })
                elif "total_cost" in entry:
                    print(f"  💰 Adding point: cost={entry['total_cost']:.3f}, accuracy={accuracy_value:.3f}")
                    all_points[target_method].append({
                        "cost": entry["total_cost"],
                        "accuracy": accuracy_value
                    })
                else:
                    print(f"  ❌ No cost found in entry")
            
            print(f"  ✅ Loaded {len(all_points['LOTUS_d'])} test points from LOTUS-d, {len(all_points['LOTUS_r&r'])} from LOTUS-r&r")
        else:
            print(f"  ⚠️  No LOTUS evaluation found at {lotus_file}")
        
        # Load PZ evaluation data if available (from local filesystem in Modal image)
        pz_file = Path("experiments/reasoning/othersystems") / dataset / "pz_evaluation.json"
        
        if pz_file.exists():
            print(f"  📄 Found PZ evaluation at: {pz_file}")
            with open(pz_file, 'r') as f:
                pz_data = json.load(f)
            
            # Check PZ data structure and load accordingly
            print(f"  📊 PZ data structure: {list(pz_data.keys())}")
            
            # Case 1: Has direct/retrieval structure
            if "direct" in pz_data or "retrieval" in pz_data:
                print(f"  📊 Loading PZ data with direct/retrieval structure...")
                
                # Load PZ direct data
                if "direct" in pz_data:
                    print(f"  📊 Loading PZ direct data...")
                    for config_name, config_data in pz_data["direct"].items():
                        if isinstance(config_data, dict) and config_name != "metadata":
                            # Find the accuracy metric using containment
                            accuracy_value = None
                            for key in config_data.keys():
                                if accuracy_metric in key or key in accuracy_metric:
                                    accuracy_value = config_data[key]
                                    break
                            
                            if accuracy_value is not None and "plan_execution_cost" in config_data:
                                all_points["PZ_direct"].append({
                                    "cost": config_data["plan_execution_cost"],
                                    "accuracy": accuracy_value
                                })
                    
                    print(f"  ✅ Loaded {len(all_points['PZ_direct'])} test points from PZ direct")
                
                # Load PZ retrieval data
                if "retrieval" in pz_data:
                    print(f"  📊 Loading PZ retrieval data...")
                    for config_name, config_data in pz_data["retrieval"].items():
                        if isinstance(config_data, dict) and config_name != "metadata":
                            # Find the accuracy metric using containment
                            accuracy_value = None
                            for key in config_data.keys():
                                if accuracy_metric in key or key in accuracy_metric:
                                    accuracy_value = config_data[key]
                                    break
                            
                            if accuracy_value is not None and "plan_execution_cost" in config_data:
                                all_points["PZ_retrieval"].append({
                                    "cost": config_data["plan_execution_cost"],
                                    "accuracy": accuracy_value
                                })
                    
                    print(f"  ✅ Loaded {len(all_points['PZ_retrieval'])} test points from PZ retrieval")
            
            # Case 2: Only has general PZ structure (no direct/retrieval)
            else:
                print(f"  📊 Loading PZ data (general structure)...")
                for config_name, config_data in pz_data.items():
                    if isinstance(config_data, dict) and config_name != "metadata":
                        accuracy_value = None
                        for key in config_data.keys():
                            if accuracy_metric in key or key in accuracy_metric:
                                accuracy_value = config_data[key]
                                break
                        
                        if accuracy_value is not None and "plan_execution_cost" in config_data:
                            all_points["PZ"].append({
                                "cost": config_data["plan_execution_cost"],
                                "accuracy": accuracy_value
                            })
                
                print(f"  ✅ Loaded {len(all_points['PZ'])} test points from PZ")
        else:
            print(f"  ⚠️  No PZ evaluation found at {pz_file}")
        
        # Print summary of loaded data
        print(f"\n📊 Data loading summary:")
        for method, points in all_points.items():
            if points:
                print(f"  {method}: {len(points)} points")
            else:
                print(f"  {method}: 0 points")
        
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
                              label="User-specified plan",
                              s=300, marker='*', alpha=0.4, edgecolors='black', linewidth=1)
                elif method == "mcts":
                    ax.scatter(costs, accuracies, 
                              color=method_colors[method],
                              label="MOAR",
                              s=200, alpha=0.4, edgecolors='black', linewidth=1)
                elif method == "simple_baseline":
                    ax.scatter(costs, accuracies, 
                              color=method_colors[method],
                              label="Simple agent",
                              s=200, alpha=0.4, edgecolors='black', linewidth=1)
                else:
                    # Custom label mapping for methods
                    label_map = {
                        "PZ_direct": "PZ-d",
                        "PZ_retrieval": "PZ-r&r", 
                        "PZ": "PZ",
                        "lotus": "LOTUS",
                        "LOTUS_d": "LOTUS-d",
                        "LOTUS_r&r": "LOTUS-r&r"
                    }
                    label = label_map.get(method, method.replace("_", " ").title())
                    
                    # Determine marker based on method
                    marker = '^' if method in ["PZ_retrieval", "LOTUS_r&r"] else 'o'  # Triangle for PZ-r&r and LOTUS-r&r, circle for others
                    
                    ax.scatter(costs, accuracies, 
                              color=method_colors[method],
                              label=label,
                              s=200, alpha=0.4, edgecolors='black', linewidth=1,
                              marker=marker)
                
                # MCTS points are plotted as dots only (no filename labels)
        
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
            if dataset == 'biodex':
                # For BioDEX, only use specific values: 0.01, 0.1, 1, 10
                tick_values = [0.01, 0.1, 1, 10]
                # Filter to only include ticks within our range
                tick_values = [t for t in tick_values if min_cost * 0.8 <= t <= max_cost * 1.2]
            else:
                for exp in range(int(min_order), int(max_order) + 1):
                    tick_values.extend([10**exp, 2*10**exp, 5*10**exp])  # 3 ticks per decade
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
        ax.set_xlabel('Cost ($) - Log Scale', fontsize=28)
        ax.set_ylabel(f'{accuracy_metric.replace("_", " ").title()}', fontsize=28)
        # Map dataset names to proper titles
        dataset_titles = {
            'cuad': 'CUAD',
            'game_reviews': 'Game Reviews', 
            'blackvault': 'BlackVault',
            'biodex': 'Biodex',
            'medec': 'Medec',
            'sustainability': 'Sustainability'
        }
        title = dataset_titles.get(dataset, dataset.upper())
        
        ax.set_title(title, fontsize=28, fontweight='bold')
        
        # Set tick label font size
        ax.tick_params(axis='both', which='major', labelsize=28)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.grid(True, which='minor', alpha=0.1, linestyle=':')
        
        # Add legend with pure transparent white background, adaptive position
        legend = ax.legend(loc='best', frameon=True, shadow=False, fontsize=20, 
                          facecolor='white', edgecolor='black')
        legend.get_frame().set_alpha(0.3)
        
        # Tight layout
        plt.tight_layout()
        
        # Save the plot
        plot_path = base_output_dir / f"{dataset}_original_final" / "test_frontier_plot.pdf"
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
                    print(f"   MCTS best accuracy: {acc_imp['our_best']:.3f} (file: {acc_imp['our_best_file']})")
                    print(f"   {other_method} best accuracy: {acc_imp['other_best']:.3f} (file: {acc_imp['other_best_file']})")
                    print(f"   Improvement: +{acc_imp['absolute']:.3f} ({acc_imp['percentage']:.3f}%)")
                else:
                    print("1. ACCURACY IMPROVEMENT: Could not calculate")
                
                # 2. Cost savings within 10% accuracy
                if metrics["cost_savings_within_10pct"]:
                    cost_sav = metrics["cost_savings_within_10pct"]
                    print(f"\n2. COST SAVINGS (within 10% of {other_method}'s best):")
                    print(f"   MCTS cost: ${cost_sav['our_cost']:.3f} (accuracy: {cost_sav['our_accuracy']:.3f})")
                    print(f"   {other_method} cost: ${cost_sav['other_cost']:.3f} (accuracy: {cost_sav['other_accuracy']:.3f})")
                    print(f"   Savings: ${cost_sav['absolute']:.3f} ({cost_sav['percentage']:.3f}%)")
                else:
                    print(f"\n2. COST SAVINGS (within 10% of {other_method}'s best): Could not calculate")
                
                # 3. Average cost savings
                if metrics["average_cost_savings"]:
                    avg_sav = metrics["average_cost_savings"]
                    print(f"\n3. AVERAGE COST SAVINGS:")
                    if avg_sav['percentage'] is not None:
                        print(f"   Average savings: ${avg_sav['absolute']:.3f} ({avg_sav['percentage']:.3f}%)")
                    else:
                        print(f"   Average savings: ${avg_sav['absolute']:.3f} (percentage not available due to zero-cost plans)")
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
        test_output = str(base_output_dir / f"{dataset}_original_final" / "tests" / "original" / f"{dataset}_baseline_test.json")
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
        test_yaml_path = Path("/tmp") / f"{dataset}_original_final_baseline_test.yaml"
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
        eval_func = get_evaluate_func(dataset, mode="test")
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
        "mcts": [],
        "lotus": [],
        "PZ": [],
        "PZ_retrieval": [],
        "PZ_direct": [],
        "LOTUS_d": [],
        "LOTUS_r&r": []
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
    
    for method in ["simple_baseline"]:
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
                # Check the method field to determine which LOTUS variant this is
                method = entry.get("method", "")
                
                # Skip join_only method as requested
                if method == "join_only":
                    continue
                
                # Determine the target method based on the method field
                if method == "simple_map":
                    target_method = "LOTUS_d"
                elif method == "join_and_rerank":
                    target_method = "LOTUS_r&r"
                else:
                    # Fallback to original lotus for unknown methods
                    target_method = "lotus"
                
                accuracy_value = None
                for key in entry.keys():
                    if accuracy_metric in key or key in accuracy_metric:
                        accuracy_value = entry[key]
                        break
                
                if accuracy_value is not None:
                    cost = entry.get("cost") or entry.get("total_cost")
                    if cost is not None:
                        all_points[target_method].append({
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
                        all_points["PZ_direct"].append({
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
                        all_points["PZ_retrieval"].append({
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
                print(f"   MCTS best accuracy: {acc_imp['our_best']:.4f} (file: {acc_imp['our_best_file']})")
                print(f"   {other_method} best accuracy: {acc_imp['other_best']:.4f} (file: {acc_imp['other_best_file']})")
                print(f"   Improvement: +{acc_imp['absolute']:.4f} ({acc_imp['percentage']:.1f}%)")
            else:
                print("1. ACCURACY IMPROVEMENT: Could not calculate")
            
            # # 2. Cost savings within 10% accuracy
            # if metrics["cost_savings_within_10pct"]:
            #     cost_sav = metrics["cost_savings_within_10pct"]
            #     print(f"\n2. COST SAVINGS (within 10% of {other_method}'s best):")
            #     print(f"   MCTS cost: ${cost_sav['our_cost']:.6f} (accuracy: {cost_sav['our_accuracy']:.4f})")
            #     print(f"   {other_method} cost: ${cost_sav['other_cost']:.6f} (accuracy: {cost_sav['other_accuracy']:.4f})")
            #     print(f"   Savings: ${cost_sav['absolute']:.6f} ({cost_sav['percentage']:.1f}%)")
            # else:
            #     print(f"\n2. COST SAVINGS (within 10% of {other_method}'s best): Could not calculate")
            
            # # 3. Average cost savings
            # if metrics["average_cost_savings"]:
            #     avg_sav = metrics["average_cost_savings"]
            #     print(f"\n3. AVERAGE COST SAVINGS:")
            #     if avg_sav['percentage'] is not None:
            #         print(f"   Average savings: ${avg_sav['absolute']:.6f} ({avg_sav['percentage']:.1f}%)")
            #     else:
            #         print(f"   Average savings: ${avg_sav['absolute']:.6f} (percentage not available due to zero-cost plans)")
            #     print(f"   Based on {avg_sav['comparisons_count']} comparisons")
            # else:
            #     print("\n3. AVERAGE COST SAVINGS: Could not calculate")
    
    print("="*60)
    
    # Save test frontier summary to file
    base_output_dir = Path(VOLUME_MOUNT_PATH) / "outputs"
    summary_path = base_output_dir / f"{dataset}_original_final" / "test_frontier_summary.json"
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


def generate_accuracy_gain_table(all_points: Dict[str, List[Dict[str, float]]], methods: List[str]) -> Dict[str, Any]:
    """
    Generate a table showing the best accuracy gain from MCTS to other methods.
    
    Args:
        all_points: Dictionary with method names as keys and lists of {cost, accuracy} dicts as values
        methods: List of method names (including original)
        
    Returns:
        Dictionary containing accuracy gain data for each method
    """
    accuracy_gain_table = {}
    
    # Get MCTS best accuracy and file
    mcts_points = all_points.get("mcts", [])
    if not mcts_points:
        return None
    
    mcts_valid_points = [p for p in mcts_points if p["accuracy"] is not None]
    if not mcts_valid_points:
        return None
    
    # Find MCTS best plan (highest accuracy)
    mcts_best_plan = max(mcts_valid_points, key=lambda x: x["accuracy"])
    mcts_best_accuracy = mcts_best_plan["accuracy"]
    mcts_best_file = mcts_best_plan.get("file", "unknown")
    
    # Calculate accuracy gain for each other method
    for method in methods:
        if method == "mcts":
            continue
            
        method_points = all_points.get(method, [])
        if not method_points:
            accuracy_gain_table[method] = {"error": "No data available"}
            continue
        
        method_valid_points = [p for p in method_points if p["accuracy"] is not None]
        if not method_valid_points:
            accuracy_gain_table[method] = {"error": "No valid accuracy data"}
            continue
        
        # Find method's best plan (highest accuracy)
        method_best_plan = max(method_valid_points, key=lambda x: x["accuracy"])
        method_best_accuracy = method_best_plan["accuracy"]
        method_best_file = method_best_plan.get("file", "unknown")
        
        # Calculate gain
        gain = mcts_best_accuracy - method_best_accuracy
        gain_pct = (gain / method_best_accuracy) * 100 if method_best_accuracy > 0 else 0
        
        accuracy_gain_table[method] = {
            "mcts_best": mcts_best_accuracy,
            "mcts_best_file": mcts_best_file,
            "other_best": method_best_accuracy,
            "other_best_file": method_best_file,
            "gain": gain,
            "gain_pct": gain_pct,
            "error": None
        }
    
    return accuracy_gain_table


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
            "mcts": [],
            "lotus": [],
            "PZ_direct": [],
            "PZ_retrieval": [],
            "PZ": [],
            "LOTUS_d": [],
            "LOTUS_r&r": []
        }
        
        # Load test_frontier_summary.json from dataset_original folder
        summary_file = base_output_dir / f"{dataset}_original_final" / "test_frontier_summary.json"
        
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
            
            # Extract results for each method
            if "results" in summary_data:
                # Handle original baseline
                if "original" in summary_data["results"] and summary_data["results"]["original"].get("success"):
                    original_result = summary_data["results"]["original"]
                    if "cost" in original_result and "accuracy" in original_result:
                        point_data = {
                            "cost": original_result["cost"],
                            "accuracy": original_result["accuracy"]
                        }
                        # Include file field if available
                        if "file" in original_result:
                            point_data["file"] = original_result["file"]
                        all_points["original"].append(point_data)
                    print(f"  ✅ Loaded {len(all_points['original'])} test points from original")
                else:
                    print("  ⚠️  No successful results found for original")
                
                # Handle other methods
                for method in METHODS:
                    if method in summary_data["results"] and summary_data["results"][method].get("success"):
                        method_results = summary_data["results"][method].get("results", [])
                        for point in method_results:
                            if "cost" in point and "accuracy" in point:
                                point_data = {
                                    "cost": point["cost"],
                                    "accuracy": point["accuracy"]
                                }
                                # Include file field if available (especially for MCTS)
                                if "file" in point:
                                    point_data["file"] = point["file"]
                                all_points[method].append(point_data)
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
                
                    # Check the method field to determine which LOTUS variant this is
                    method = entry.get("method", "")
                    
                    # Skip join_only method as requested
                    if method == "join_only":
                        continue
                    
                    # Determine the target method based on the method field
                    if method == "simple_map":
                        target_method = "LOTUS_d"
                    elif method == "join_and_rerank":
                        target_method = "LOTUS_r&r"
                    else:
                        # Fallback to original lotus for unknown methods
                        target_method = "lotus"
                    
                    accuracy_value = None
                    for key in entry.keys():
                        if accuracy_metric in key or key in accuracy_metric:
                            accuracy_value = entry[key]
                            break
                    
                    if accuracy_value is not None:
                        cost = entry.get("cost") or entry.get("total_cost")
                        if cost is not None:
                            all_points[target_method].append({
                                "cost": cost,
                                "accuracy": accuracy_value
                            })
            
            print(f"  ✅ Loaded {len(all_points['LOTUS_d'])} test points from LOTUS-d, {len(all_points['LOTUS_r&r'])} from LOTUS-r&r")
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
                            all_points["PZ_direct"].append({
                                "cost": config_data["plan_execution_cost"],
                                "accuracy": accuracy_value
                            })
                
                print(f"  ✅ Loaded {len(all_points['PZ_direct'])} test points from PZ direct")
            
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
                            all_points["PZ_retrieval"].append({
                                "cost": config_data["plan_execution_cost"],
                                "accuracy": accuracy_value
                            })
                
                print(f"  ✅ Loaded {len(all_points['PZ_retrieval'])} test points from PZ retrieval")
            else:
                for config_name, config_data in pz_data.items():
                    if isinstance(config_data, dict) and config_name != "metadata":
                        accuracy_value = None
                        for key in config_data.keys():
                            if accuracy_metric in key or key in accuracy_metric:
                                accuracy_value = config_data[key]
                                break
                        
                        if accuracy_value is not None and "plan_execution_cost" in config_data:
                            all_points["PZ"].append({
                                "cost": config_data["plan_execution_cost"],
                                "accuracy": accuracy_value
                            })
                
                print(f"  ✅ Loaded {len(all_points['PZ'])} test points from PZ")
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
            print(f"Original best accuracy: {original_best_accuracy:.3f}")
        print()
        
        for method in methods:
            best_info = best_matrix_result["method_info"][method]
            # Only get avg_info and cov_info if the method exists in those matrices
            avg_info = avg_matrix_result["method_info"].get(method, {"valid_points": 0, "points_above_original": 0})
            cov_info = coverage_matrix_result["method_info"].get(method, {"valid_points": 0, "points_above_original": 0})
            print(f"{method:15} | Total points: {best_info['valid_points']} | Points above original: {best_info['points_above_original']}")
        print()
        
        # Print matrix headers
        header = f"{'Method':>15}"
        for method in methods:
            header += f"{method:>25}"
        
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
                    row += f"{'None':>20}"
                elif value == "--":
                    row += f"{'--':>20}"
                elif value == "n/a":
                    row += f"{'n/a':>20}"
                elif value == "—":
                    row += f"{'—':>20}"
                elif isinstance(value, dict):
                    # Format as "absolute (ratio)x (savings_pct%)" - shows what fraction we use and what percentage we save
                    row += f"{value['absolute']:>8.3f} ({value['ratio']:>4.3f}x, {value['savings_pct']:>4.3f}%)"
                else:
                    row += f"{value:>20}"
            print(row)
        
        # Print Average Cost Savings Matrix
        print("\nAVERAGE COST SAVINGS MATRIX:")
        # Create header for average matrix (only methods that exist in avg_matrix)
        avg_methods = avg_matrix_result["methods"]
        avg_header = f"{'Method':>15}"
        for method in avg_methods:
            avg_header += f"{method:>25}"
        print("-" * len(avg_header))
        print(avg_header)
        print("-" * len(avg_header))
        
        for method_a in avg_methods:
            row = f"{method_a:>15}"
            for method_b in avg_methods:
                value = avg_matrix_result["matrix"][method_a][method_b]
                if value is None:
                    row += f"{'None':>20}"
                elif value == "--":
                    row += f"{'--':>20}"
                elif value == "n/a":
                    row += f"{'n/a':>20}"
                elif value == "—":
                    row += f"{'—':>20}"
                elif isinstance(value, dict):
                    # Format as "absolute (ratio)x (savings_pct%)" - shows what fraction we use and what percentage we save
                    row += f"{value['absolute']:>8} ({value['ratio']:>4}x)"
                else:
                    row += f"{value:>20}"
            print(row)
        
        # Print Coverage Matrix
        print("\nCOVERAGE MATRIX:")
        # Create header for coverage matrix (only methods that exist in coverage_matrix)
        cov_methods = coverage_matrix_result["methods"]
        cov_header = f"{'Method':>15}"
        for method in cov_methods:
            cov_header += f"{method:>12}"
        print("-" * len(cov_header))
        print(cov_header)
        print("-" * len(cov_header))
        
        for method_a in cov_methods:
            row = f"{method_a:>15}"
            for method_b in cov_methods:
                value = coverage_matrix_result["matrix"][method_a][method_b]
                if value is None:
                    row += f"{'None':>12}"
                elif value == "--":
                    row += f"{'--':>12}"
                elif value == "n/a":
                    row += f"{'n/a':>12}"
                elif value == "—":
                    row += f"{'—':>12}"
                else:
                    row += f"{value:>12.3f}"
            print(row)
        
        print("="*80)
        
        # Generate and print accuracy gain table
        print("\n" + "="*80)
        print("BEST ACCURACY GAIN FROM MCTS TO OTHER METHODS")
        print("="*80)
        
        # For accuracy gain table, we need to include original if it exists in all_points
        methods_with_original = methods
        if "original" in all_points and all_points["original"] and "original" not in methods:
            methods_with_original = methods + ["original"]
        accuracy_gain_table = generate_accuracy_gain_table(all_points, methods_with_original)
        if accuracy_gain_table:
            print("\nACCURACY GAIN TABLE:")
            print("-" * 60)
            print(f"{'Method':>15} {'MCTS Best':>12} {'Other Best':>12} {'Gain':>12} {'Gain %':>10}")
            print("-" * 60)
            
            for method, data in accuracy_gain_table.items():
                if data["error"]:
                    print(f"{method:>15} {'ERROR':>12} {'ERROR':>12} {'ERROR':>12} {'ERROR':>10}")
                else:
                    print(f"{method:>15} {data['mcts_best']:>12.3f} {data['other_best']:>12.3f} {data['gain']:>+12.3f} {data['gain_pct']:>+9.3f}%")
            
            print("-" * 60)
        else:
            print("❌ Could not generate accuracy gain table")
        
        print("="*80)
        
        # Save all matrices to files
        timestamp = datetime.now().isoformat()
        
        # Save Best Cost Savings Matrix
        best_matrix_path = base_output_dir / f"{dataset}_original_final" / "best_cost_savings_matrix.json"
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
        plot_matrix(best_matrix_result, 'best_cost_savings', dataset, base_output_dir / f"{dataset}_original_final")
        
        # Save Average Cost Savings Matrix
        avg_matrix_path = base_output_dir / f"{dataset}_original_final" / "avg_cost_savings_matrix.json"
        
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
        plot_matrix(avg_matrix_result, 'avg_cost_savings', dataset, base_output_dir / f"{dataset}_original_final")
        
        # Save Coverage Matrix
        coverage_matrix_path = base_output_dir / f"{dataset}_original_final" / "coverage_matrix.json"
        
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
        plot_matrix(coverage_matrix_result, 'coverage', dataset, base_output_dir / f"{dataset}_original_final")
        
        # Save accuracy gain table
        if accuracy_gain_table:
            accuracy_gain_path = base_output_dir / f"{dataset}_original_final" / "accuracy_gain_table.json"
            accuracy_gain_data = {
                "dataset": dataset,
                "timestamp": timestamp,
                "mcts_best_accuracy": accuracy_gain_table[list(accuracy_gain_table.keys())[0]]["mcts_best"] if accuracy_gain_table else None,
                "accuracy_gains": accuracy_gain_table
            }
            
            with open(accuracy_gain_path, 'w') as f:
                json.dump(accuracy_gain_data, f, indent=2)
            
            print(f"\n✅ All matrices and accuracy gain table saved and visualized:")
            print(f"   Best cost savings matrix: {best_matrix_path}")
            print(f"   Average cost savings matrix: {avg_matrix_path}")
            print(f"   Coverage matrix: {coverage_matrix_path}")
            print(f"   Accuracy gain table: {accuracy_gain_path}")
            print(f"   Matrix plots saved in: {base_output_dir / f'{dataset}_original_final'}")
        else:
            print(f"\n✅ All matrices saved and visualized:")
            print(f"   Best cost savings matrix: {best_matrix_path}")
            print(f"   Average cost savings matrix: {avg_matrix_path}")
            print(f"   Coverage matrix: {coverage_matrix_path}")
            print(f"   Matrix plots saved in: {base_output_dir / f'{dataset}_original_final'}")
        
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
        method: Method to run ('simple_baseline','mcts', or 'all' for all methods)
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
                    print(f"   Original best accuracy: {matrix_result['original_best_accuracy']:.3f}")
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