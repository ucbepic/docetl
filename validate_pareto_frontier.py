#!/usr/bin/env python3
"""
Validate Pareto frontier points on validation set.

This script:
1. Loads pareto_frontier.json from the specified folder
2. For each frontier point, gets the corresponding YAML pipeline file
3. Modifies the YAML to use validation data instead of train data
4. Runs the pipeline and calculates cost/accuracy on validation set
5. Finds the true Pareto frontier points from validation results
6. Saves results to pareto_frontier_validate.json
7. Generates a plot with frontier points in blue and others in gray
"""

import json
import yaml
import traceback
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import modal

# Import DocETL components
from docetl.runner import DSLRunner
from experiments.reasoning.utils import app, volume, VOLUME_MOUNT_PATH, image
from experiments.reasoning.evaluation.utils import dataset_accuracy_metrics, get_evaluate_func


def find_pareto_frontier_points(points: List[Dict[str, Any]], accuracy_metric: str) -> List[Dict[str, Any]]:
    """
    Find Pareto frontier points from a list of cost/accuracy points.
    
    Args:
        points: List of dictionaries with 'cost' and accuracy metric keys
        accuracy_metric: Name of the accuracy metric to use
        
    Returns:
        List of Pareto frontier points
    """
    if not points:
        return []
    
    # Filter out points with missing data
    valid_points = []
    for point in points:
        if (point.get("cost") is not None and 
            point.get(accuracy_metric) is not None and
            point["cost"] > 0):  # Avoid zero/negative costs
            valid_points.append(point)
    
    if not valid_points:
        return []
    
    # Sort by cost (ascending) and accuracy (descending)
    valid_points.sort(key=lambda x: (x["cost"], -x[accuracy_metric]))
    
    # Find Pareto frontier using greedy approach
    frontier_points = []
    best_accuracy = float('-inf')
    
    for point in valid_points:
        accuracy = point[accuracy_metric]
        if accuracy > best_accuracy:
            frontier_points.append(point)
            best_accuracy = accuracy
    
    return frontier_points


def plot_validation_results(all_points: List[Dict[str, Any]], 
                           frontier_points: List[Dict[str, Any]], 
                           accuracy_metric: str, 
                           dataset: str, 
                           output_dir: Path):
    """
    Create a plot showing validation results with frontier points in blue and others in gray.
    Also shows original points as triangles in a different color.
    
    Args:
        all_points: All validation points
        frontier_points: Pareto frontier points
        accuracy_metric: Name of the accuracy metric
        dataset: Dataset name
        output_dir: Output directory for saving the plot
    """
    if not all_points:
        print("No points to plot")
        return
    
    # Use non-interactive backend for Modal
    matplotlib.use('Agg')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Separate frontier and non-frontier points
    frontier_costs = [p["cost"] for p in frontier_points]
    frontier_accuracies = [p[accuracy_metric] for p in frontier_points]
    
    non_frontier_points = []
    frontier_set = set((p["cost"], p[accuracy_metric]) for p in frontier_points)
    for point in all_points:
        # Check if point has the required keys and is not in frontier
        if (point.get("cost") is not None and 
            point.get(accuracy_metric) is not None and
            (point["cost"], point[accuracy_metric]) not in frontier_set):
            non_frontier_points.append(point)
    
    non_frontier_costs = [p["cost"] for p in non_frontier_points]
    non_frontier_accuracies = [p[accuracy_metric] for p in non_frontier_points]
    
    # Extract original points
    original_points = []
    for point in all_points:
        if "original_point" in point and point["original_point"]:
            orig_point = point["original_point"]
            if orig_point.get("cost") is not None and orig_point.get("accuracy") is not None:
                original_points.append({
                    "cost": orig_point["cost"],
                    "accuracy": orig_point["accuracy"],
                    "file": point["file"]  # Use validation file name for labeling
                })
    
    original_costs = [p["cost"] for p in original_points]
    original_accuracies = [p["accuracy"] for p in original_points]
    
    # Plot original points as triangles in orange
    if original_costs:
        ax.scatter(original_costs, original_accuracies, 
                  color='orange', s=80, alpha=0.7, label='Original Points', 
                  marker='^', edgecolors='black', linewidth=0.5)
    
    # Plot non-frontier points in gray
    if non_frontier_costs:
        ax.scatter(non_frontier_costs, non_frontier_accuracies, 
                  color='gray', s=60, alpha=0.6, label='Non-Frontier', 
                  edgecolors='black', linewidth=0.5)
    
    # Plot frontier points in blue
    if frontier_costs:
        ax.scatter(frontier_costs, frontier_accuracies, 
                  color='blue', s=100, alpha=0.8, label='Pareto Frontier', 
                  edgecolors='black', linewidth=1)
    
    # Add labels for all validation points (both frontier and non-frontier)
    for point in all_points:
        if point.get("cost") is not None and point.get(accuracy_metric) is not None:
            cost = point["cost"]
            accuracy = point[accuracy_metric]
            file_name = point.get("file", "unknown")
            
            # Add text annotation
            ax.annotate(file_name, (cost, accuracy), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Add labels for original points (triangles)
    for point in original_points:
        cost = point["cost"]
        accuracy = point["accuracy"]
        file_name = point.get("file", "unknown")
        
        # Add text annotation for original points
        ax.annotate(f"{file_name} (orig)", (cost, accuracy), 
                   xytext=(5, -15), textcoords='offset points',
                   fontsize=8, alpha=0.8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7))
    
    # Set log scale for x-axis (cost)
    ax.set_xscale('log')
    
    # Get the range of costs to set appropriate ticks
    all_costs = [p["cost"] for p in all_points]
    if original_costs:
        all_costs.extend(original_costs)
    
    if all_costs:
        min_cost = min(all_costs)
        max_cost = max(all_costs)
        
        # Handle zero or negative costs for log scale
        if min_cost <= 0:
            min_cost = max(0.01, min_cost)
            print(f"⚠️  Found zero or negative cost, using {min_cost} for log scale")
        
        # Set x-axis limits with some padding
        ax.set_xlim(min_cost * 0.8, max_cost * 1.2)
        
        # Create explicit ticks based on the data range
        min_order = np.floor(np.log10(min_cost * 0.8))
        max_order = np.ceil(np.log10(max_cost * 1.2))
        
        # Handle infinite values
        if not np.isfinite(min_order) or not np.isfinite(max_order):
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
        
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    
    # Labels and title
    ax.set_xlabel('Cost ($) - Log Scale', fontsize=12)
    ax.set_ylabel(f'{accuracy_metric.replace("_", " ").title()}', fontsize=12)
    ax.set_title(f'Validation Results vs Original Points - {dataset.upper()} Dataset', fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.grid(True, which='minor', alpha=0.1, linestyle=':')
    
    # Add legend
    ax.legend(loc='best', frameon=True, shadow=True)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot
    plot_path = output_dir / f"{dataset}_validation_frontier.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Validation frontier plot saved: {plot_path}")


@app.function(
    image=image, 
    secrets=[modal.Secret.from_dotenv()], 
    volumes={VOLUME_MOUNT_PATH: volume}, 
    timeout=60 * 60
)
def validate_pareto_frontier_remote(folder_path: str, dataset: str) -> Dict[str, Any]:
    """
    Validate Pareto frontier points on validation set.
    
    Args:
        folder_path: Path to folder containing pareto_frontier.json
        dataset: Dataset name
        
    Returns:
        Dictionary with validation results
    """
    try:
        print(f"\n{'='*60}")
        print(f"Validating Pareto frontier for {dataset}")
        print(f"Folder: {folder_path}")
        print(f"{'='*60}\n")
        
        # Convert to Modal volume path
        folder_path = Path(VOLUME_MOUNT_PATH) / folder_path
        pareto_file = folder_path / "pareto_frontier.json"
        
        # Check if pareto frontier file exists
        if not pareto_file.exists():
            print(f"❌ Pareto frontier file not found: {pareto_file}")
            return {
                "success": False,
                "error": f"Pareto frontier file not found: {pareto_file}"
            }
        
        # Load pareto frontier
        with open(pareto_file, 'r') as f:
            pareto_data = json.load(f)
        
        frontier_points = pareto_data
        print(f"📊 Found {len(frontier_points)} frontier points")
        
        if not frontier_points:
            return {
                "success": False,
                "error": "No frontier points found in pareto file"
            }
        
        # Get accuracy metric for dataset
        accuracy_metric = dataset_accuracy_metrics.get(dataset, "accuracy")
        print(f"📈 Using accuracy metric: {accuracy_metric}")
        
        # Get evaluation function
        eval_func = get_evaluate_func(dataset)
        if not eval_func:
            return {
                "success": False,
                "error": f"No evaluation function found for dataset: {dataset}"
            }
        
        # Process each frontier point
        validation_results = []
        for i, point in enumerate(frontier_points):
            print(f"\n--- Processing frontier point {i+1}/{len(frontier_points)} ---")
            
            # Extract file name from point
            point_file = point.get("config_path")
            if not point_file:
                print("⚠️  No file field in frontier point, skipping")
                continue
            
            print(f"📄 Processing {point_file}")
            yaml_file = Path(point_file)
            # Try alternative locations if not found
            if not yaml_file.exists():
               
                print(f"⚠️  YAML file not found: {yaml_file}, skipping")
                continue
            
            print(f"📄 Using YAML file: {yaml_file}")
            
            # Load and modify YAML
            with open(yaml_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Change dataset path from train to validation
            if 'datasets' in config:
                for dataset_name, dataset_config in config['datasets'].items():
                    original_path = dataset_config.get('path', '')
                    # Replace 'train' with 'val' in the path
                    val_path = original_path.replace('/train/', '/val/')
                    config['datasets'][dataset_name]['path'] = val_path
                    print(f"📂 Changed dataset path to: {val_path}")
            
            base_name = Path(point_file).stem
            # Set output path for validation results
            val_output = str(folder_path / "validation_results" / f"{base_name}_validation.json")
            config['pipeline']['output']['path'] = val_output
            print(f"📤 Output path: {val_output}")
            
            # Create output directory
            Path(val_output).parent.mkdir(parents=True, exist_ok=True)
            
            # Save modified YAML to temporary file
            temp_yaml_path = folder_path / f"{base_name}_validation_temp.yaml"
            with open(temp_yaml_path, 'w') as f:
                yaml.dump(config, f, sort_keys=False)
            
            try:
                # Run the pipeline
                print("🚀 Running pipeline...")
                start_time = time.time()
                runner = DSLRunner.from_yaml(str(temp_yaml_path))
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
                accuracy_results = eval_func(base_name, val_output)
                accuracy = accuracy_results.get(accuracy_metric, 0.0)
                
                print(f"📈 Accuracy ({accuracy_metric}): {accuracy:.4f}")
                
                # Store result
                validation_results.append({
                    "file": base_name,
                    "cost": total_cost,
                    "accuracy": accuracy,
                    "accuracy_metric": accuracy_metric,
                    "latency": latency,
                    "original_point": point
                })
                
            except Exception as e:
                print(f"❌ Error running pipeline: {e}")
                traceback.print_exc()
                validation_results.append({
                    "file": base_name,
                    "error": str(e),
                    "original_point": point
                })
            
            finally:
                # Clean up temporary YAML file
                if temp_yaml_path.exists():
                    temp_yaml_path.unlink()
        
        # Find Pareto frontier from validation results
        print(f"\n📊 Finding Pareto frontier from {len(validation_results)} validation results...")
        
        # Filter out failed results
        valid_results = [r for r in validation_results if "error" not in r]
        print(f"📈 Valid results: {len(valid_results)}")
        
        if not valid_results:
            return {
                "success": False,
                "error": "No valid validation results found"
            }
        
        # Find Pareto frontier points
        frontier_points = find_pareto_frontier_points(valid_results, accuracy_metric)
        print(f"🎯 Found {len(frontier_points)} Pareto frontier points")
        
        # Create validation frontier data
        validation_frontier_data = {
            "dataset": dataset,
            "accuracy_metric": accuracy_metric,
            "total_points": len(valid_results),
            "frontier_points": len(frontier_points),
            "validation_results": validation_results,
            "frontier_points": frontier_points,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save validation frontier results
        validation_file = folder_path / "pareto_frontier_validate.json"
        with open(validation_file, 'w') as f:
            json.dump(validation_frontier_data, f, indent=2)
        
        print(f"✅ Validation results saved to: {validation_file}")
        
        # Generate plot
        print("\n📊 Generating validation frontier plot...")
        plot_validation_results(valid_results, frontier_points, accuracy_metric, dataset, folder_path)
        
        # Commit changes to Modal volume
        volume.commit()
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"VALIDATION SUMMARY FOR {dataset.upper()}")
        print(f"{'='*60}")
        print(f"Total points processed: {len(validation_results)}")
        print(f"Valid results: {len(valid_results)}")
        print(f"Pareto frontier points: {len(frontier_points)}")
        
        if frontier_points:
            print(f"\nFrontier points:")
            for i, point in enumerate(frontier_points):
                print(f"  {i+1}. {point['file']}: Cost=${point['cost']:.6f}, {accuracy_metric}={point['accuracy']:.4f}")
        
        return {
            "success": True,
            "dataset": dataset,
            "total_points": len(validation_results),
            "valid_points": len(valid_results),
            "frontier_points": len(frontier_points),
            "validation_file": str(validation_file),
            "frontier_points_data": frontier_points
        }
        
    except Exception as e:
        print(f"❌ Error in validate_pareto_frontier: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


@app.function(
    image=image, 
    secrets=[modal.Secret.from_dotenv()], 
    volumes={VOLUME_MOUNT_PATH: volume}, 
    timeout=60 * 10
)
def plot_from_existing_validation(folder_path: str, dataset: str) -> Dict[str, Any]:
    """
    Plot validation results from existing pareto_frontier_validate.json file.
    
    Args:
        folder_path: Path to folder containing pareto_frontier_validate.json
        dataset: Dataset name
        
    Returns:
        Dictionary with plot generation status
    """
    try:
        print(f"\n📊 Plotting from existing validation results for {dataset}")
        print(f"Folder: {folder_path}")
        
        # Convert to Modal volume path
        folder_path = Path(VOLUME_MOUNT_PATH) / folder_path
        validation_file = folder_path / "pareto_frontier_validate.json"
        print(f"Validation file: {validation_file}")
        print(f"Folder path: {folder_path}")
        print(f"Folder exists: {folder_path.exists()}")
        print(f"Validation file exists: {validation_file.exists()}")
        
        # List contents of the folder to debug
        if folder_path.exists():
            print(f"Folder contents:")
            for item in folder_path.iterdir():
                print(f"  - {item.name}")
        else:
            print(f"❌ Folder does not exist: {folder_path}")
        
        # Check if validation file exists
        if not validation_file.exists():
            print(f"❌ Validation file not found: {validation_file}")
            return {
                "success": False,
                "error": f"Validation file not found: {validation_file}"
            }
        
        # Load validation results
        with open(validation_file, 'r') as f:
            validation_data = json.load(f)
        
        # Extract data
        all_points = validation_data.get("validation_results", [])
        frontier_points = validation_data.get("frontier_points", [])
        accuracy_metric = validation_data.get("accuracy_metric", "accuracy")
        
        print(f"📊 Found {len(all_points)} total points and {len(frontier_points)} frontier points")
        print(f"📊 Accuracy metric: {accuracy_metric}")
        print(f"📊 Sample validation result: {all_points[0] if all_points else 'None'}")
        
        if not all_points:
            return {
                "success": False,
                "error": "No validation results found in file"
            }
        
        # Filter out failed results - check for both 'accuracy' and the specific metric
        valid_results = []
        for r in all_points:
            if "error" not in r and r.get("cost") is not None:
                # Check if accuracy is stored in 'accuracy' field or the specific metric field
                accuracy_value = r.get("accuracy") or r.get(accuracy_metric)
                if accuracy_value is not None:
                    # Normalize the result to use the correct metric name
                    normalized_result = r.copy()
                    normalized_result[accuracy_metric] = accuracy_value
                    valid_results.append(normalized_result)
        
        print(f"📈 Valid results: {len(valid_results)}")
        
        # Debug: show why results are invalid
        if not valid_results and all_points:
            print("🔍 Debugging invalid results:")
            for i, result in enumerate(all_points[:3]):  # Show first 3 results
                print(f"  Result {i}: {result}")
                has_error = "error" in result
                has_cost = result.get("cost") is not None
                has_accuracy = result.get("accuracy") is not None
                has_metric = result.get(accuracy_metric) is not None
                print(f"    Has error: {has_error}, Has cost: {has_cost}, Has accuracy: {has_accuracy}, Has {accuracy_metric}: {has_metric}")
        
        if not valid_results:
            return {
                "success": False,
                "error": "No valid validation results found"
            }
        
        # If no frontier points were found, calculate them from valid results
        if not frontier_points and valid_results:
            print("📊 No frontier points found, calculating from valid results...")
            frontier_points = find_pareto_frontier_points(valid_results, accuracy_metric)
            print(f"📊 Calculated {len(frontier_points)} frontier points")
        
        # Save frontier points to a new JSON file
        if frontier_points:
            frontier_file = folder_path / f"pareto_frontier_{dataset}_validate.json"
            frontier_data = {
                "dataset": dataset,
                "accuracy_metric": accuracy_metric,
                "frontier_points": frontier_points,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source": "calculated_from_validation_results"
            }
            
            with open(frontier_file, 'w') as f:
                json.dump(frontier_data, f, indent=2)
            
            print(f"💾 Frontier points saved to: {frontier_file}")
        
        # Generate plot
        print("📊 Generating validation frontier plot...")
        plot_validation_results(valid_results, frontier_points, accuracy_metric, dataset, folder_path)
        
        # Commit changes to Modal volume
        volume.commit()
        
        return {
            "success": True,
            "dataset": dataset,
            "total_points": len(all_points),
            "valid_points": len(valid_results),
            "frontier_points": len(frontier_points),
            "plot_path": str(folder_path / f"{dataset}_validation_frontier.png")
        }
        
    except Exception as e:
        print(f"❌ Error plotting from existing validation: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


@app.local_entrypoint()
def main(folder_path: str, dataset: str):
    """
    Main entrypoint for the validation script.
    
    Args:
        folder_path: Path to folder containing pareto_frontier.json (relative to Modal volume)
        dataset: Dataset name (e.g., 'cuad', 'biodex', etc.)
    """
    print(f"🚀 Starting validation for {dataset} in folder: {folder_path}")
    
    # Run validation on Modal
    result = validate_pareto_frontier_remote.remote(folder_path, dataset)
    
    if result["success"]:
        print(f"\n✅ Validation completed successfully!")
        print(f"   Total points processed: {result['total_points']}")
        print(f"   Valid points: {result['valid_points']}")
        print(f"   Pareto frontier points: {result['frontier_points']}")
        print(f"   Results saved to: {result['validation_file']}")
        
        if result.get('frontier_points_data'):
            print(f"\nFrontier points:")
            for i, point in enumerate(result['frontier_points_data']):
                print(f"  {i+1}. {point['file']}: Cost=${point['cost']:.6f}, {point['accuracy_metric']}={point['accuracy']:.4f}")
    else:
        print(f"\n❌ Validation failed: {result.get('error', 'Unknown error')}")


@app.local_entrypoint()
def plot_only(folder_path: str, dataset: str):
    """
    Entrypoint for just plotting from existing validation results.
    
    Args:
        folder_path: Path to folder containing pareto_frontier_validate.json (relative to Modal volume)
        dataset: Dataset name (e.g., 'cuad', 'biodex', etc.)
    """
    print(f"📊 Plotting validation results for {dataset} in folder: {folder_path}")
    
    # Run plotting on Modal
    result = plot_from_existing_validation.remote(folder_path, dataset)
    
    if result["success"]:
        print(f"\n✅ Plot generated successfully!")
        print(f"   Total points: {result['total_points']}")
        print(f"   Valid points: {result['valid_points']}")
        print(f"   Pareto frontier points: {result['frontier_points']}")
        print(f"   Plot saved to: {result['plot_path']}")
    else:
        print(f"\n❌ Plot generation failed: {result.get('error', 'Unknown error')}")
