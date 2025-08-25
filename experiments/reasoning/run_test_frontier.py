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
from typing import Any, Dict
import modal
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt

from docetl.runner import DSLRunner
from experiments.reasoning.utils import app, volume, VOLUME_MOUNT_PATH, image
from experiments.reasoning.evaluation.utils import dataset_accuracy_metrics, get_evaluate_func

# Dataset configurations
DATASETS = ["cuad", "blackvault", "game_reviews", "sustainability", "biodex", "medec", "facility"]
METHODS = ["simple_baseline", "baseline", "mcts"]

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
        experiment_dir = base_output_dir / f"{dataset}_{method}"
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
                runner = DSLRunner.from_yaml(str(test_yaml_path))
                runner.load()
                
                if runner.last_op_container:
                    data, _, _ = runner.last_op_container.next()
                    runner.save(data)
                
                total_cost = runner.total_cost
                runner.reset_env()
                
                print(f"✅ Pipeline completed. Cost: ${total_cost:.6f}")
                
                # Evaluate accuracy
                eval_func = get_evaluate_func(dataset)
                if eval_func:
                    # Evaluate results
                    accuracy_results = eval_func(base_name, test_output)
                    
                    # Get the appropriate accuracy metric
                    accuracy_metric = dataset_accuracy_metrics.get(dataset, "accuracy")
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
            "pz": []
        }
        
        # Method colors
        method_colors = {
            "original": "#ffd700",           # Gold/Yellow
            "simple_baseline": "#2ecc71",    # Green
            "baseline": "#1f77b4",           # Blue (darker blue)
            "mcts": "#d62728",               # Red (darker red)
            "lotus": "#c27cf3",              # Light purple
            "pz": "#ff0b50"                  # Pink/magenta
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
                if "lotus_test.json" in entry.get("file", ""):
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
            
            print(f"  ✅ Loaded {len(all_points['lotus'])} test points from LOTUS")
        else:
            print(f"  ⚠️  No LOTUS evaluation found at {lotus_file}")
        
        # Load PZ evaluation data if available (from local filesystem in Modal image)
        pz_file = Path("experiments/reasoning/othersystems") / dataset / "pz_evaluation.json"
        
        if pz_file.exists():
            print(f"  📄 Found PZ evaluation at: {pz_file}")
            with open(pz_file, 'r') as f:
                pz_data = json.load(f)
            
            # Extract results from each PZ configuration
            for config_name, config_data in pz_data.items():
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
                        print(f"  ❌ Could not find accuracy metric '{accuracy_metric}' in PZ config '{config_name}'")
                        print(f"     Available keys: {list(config_data.keys())}")
                        continue
                    
                    if "plan_execution_cost" in config_data:
                        all_points["pz"].append({
                            "cost": config_data["plan_execution_cost"],
                            "accuracy": accuracy_value
                        })
            
            print(f"  ✅ Loaded {len(all_points['pz'])} test points from PZ")
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
        
        if all_costs:
            min_cost = min(all_costs)
            max_cost = max(all_costs)
            
            # Set x-axis limits with some padding
            ax.set_xlim(min_cost * 0.8, max_cost * 1.2)
            
            # Use standard log scale ticks and formatting
            import matplotlib.ticker as ticker
            import numpy as np
            
            # Create explicit ticks based on the data range
            # Find the order of magnitude range
            min_order = np.floor(np.log10(min_cost * 0.8))
            max_order = np.ceil(np.log10(max_cost * 1.2))
            
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
        
        # Commit changes to Modal volume
        volume.commit()
        
        return {
            "success": True,
            "plot_path": str(plot_path),
            "total_points": total_points,
            "points_by_method": {m: len(p) for m, p in all_points.items()}
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
        
        # Create output directory
        Path(test_output).parent.mkdir(parents=True, exist_ok=True)
        
        # Save modified YAML to /tmp
        test_yaml_path = Path("/tmp") / f"{dataset}_original_baseline_test.yaml"
        with open(test_yaml_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False)
        
        # Run the pipeline
        print("🚀 Running original baseline pipeline...")
        runner = DSLRunner.from_yaml(str(test_yaml_path))
        runner.load()
        
        if runner.last_op_container:
            data, _, _ = runner.last_op_container.next()
            runner.save(data)
        
        total_cost = runner.total_cost
        runner.reset_env()
        
        print(f"✅ Pipeline completed. Cost: ${total_cost:.6f}")
        
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


@app.local_entrypoint()
def main(dataset: str = "cuad", method: str = "all", plot_only: bool = False):
    """
    Main entrypoint for running test frontier evaluation.
    
    Args:
        dataset: Dataset to process ('cuad', 'blackvault', etc., or 'all' for all datasets)
        method: Method to run ('simple_baseline', 'baseline', 'mcts', or 'all' for all methods)
        plot_only: If True, only generate the plot without running evaluations
    """
    if dataset not in DATASETS + ["all"]:
        print(f"❌ Invalid dataset: {dataset}")
        print(f"   Valid options: {', '.join(DATASETS + ['all'])}")
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