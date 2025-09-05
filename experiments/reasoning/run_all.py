#!/usr/bin/env python3
"""
Bulk experiment runner for DocETL reasoning experiments using Modal.

- Reads a JSON config describing which datasets to run and iteration counts
- Spawns both baseline and MCTS runs on Modal concurrently per dataset
- Waits for all runs to complete and prints a simple summary

Config schema (example):
{
  "experiments": [
    {
      "dataset": "medec",
      "baseline": { "iterations": 3 },
      "mcts": { "max_iterations": 30 }
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal
import traceback
from datetime import datetime
import os
import matplotlib
import yaml
import matplotlib.pyplot as plt
from docetl.runner import DSLRunner
from docetl.utils import extract_output_from_json
from experiments.reasoning.utils import create_original_query_result
from docetl.reasoning_optimizer.directives import AVAILABLE_MODELS, OperatorFusionDirective
from copy import deepcopy

# Import the existing Modal functions and shared volume/mount from the experiment runners
from experiments.reasoning import run_mcts as mcts_mod
from experiments.reasoning import run_baseline as baseline_mod
from experiments.reasoning import run_simple_baseline as simple_baseline_mod
from experiments.reasoning.evaluation.utils import run_dataset_evaluation, get_evaluate_func, dataset_accuracy_metrics
from experiments.reasoning.utils import app, create_original_query_result, volume, VOLUME_MOUNT_PATH, image  # use the same App as the runners

@app.function(image=image, volumes={VOLUME_MOUNT_PATH: volume}, timeout=60)
def check_optimized_config_exists(original_path: str) -> Dict[str, Any]:
    print(f"Checking if original config exists at {original_path}")
    original_path = Path(original_path)
    if original_path.exists():
        with open(original_path, 'r') as f:
            config = yaml.safe_load(f)
        return {"exists": True, "config": config, "path": str(original_path)}
    else:
        return {"exists": False, "config": None, "path": str(original_path)}

from experiments.reasoning.plot_result import (
    find_pareto_frontier, calculate_hypervolume_comparison, 
    plot_pareto_frontier_comparison, dataset_metrics
)
from docetl.mcts.mcts_utils import fix_models

# Known defaults for dataset YAMLs and sample dataset inputs
DEFAULT_YAML_PATHS: Dict[str, str] = {
    "cuad": "experiments/reasoning/pipelines/cuad.yaml",
    "blackvault": "experiments/reasoning/pipelines/blackvault.yaml",
    "game_reviews": "experiments/reasoning/pipelines/game_reviews.yaml",
    "sustainability": "experiments/reasoning/pipelines/sustainability.yaml",
    "biodex": "experiments/reasoning/pipelines/biodex.yaml",
    "medec": "experiments/reasoning/pipelines/medec.yaml",
    "facility": "experiments/reasoning/pipelines/facility.yaml",
}

# Always prefer train-split defaults for MCTS sample inputs
DEFAULT_DATASET_PATHS: Dict[str, str] = {
    "cuad": "experiments/reasoning/data/train/cuad.json",
    "blackvault": "experiments/reasoning/data/train/blackvault.json",
    "game_reviews": "experiments/reasoning/data/train/game_reviews.json",
    "sustainability": "experiments/reasoning/data/train/sustainability.json",
    "biodex": "experiments/reasoning/data/train/biodex.json",
    "medec": "experiments/reasoning/data/train/medec.json",
    "facility": "experiments/reasoning/data/train/facility.json",
}

# Users can edit this CONFIG dict directly before running via Modal
CONFIG: Dict[str, Any] = {
    "experiments": [
        {
            "dataset": "medec",
            "build_first_layer": True,
            # "original_config_path": "medec/gpt_4o_mini_config.yaml",
            # "baseline": {"iterations": 10},
            # "simple_baseline": {"iterations": 10},
            # "original_cost": 0.008168,
            "mcts": {"max_iterations": 30}
        }
    ]
}

def _get_with_default(mapping: Dict[str, str], key: str, override: Optional[str]) -> str:
    if override:
        return override
    if key not in mapping:
        raise ValueError(f"No default path known for dataset '{key}'. Please provide it explicitly in the config.")
    return mapping[key]


def _spawn_baseline(
    dataset: str,
    yaml_path: str,
    *,
    experiment_name: str,
    iterations: int,
    model: Optional[str],
    data_dir: Optional[str],
    output_dir: Optional[str],
    ground_truth: Optional[str],
    original_query_result: Dict[str, Any],
):
    # Uses baseline_mod.run_baseline_remote which is bound to the shared named volume
    return baseline_mod.run_baseline_remote.spawn(
        yaml_path=yaml_path,
        data_dir=data_dir,
        output_dir=output_dir,
        model=model or baseline_mod.DEFAULT_MODEL,
        max_tpm=baseline_mod.DEFAULT_MAX_TPM,
        iterations=iterations,
        experiment_name=experiment_name,
        dataset=dataset,
        ground_truth_path=ground_truth,
        original_query_result=original_query_result,
    )


def _spawn_mcts(
    dataset: str,
    yaml_path: str,
    dataset_path: str,
    *,
    experiment_name: str,
    max_iterations: int,
    exploration_weight: Optional[float],
    model: Optional[str],
    data_dir: Optional[str],
    output_dir: Optional[str],
    ground_truth: Optional[str],
    original_query_result: Dict[str, Any],
    build_first_layer: Optional[bool] = False,
):
    # Uses mcts_mod.run_mcts_remote which is bound to the shared named volume
    return mcts_mod.run_mcts_remote.spawn(
        yaml_path=yaml_path,
        dataset_path=dataset_path,
        data_dir=data_dir,
        output_dir=output_dir,
        experiment_name=experiment_name,
        max_iterations=max_iterations,
        exploration_weight=exploration_weight if exploration_weight is not None else 1.414,
        model=model or mcts_mod.DEFAULT_MODEL,
        dataset=dataset,
        ground_truth_path=ground_truth,
        original_query_result=original_query_result,
        build_first_layer=build_first_layer,
    )


def _spawn_simple_baseline(
    dataset: str,
    *,
    experiment_name: str,
    model: Optional[str],
    output_dir: Optional[str],
    ground_truth: Optional[str],
    original_query_result: Dict[str, Any],
):
    # Uses simple_baseline_mod.run_simple_baseline_remote which is bound to the shared named volume
    return simple_baseline_mod.run_simple_baseline_remote.spawn(
        dataset=dataset,
        output_dir=output_dir,
        model=model or "o3",  # Default to o3 model
        experiment_name=experiment_name,
        ground_truth_path=ground_truth,
        original_query_result=original_query_result,
    )


@app.function(image=image, secrets=[modal.Secret.from_dotenv()], volumes={VOLUME_MOUNT_PATH: volume}, timeout=60 * 60)
def run_original_query_remote(yaml_path: str, dataset: str, experiment_name: str, 
                             data_dir: Optional[str] = None, output_dir: Optional[str] = None,
                             original_cost: Optional[float] = None) -> Dict[str, Any]:
    """Execute the original query plan once in Modal and return results."""
    
    try:
        # Set up Modal environment
        os.environ["EXPERIMENT_OUTPUT_DIR"] = str(Path(VOLUME_MOUNT_PATH) / "outputs")
        
        # Set up output directory in Modal volume
        if output_dir is None:
            output_dir = str(Path(VOLUME_MOUNT_PATH) / "outputs")
        else:
            # Ensure output_dir is within the Modal volume
            if not output_dir.startswith(VOLUME_MOUNT_PATH):
                output_dir = str(Path(VOLUME_MOUNT_PATH) / "outputs")
        
        output_path = Path(output_dir) / experiment_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Check if we should use existing results
        baseline_json_path = output_path / "original_output.json"
        baseline_yaml_path = output_path / "baseline_config.yaml"
        
        should_use_existing = False
        existing_cost = 0.0
        
        if baseline_json_path.exists() and baseline_yaml_path.exists():
            print(f"✅ Found existing original query results for {experiment_name}")
            
            # Use existing results if we have a valid cost from any source
            if original_cost is not None:
                existing_cost = original_cost
                should_use_existing = True
                print(f"   ✅ Using provided original cost: ${existing_cost:.6f}")
            else:
                # Try to get cost from existing config
                try:
                    with open(baseline_yaml_path, 'r') as f:
                        existing_config = yaml.safe_load(f)
                        config_cost = existing_config.get('total_cost', 0.0)
                        if config_cost > 0:
                            existing_cost = config_cost
                            should_use_existing = True
                            print(f"   ✅ Found cost in config: ${existing_cost:.6f}")
                except Exception:
                    pass
                    
            if should_use_existing:
                try:
                    sample_output = extract_output_from_json(str(baseline_yaml_path), str(baseline_json_path))[:1]
                except Exception as e:
                    print(f"⚠️  Could not load existing output: {e}")
                    sample_output = []
                    
                return create_original_query_result(
                    success=True,
                    cost=existing_cost,
                    output_file_path=str(baseline_json_path),
                    sample_output=sample_output
                )
        
        print(f"🔄 Executing fresh original query for {experiment_name} (no valid existing cost found)")
        
        # Load original YAML
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Redirect output path to experiment folder in Modal volume
        print("baseline_json_path (Modal):", baseline_json_path)
        try:
            config['pipeline']['output']['path'] = str(baseline_json_path)
        except Exception:
            # Fallback if structure is different
            config.setdefault('pipeline', {}).setdefault('output', {})['path'] = str(baseline_json_path)

        # Set data directory if provided
        if data_dir:
            os.environ['EXPERIMENT_DATA_DIR'] = data_dir

        # Force fresh run
        config['bypass_cache'] = True

        # Save modified YAML for provenance in Modal volume
        baseline_yaml_path = output_path / "baseline_config.yaml"
        with open(baseline_yaml_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False)

        # Run pipeline
        runner = DSLRunner.from_yaml(str(baseline_yaml_path))
        runner.load()
        if runner.last_op_container:
            data, _, _ = runner.last_op_container.next()
            runner.save(data)
        total_cost = runner.total_cost
        runner.reset_env()

        # Load sample output (truncate if huge)
        sample_output = []
        try:
            sample_output = extract_output_from_json(str(baseline_yaml_path), str(baseline_json_path))[:1]
        except Exception as e:
            print(f"⚠️  Could not load baseline output JSON: {e}")

        # Commit changes to Modal volume
        volume.commit()
        
        return create_original_query_result(
            success=True,
            cost=total_cost,
            output_file_path=str(baseline_json_path),
            sample_output=sample_output
        )
    except Exception as e:
        print(f"❌ Original query execution failed: {e}")
        return create_original_query_result(
            success=False,
            cost=0.0,
            output_file_path=None,
            sample_output=[],
            error=str(e)
        )


def run_original_query(yaml_path: str, dataset: str, experiment_name: str, 
                      data_dir: Optional[str] = None, output_dir: Optional[str] = None, 
                      original_cost: Optional[float] = None) -> Dict[str, Any]:
    """Execute the original query plan once using Modal and return results."""
    return run_original_query_remote.remote(yaml_path, dataset, experiment_name, data_dir, output_dir, original_cost)


def try_fusion_directives(yaml_path: str, dataset: str, experiment_name: str, 
                         data_dir: Optional[str] = None, output_dir: Optional[str] = None) -> Optional[str]:
    """Try applying operator fusion directives to the original plan and return the path to the optimized YAML if successful."""
    try:
        # Load original YAML
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        operations = config.get('pipeline', {}).get('operations', [])
        if len(operations) < 2:
            print(f"⚠️  Not enough operations for fusion (found {len(operations)}), skipping fusion attempt")
            return None
        
        # Try operator fusion (works on consecutive map, filter operations)
        fusion_applied = False
        
        # Look for consecutive map-map, map-filter, filter-map, or filter-filter operations
        for i in range(len(operations) - 1):
            op1, op2 = operations[i], operations[i + 1]
            op1_type = op1.get('type')
            op2_type = op2.get('type')
            
            if ((op1_type in ['map', 'filter'] and op2_type in ['map', 'filter']) and 
                (op1_type != 'reduce' and op2_type != 'reduce')):
                
                print(f"🔄 Attempting operator fusion: {op1['name']} ({op1_type}) + {op2['name']} ({op2_type})")
                
                try:
                    fusion_directive = OperatorFusionDirective()
                    new_operations, _ = fusion_directive.instantiate(
                        operators=operations,
                        target_ops=[op1['name'], op2['name']],
                        agent_llm="gpt-4o-mini",  # Use a smaller model for fusion
                        global_default_model=config.get('default_model', 'gpt-4o-mini')
                    )
                    operations = new_operations
                    fusion_applied = True
                    print(f"✅ Operator fusion applied successfully")
                    break  # Apply one fusion at a time
                except Exception as e:
                    print(f"⚠️  Operator fusion failed: {e}")
                    continue
        
        if not fusion_applied:
            print(f"ℹ️  No fusion directives could be applied to the pipeline")
            return None
        
        # Save the fused configuration
        fused_config = deepcopy(config)
        fused_config['pipeline']['operations'] = operations
        
        # Set up output directory
        if output_dir is None:
            output_dir = "./outputs"
        
        output_path = Path(output_dir) / f"{experiment_name}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        fused_yaml_path = output_path / "orig_fused_config.yaml"
        with open(fused_yaml_path, 'w') as f:
            yaml.dump(fused_config, f, sort_keys=False)
        
        print(f"💾 Fused configuration saved to: {fused_yaml_path}")
        return str(fused_yaml_path)
        
    except Exception as e:
        print(f"❌ Error applying fusion directives: {e}")
        return None


@app.function(image=image, secrets=[modal.Secret.from_dotenv()], volumes={VOLUME_MOUNT_PATH: volume}, timeout=60 * 60)
def test_single_model_remote(base_yaml_path: str, model: str, dataset: str, experiment_name: str,
                            data_dir: Optional[str] = None, output_dir: Optional[str] = None,
                            ground_truth: Optional[str] = None) -> Dict[str, Any]:
    """Test a single model performance on the original planin Modal environment."""
    try:
        # Set up Modal environment
        os.environ["EXPERIMENT_OUTPUT_DIR"] = str(Path(VOLUME_MOUNT_PATH) / "outputs")
        
        # Read the base config from Modal volume
        with open(base_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set the default model for the entire pipeline
        config['default_model'] = model
        fix_models(config)
        config['bypass_cache'] = True
        
        # Set up output directory in Modal volume
        if output_dir is None:
            output_dir = str(Path(VOLUME_MOUNT_PATH) / "outputs")
        else:
            if not output_dir.startswith(VOLUME_MOUNT_PATH):
                output_dir = str(Path(VOLUME_MOUNT_PATH) / "outputs")
        
        # Create dataset test directory structure 
        dataset_test_dir = Path(output_dir) / f"{dataset}"
        dataset_test_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the model-specific config in Modal volume with the desired naming
        model_name_clean = model.replace('/', '_').replace('-', '_')
        baseline_yaml_path = dataset_test_dir / f"{model_name_clean}_config.yaml"
        baseline_json_path = dataset_test_dir / f"{model_name_clean}_output.json"
        config['pipeline']['output']['path'] = str(baseline_json_path)

        with open(baseline_yaml_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False)
        
        print(f"📝 Saved model config to Modal volume: {baseline_yaml_path}")
    
        # Run pipeline directly
        try:
            runner = DSLRunner.from_yaml(str(baseline_yaml_path))
            runner.load()
            if runner.last_op_container:
                data, _, _ = runner.last_op_container.next()
                runner.save(data)
            total_cost = runner.total_cost
            runner.reset_env()
            
            # Load sample output
            sample_output = []
            try:
                sample_output = extract_output_from_json(str(baseline_yaml_path), str(baseline_json_path))[:1]
            except Exception as e:
                print(f"⚠️  Could not load output JSON: {e}")
            
            # Commit changes to Modal volume
            volume.commit()
            
            model_result = create_original_query_result(
                success=True,
                cost=total_cost,
                output_file_path=str(baseline_json_path),
                sample_output=sample_output
            )
        except Exception as e:
            print(f"❌ Pipeline execution failed for {model}: {e}")
            model_result = create_original_query_result(
                success=False,
                cost=0.0,
                output_file_path=None,
                sample_output=[],
                error=str(e)
            )
        
        if model_result["success"]:
            cost = model_result["cost"]
            
            # Calculate accuracy if ground truth is provided
            accuracy = None
            try:
                # Use the dataset's evaluation function
                evaluate_func = get_evaluate_func(dataset)
                print(f"Evaluate function: {evaluate_func} ****")
                if evaluate_func:
                    print("YES")
                    accuracy = evaluate_func("test_single_model", model_result["output_file_path"])
                    print("AFTER")
                    print(f"Accuracy: {accuracy}")
                    print(dataset_accuracy_metrics.get(dataset, 'accuracy'))
                    accuracy = accuracy.get(dataset_accuracy_metrics.get(dataset, 'accuracy'), None)
                    print(f"Accuracy: {accuracy}")
            except Exception as eval_e:
                print(f"⚠️  Accuracy evaluation failed for {model}: {eval_e}")
        
            return {
                "model": model,
                "cost": cost,
                "accuracy": accuracy,
                "output_file": model_result["output_file_path"],
                "success": True
            }
        else:
            print("*****HERE2")
            return {
                "model": model,
                "cost": 0.0,
                "accuracy": None,
                "error": model_result.get("error", "Unknown error"),
                "success": False
            }
    except Exception as e:
        print(f"❌ Model {model} failed with exception: {e}")
        return {
            "model": model,
            "cost": 0.0,
            "accuracy": None,
            "error": str(e),
            "success": False
        }


def test_model_performance(yaml_path: str, dataset: str, experiment_name: str, 
                          data_dir: Optional[str] = None, output_dir: Optional[str] = None,
                          ground_truth: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Test all available models and return their cost and accuracy performance."""
    results = {}
    
    # Test each model using Modal - run sequentially to avoid issues
    for model in AVAILABLE_MODELS:
        print(f"🧪 Testing model: {model}")
        try:
            result = test_single_model_remote.remote(
                base_yaml_path=yaml_path,
                model=model,
                dataset=dataset,
                experiment_name=experiment_name,
                data_dir=data_dir,
                output_dir=output_dir,
                ground_truth=ground_truth
            )
            results[model] = result
            if result["success"]:
                print(f"✅ Model {model}: cost=${result['cost']:.6f}, accuracy={result['accuracy']}")
            else:
                print(f"❌ Model {model} failed: {result['error']}")
        except Exception as e:
            results[model] = {
                "cost": 0.0,
                "accuracy": None,
                "error": str(e),
                "success": False
            }
            print(f"❌ Model {model} failed with exception: {e}")
    
    return results


@app.function(image=image, secrets=[modal.Secret.from_dotenv()], volumes={VOLUME_MOUNT_PATH: volume}, timeout=60 * 10)
def save_optimized_config_remote(original_yaml_path: str, best_model: str, dataset: str, 
                                experiment_name: str, output_dir: Optional[str] = None) -> str:
    """Save the optimized configuration with best model in Modal volume."""
    
    # Set up Modal environment
    os.environ["EXPERIMENT_OUTPUT_DIR"] = str(Path(VOLUME_MOUNT_PATH) / "outputs")
    output_dir = str(Path(VOLUME_MOUNT_PATH) / "outputs")
    
    # Read the base configuration
    with open(original_yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply the best model
    config['default_model'] = best_model
    for op in config.get('pipeline', {}).get('operations', []):
        if 'model' not in op:
            op['model'] = best_model
    fix_models(config)
    
    # Save to Modal volume with the desired path structure
    dataset_test_dir = Path(output_dir) / f"{dataset}_test"
    dataset_test_dir.mkdir(parents=True, exist_ok=True)
    
    optimized_yaml_path = dataset_test_dir / "optimized_config.yaml"
    with open(optimized_yaml_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    # Commit changes to Modal volume
    volume.commit()
    
    print(f"💾 Optimized configuration saved to Modal volume: {optimized_yaml_path}")
    return str(optimized_yaml_path)


def select_best_model_config(model_results: Dict[str, Dict[str, Any]], 
                           original_yaml_path: str, 
                           dataset: str,
                           experiment_name: str,
                           output_dir: Optional[str] = None) -> Tuple[str, str, float, Optional[float]]:
    """Select the best performing model configuration and return the optimized YAML path."""
    
    # Filter successful models
    successful_models = {model: result for model, result in model_results.items() if result["success"]}
    
    if not successful_models:
        print("⚠️  No models succeeded, using original configuration")
        return original_yaml_path, "original", 0.0, None
    
    # Find the model with best accuracy (if accuracy is available)
    models_with_accuracy = {model: result for model, result in successful_models.items() 
                          if result["accuracy"] is not None}
    
    if models_with_accuracy:
        # Select model with highest accuracy
        best_model = max(models_with_accuracy.keys(), 
                        key=lambda m: models_with_accuracy[m]["accuracy"])
        best_result = models_with_accuracy[best_model]
        print(f"🏆 Best model by accuracy: {best_model} (accuracy={best_result['accuracy']:.4f}, cost=${best_result['cost']:.6f})")
    else:
        # Fallback to lowest cost model
        best_model = min(successful_models.keys(), 
                        key=lambda m: successful_models[m]["cost"])
        best_result = successful_models[best_model]
        print(f"🏆 Best model by cost: {best_model} (cost=${best_result['cost']:.6f})")
    
    # Save the optimized configuration in Modal volume
    optimized_yaml_path = save_optimized_config_remote.remote(
        original_yaml_path=original_yaml_path,
        best_model=best_model,
        dataset=dataset,
        experiment_name=experiment_name,
        output_dir=output_dir
    )
    
    return optimized_yaml_path, best_model, best_result["cost"], best_result["accuracy"]


@app.function(image=image, secrets=[modal.Secret.from_dotenv()], volumes={VOLUME_MOUNT_PATH: volume}, timeout=60 * 30)
def generate_plots_for_experiments_remote(dataset: str, experiments: List[str], output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Generate Pareto frontier comparison plots for specified experiments in Modal.
    
    Args:
        dataset: Dataset name (e.g., 'cuad', 'biodex')
        experiments: List of experiment names (e.g., ['baseline', 'mcts', 'simple_baseline'])
        output_dir: Optional output directory override
    """
    
    matplotlib.use('Agg')  # Use non-interactive backend for Modal
    
    try:
        print(f"\n📊 Generating comparison plots for {dataset} with experiments: {experiments}")
        
        # Set up Modal environment
        os.environ["EXPERIMENT_OUTPUT_DIR"] = str(Path(VOLUME_MOUNT_PATH) / "outputs")
        
        # Set up paths to evaluation files in Modal volume
        base_output_dir = str(Path(VOLUME_MOUNT_PATH) / "outputs")
        
        # Build evaluation file paths based on provided experiments
        evaluation_files = {}
        pareto_points = {}
        
        for exp in experiments:
            eval_file = f"{base_output_dir}/{dataset}_{exp}/evaluation_metrics.json"
            evaluation_files[exp] = eval_file
            
            if Path(eval_file).exists():
                pareto_points[exp] = find_pareto_frontier(eval_file, dataset)
                print(f"✅ Found {len(pareto_points[exp])} {exp} Pareto points")
            else:
                print(f"⚠️ File not found: {eval_file}")
                pareto_points[exp] = []
        
        # Check if dataset is supported
        if dataset not in dataset_metrics:
            return {
                "success": False,
                "error": f"Dataset '{dataset}' not supported for plotting. Supported datasets: {list(dataset_metrics.keys())}"
            }
            
        # Check if we have enough data
        valid_experiments = [exp for exp in experiments if len(pareto_points[exp]) > 0]
        if len(valid_experiments) < 2:
            return {
                "success": False,
                "error": f"Need at least 2 experiments with data to generate comparison plots. Found {len(valid_experiments)} valid experiments."
            }
            
        print(f"📐 Calculating hypervolumes for experiments: {valid_experiments}")
        
        # Calculate hypervolumes - pass files in order expected by the function
        baseline_file = evaluation_files.get('baseline', evaluation_files[valid_experiments[0]])
        mcts_file = evaluation_files.get('mcts', evaluation_files[valid_experiments[0]])  
        simple_file = evaluation_files.get('simple_baseline', evaluation_files[valid_experiments[0]])
        
        baseline_points = pareto_points.get('baseline', [])
        mcts_points = pareto_points.get('mcts', [])
        simple_points = pareto_points.get('simple_baseline', [])
        
        baseline_hv, mcts_hv, simple_hv, _, reference_point = calculate_hypervolume_comparison(
            baseline_file, mcts_file, simple_file, None, dataset,
            baseline_points, mcts_points, simple_points, None
        )
        
        print(f"📊 Hypervolume Results:")
        for exp in valid_experiments:
            if exp == 'baseline':
                print(f"   Baseline: {baseline_hv:.4f}")
            elif exp == 'mcts':
                print(f"   MCTS: {mcts_hv:.4f}")
            elif exp == 'simple_baseline':
                print(f"   Simple Baseline: {simple_hv:.4f}")
        
        # Save plots to the original output directory
        plot_output_dir = Path(base_output_dir) / f"{dataset}_original"
        plot_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🎨 Generating comparison plot...")
        plot_pareto_frontier_comparison(
            baseline_file, mcts_file, simple_file, None, dataset,
            baseline_points, mcts_points, simple_points, None,
            output_path=str(plot_output_dir), reference_point=reference_point
        )
        
        plot_path = str(plot_output_dir / f"pareto_frontier_comparison_{dataset}.png")
        print(f"📈 Comparison plot saved to: {plot_path}")
        
        # Save hypervolume summary
        hypervolume_summary_path = plot_output_dir / f"hypervolume_summary_{dataset}.txt"
        with open(hypervolume_summary_path, 'w') as f:
            f.write(f"Hypervolume Comparison Results - {dataset.upper()} Dataset\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Dataset: {dataset}\n")
            f.write(f"Experiments: {', '.join(valid_experiments)}\n")
            f.write(f"Accuracy Metric: {dataset_metrics[dataset]}\n")
            f.write(f"Reference Point: accuracy={reference_point['accuracy']:.4f}, cost=${reference_point['cost']:.6f}\n\n")
            
            f.write("HYPERVOLUME RESULTS:\n")
            f.write("-" * 20 + "\n")
            if 'baseline' in valid_experiments:
                f.write(f"Baseline Hypervolume:       {baseline_hv:.6f}\n")
            if 'mcts' in valid_experiments:
                f.write(f"MCTS Hypervolume:           {mcts_hv:.6f}\n")
            if 'simple_baseline' in valid_experiments:
                f.write(f"Simple Baseline Hypervolume: {simple_hv:.6f}\n")
            f.write("\n")
            
            f.write("PARETO FRONTIER POINTS:\n")
            f.write("-" * 25 + "\n")
            
            for exp in valid_experiments:
                points = pareto_points[exp]
                f.write(f"\n{exp.title()} ({len(points)} points):\n")
                for i, (iteration, accuracy, cost) in enumerate(points):
                    f.write(f"  {i+1:2d}. Iteration {iteration}: {dataset_metrics[dataset]}={accuracy:.4f}, cost=${cost:.6f}\n")
            
            f.write(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        
        print(f"📄 Hypervolume summary saved to: {hypervolume_summary_path}")
        
        # Close any open plots to free memory
        plt.close('all')
        
        # Commit changes to Modal volume
        volume.commit()
        
        return {
            "success": True,
            "plot_path": plot_path,
            "summary_path": str(hypervolume_summary_path),
            "hypervolumes": {
                exp: (baseline_hv if exp == 'baseline' else 
                     mcts_hv if exp == 'mcts' else 
                     simple_hv if exp == 'simple_baseline' else 0.0)
                for exp in valid_experiments
            },
            "experiments_processed": valid_experiments
        }
        
    except Exception as e:
        print(f"❌ Error generating comparison plots: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


def run_from_config(config: Dict[str, Any]) -> int:
    experiments: List[Dict[str, Any]] = config.get("experiments", [])
    if not experiments:
        print("No experiments found in config.")
        return 1

    results: List[Tuple[str, str, modal.functions._FunctionCall]] = []

    for exp in experiments:
        dataset: str = exp["dataset"].lower()
        yaml_path: str = _get_with_default(
            DEFAULT_YAML_PATHS, dataset, exp.get("yaml_path")
        )

        # Per-experiment optional overrides
        data_dir: Optional[str] = exp.get("data_dir")
        output_dir: Optional[str] = exp.get("output_dir")
        ground_truth: Optional[str] = exp.get("ground_truth")
        original_cost: Optional[float] = exp.get("original_cost", 0.0)
        original_config_path: Optional[str] = exp.get("original_config_path", None)
        
        build_first_layer: Optional[bool] = exp.get("build_first_layer", False)
        
        print(f"{'='*60}")
        print(f"PROCESSING DATASET: {dataset.upper()}")
        print(f"{'='*60}")
        
        original_path = str(Path(VOLUME_MOUNT_PATH) / "outputs") + f"/{original_config_path}"
        
        # Check locally first, then check Modal volume
        modal_result = check_optimized_config_exists.remote(str(original_path))
        print(f"Modal result: {modal_result}")
        if build_first_layer:
            print(f"🚀 Building first layer of the search tree manually...")
            original_result = None
            original_yaml_path = yaml_path
            
        elif modal_result["exists"]:
            print(f"✅ Found existing original configuration {original_path}")
            print(f"🚀 Jumping directly to Step 4...")
            
            original_config = modal_result["config"]
            original_yaml_path = modal_result["path"]
            
            # Create original_result using the provided optimized cost
            original_result = create_original_query_result(
                success=True,
                cost=original_cost,
                output_file_path=original_config['pipeline']['output']['path'],
                sample_output=[]
            )
            
            print(f"✅ Using existing optimized configuration:")
            print(f"   Cost: ${original_cost:.6f}")
            print(f"   Config: {original_yaml_path}")
            
        else:
            print("HERE")
            original_yaml_path = yaml_path
            print(f"📋 No existing optimized configuration found, proceeding with optimization...")
            
            # Step 1: Try to apply fusion directives to the original plan
            print(f"\n🔄 Step 1: Applying fusion directives to original plan...")
            fused_yaml_path = try_fusion_directives(
                yaml_path=yaml_path,
                dataset=dataset,
                experiment_name=f"{dataset}_test",
                data_dir=data_dir,
                output_dir=output_dir
            )
            
            # Use fused config if available, otherwise use original
            working_yaml_path = fused_yaml_path if fused_yaml_path else yaml_path
            config_type = "fused" if fused_yaml_path else "original"
            print(f"ℹ️  Using {config_type} configuration: {working_yaml_path}")
            
            # Step 2: Test all models on the working configuration
            print(f"\n🧪 Step 2: Testing all available models...")
            model_results = test_model_performance(
                yaml_path=working_yaml_path,
                dataset=dataset,
                experiment_name=f"{dataset}_test",
                data_dir=data_dir,
                output_dir=output_dir,
                ground_truth=ground_truth
            )
            
            # Step 3: Find Pareto frontier models and save results
            print(f"\n🏆 Step 3: Finding Pareto frontier models...")
            
            # Extract successful models with accuracy and cost data
            successful_models = {model: result for model, result in model_results.items() 
                               if result["success"] and result["accuracy"] is not None}
            
            if not successful_models:
                print("⚠️  No models succeeded with valid accuracy data")
                raise Exception("No models succeeded with valid accuracy data")
            else:
                # Find Pareto frontier models on accuracy-cost trade-off
                model_points = [(model, result["accuracy"], result["cost"]) 
                               for model, result in successful_models.items()]
                
                # Sort by cost (ascending) and accuracy (descending)
                sorted_models = sorted(model_points, key=lambda x: (x[2], -x[1]))
                
                pareto_models = []
                best_accuracy = float('-inf')
                
                for model, accuracy, cost in sorted_models:
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        pareto_models.append((model, accuracy, cost))
                
                print(f"📊 Found {len(pareto_models)} Pareto frontier models:")
                for i, (model, accuracy, cost) in enumerate(pareto_models):
                    print(f"   {i+1}. {model}: accuracy={accuracy:.4f}, cost=${cost:.6f}")
                
                # Prepare data to save
                MODEL_COSTS = {model: result["cost"] for model, result in model_results.items() 
                              if result["success"]}
                MODEL_STATS = {model: {"accuracy": result["accuracy"], "cost": result["cost"], 
                                     "success": result["success"]} 
                              for model, result in model_results.items()}
                first_layer_yaml_paths = {model: result.get("output_file") 
                                        for model, result in model_results.items() if result["success"]}
                FRONTIER_MODELS = [{"model": model, "accuracy": accuracy, "cost": cost} 
                                  for model, accuracy, cost in pareto_models]
                
                # Save the results to Modal volume
                os.environ["EXPERIMENT_OUTPUT_DIR"] = str(Path(VOLUME_MOUNT_PATH) / "outputs")
                output_dir_path = Path(VOLUME_MOUNT_PATH) / "outputs" / f"{dataset}"
                output_dir_path.mkdir(parents=True, exist_ok=True)
                
                # Save all the required data
                pareto_results = {
                    "MODEL_COSTS": MODEL_COSTS,
                    "MODEL_STATS": MODEL_STATS, 
                    "first_layer_yaml_paths": first_layer_yaml_paths,
                    "FRONTIER_MODELS": FRONTIER_MODELS
                }
                
                pareto_results_path = output_dir_path / "pareto_frontier_results.json"
                with open(pareto_results_path, 'w') as f:
                    json.dump(pareto_results, f, indent=2)
                
                print(f"💾 Pareto frontier results saved to: {pareto_results_path}")
                
                raise Exception("Stop here")

        # Now proceed with the regular search methods using the optimized plan
        print(f"\n🚀 Step 4: Starting search methods with optimized configuration...")
        
        # Baseline block
        baseline_cfg: Optional[Dict[str, Any]] = exp.get("baseline")
        if baseline_cfg:
            bl_name: str = f"{dataset}_baseline"
            bl_iters: int = int(baseline_cfg.get("iterations", 1))
            bl_model: Optional[str] = baseline_cfg.get("model")

            call = _spawn_baseline(
                dataset=dataset,
                yaml_path=original_yaml_path,
                experiment_name=bl_name,
                iterations=bl_iters,
                model=bl_model,
                data_dir=data_dir,
                output_dir=output_dir,
                ground_truth=ground_truth,
                original_query_result=original_result,
            )
            results.append((dataset, f"baseline:{bl_name}", call))
            print(f"Spawned baseline for {dataset} as {bl_name}")

        # MCTS block
        mcts_cfg: Optional[Dict[str, Any]] = exp.get("mcts")
        if mcts_cfg:
            mc_name: str = f"{dataset}_mcts"
            mc_max: int = int(mcts_cfg.get("max_iterations", 10))
            mc_c: Optional[float] = mcts_cfg.get("exploration_weight")
            mc_model: Optional[str] = mcts_cfg.get("model")
            ds_path = _get_with_default(DEFAULT_DATASET_PATHS, dataset, exp.get("dataset_path"))
            
            call = _spawn_mcts(
                dataset=dataset,
                yaml_path=original_yaml_path,  
                dataset_path=ds_path,
                experiment_name=mc_name,
                max_iterations=mc_max,
                exploration_weight=mc_c,
                model=mc_model,
                data_dir=data_dir,
                output_dir=output_dir,
                ground_truth=ground_truth,
                original_query_result=original_result,
                build_first_layer=build_first_layer
            )
            results.append((dataset, f"mcts:{mc_name}", call))
            print(f"Spawned MCTS for {dataset} as {mc_name}")

        # Simple baseline block
        simple_baseline_cfg: Optional[Dict[str, Any]] = exp.get("simple_baseline")
        if simple_baseline_cfg:
            sb_name: str = f"{dataset}_simple_baseline"
            sb_model: Optional[str] = simple_baseline_cfg.get("model", "o3")

            call = _spawn_simple_baseline(
                dataset=dataset,
                experiment_name=sb_name,
                model=sb_model,
                output_dir=output_dir,
                ground_truth=ground_truth,
                original_query_result=original_result,
            )
            results.append((dataset, f"simple_baseline:{sb_name}", call))
            print(f"Spawned simple baseline for {dataset} as {sb_name}")

    # Wait for all
    print(f"Waiting for {len(results)} jobs to complete...")
    failures = 0
    for dataset, label, call in results:
        try:
            res = call.get()
            print(f"✓ {label} ({dataset}) finished: {str(res)[:120]}...")
        except Exception as e:
            failures += 1
            print(f"✗ {label} ({dataset}) failed: {e}")

    if failures:
        print(f"Completed with {failures} failure(s)")
        # Still try to generate plots for successful experiments
        
    print("All experiments completed successfully")
    
    # Generate comparison plots for each dataset
    print(f"\n{'='*60}")
    print("GENERATING COMPARISON PLOTS")
    print(f"{'='*60}")
    
    datasets_processed = set()
    for exp in experiments:
        dataset = exp["dataset"].lower()
        if dataset not in datasets_processed:
            datasets_processed.add(dataset)
            output_dir = exp.get("output_dir")
            result = generate_plots_for_experiments_remote.remote(dataset, ['baseline', 'mcts', 'simple_baseline'], output_dir)
            if result["success"]:
                print(f"✅ Comparison plots generated for {dataset}!")
                print(f"📈 Plot: {result['plot_path']}")
                print(f"📄 Summary: {result['summary_path']}")
            else:
                print(f"❌ Plot generation failed for {dataset}: {result['error']}")
    
    return 2 if failures else 0

@app.function(image=image, secrets=[modal.Secret.from_dotenv()], volumes={VOLUME_MOUNT_PATH: volume}, timeout=60 * 30)
def run_plots_from_config() -> int:
    """Generate comparison plots for all datasets in config using existing Modal volume results."""
    
    config = CONFIG
    experiments: List[Dict[str, Any]] = config.get("experiments", [])
    if not experiments:
        print("No experiments found in config.")
        return 1

    print(f"\n{'='*60}")
    print("GENERATING COMPARISON PLOTS FROM EXISTING RESULTS")
    print(f"{'='*60}")
    
    datasets_processed = set()
    failures = 0
    
    for exp in experiments:
        dataset = exp["dataset"].lower()
        if dataset not in datasets_processed:
            datasets_processed.add(dataset)
            output_dir = exp.get("output_dir")
            
            # Determine which experiments to plot based on config
            experiments_to_plot = []
            if exp.get("baseline"):
                experiments_to_plot.append("baseline")
            if exp.get("mcts"):
                experiments_to_plot.append("mcts")
            if exp.get("simple_baseline"):
                experiments_to_plot.append("simple_baseline")
            
            if len(experiments_to_plot) < 2:
                print(f"⚠️ Skipping {dataset}: need at least 2 experiments to plot")
                continue
                
            print(f"\n📊 Generating plots for {dataset} with experiments: {experiments_to_plot}")
            
            try:
                result = generate_plots_for_experiments_remote.remote(dataset, experiments_to_plot, output_dir)
                if result["success"]:
                    print(f"✅ Comparison plots generated for {dataset}!")
                    print(f"📈 Plot: {result['plot_path']}")
                    print(f"📄 Summary: {result['summary_path']}")
                    print(f"📊 Processed experiments: {', '.join(result['experiments_processed'])}")
                else:
                    print(f"❌ Plot generation failed for {dataset}: {result['error']}")
                    failures += 1
            except Exception as e:
                print(f"❌ Error generating plots for {dataset}: {e}")
                failures += 1
    
    print(f"\n{'='*60}")
    if failures == 0:
        print(f"✅ Successfully generated plots for {len(datasets_processed)} datasets")
    else:
        print(f"⚠️ Completed with {failures} failure(s) out of {len(datasets_processed)} datasets")
    print(f"{'='*60}")
    
    return 1 if failures > 0 else 0


@app.local_entrypoint()
def main() -> None:
    run_from_config(CONFIG)
