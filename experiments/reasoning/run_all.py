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

# Import the existing Modal functions and shared volume/mount from the experiment runners
from experiments.reasoning import run_mcts as mcts_mod
from experiments.reasoning import run_baseline as baseline_mod
from experiments.reasoning import run_simple_baseline as simple_baseline_mod
from experiments.reasoning.evaluation.utils import run_dataset_evaluation, get_evaluate_func, dataset_accuracy_metrics
from experiments.reasoning.utils import app, create_original_query_result, volume, VOLUME_MOUNT_PATH, image  # use the same App as the runners
from experiments.reasoning.plot_result import (
    find_pareto_frontier, calculate_hypervolume_comparison, 
    plot_pareto_frontier_comparison, dataset_metrics
)

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
        # {
        #     "dataset": "game_reviews",
        #     "original_cost": 0.60220281,  # Cost of the original query execution
        #     "mcts": {"max_iterations": 30}
        # }
        {
            "dataset": "cuad",
            "original_cost": 0.13,
            "baseline": {"iterations": 10},
            "simple_baseline": {"iterations": 10},
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
        original_cost: Optional[float] = exp.get("original_cost")
        
        # Execute original query plan once
        print(f"🔄 Executing original query plan for {dataset}...")
        print("output_dir", output_dir)
        original_result = run_original_query(
            yaml_path=yaml_path,
            dataset=dataset,
            experiment_name=f"{dataset}_original",
            data_dir=data_dir,
            output_dir=output_dir,
            original_cost=original_cost
        )
        
        if original_result["success"]:
            print(f"✅ Original query executed successfully, cost: ${original_result['cost']:.4f}")
        else:
            print(f"❌ Original query failed: {original_result['error']}")

        # Baseline block
        baseline_cfg: Optional[Dict[str, Any]] = exp.get("baseline")
        if baseline_cfg:
            bl_name: str = f"{dataset}_baseline"
            bl_iters: int = int(baseline_cfg.get("iterations", 1))
            bl_model: Optional[str] = baseline_cfg.get("model")

            call = _spawn_baseline(
                dataset=dataset,
                yaml_path=yaml_path,
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
            mc_max: int = int(mcts_cfg.get("max_iterations", 100))
            mc_c: Optional[float] = mcts_cfg.get("exploration_weight")
            mc_model: Optional[str] = mcts_cfg.get("model")
            ds_path = _get_with_default(DEFAULT_DATASET_PATHS, dataset, exp.get("dataset_path"))

            call = _spawn_mcts(
                dataset=dataset,
                yaml_path=yaml_path,
                dataset_path=ds_path,
                experiment_name=mc_name,
                max_iterations=mc_max,
                exploration_weight=mc_c,
                model=mc_model,
                data_dir=data_dir,
                output_dir=output_dir,
                ground_truth=ground_truth,
                original_query_result=original_result,
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
