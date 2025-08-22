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
        
        # Check if original query results already exist
        baseline_json_path = output_path / "original_output.json"
        baseline_yaml_path = output_path / "baseline_config.yaml"
        
        if baseline_json_path.exists() and baseline_yaml_path.exists():
            print(f"✅ Found existing original query results for {experiment_name}")
            print(f"   Original output: {baseline_json_path}")
            print(f"   Baseline config: {baseline_yaml_path}")
            
            # Load existing results
            try:
                sample_output = extract_output_from_json(str(baseline_yaml_path), str(baseline_json_path))[:1]
            except Exception as e:
                print(f"⚠️  Could not load existing baseline output JSON: {e}")
                sample_output = []
            
            # Use provided original_cost if available, otherwise try to extract from config
            if original_cost is not None:
                total_cost = original_cost
                print(f"   ✅ Using provided original cost: ${total_cost:.6f}")
            else:
                total_cost = 0.0
                try:
                    with open(baseline_yaml_path, 'r') as f:
                        existing_config = yaml.safe_load(f)
                        # Cost might be stored in the config or we can set it to a default
                        total_cost = existing_config.get('total_cost', 0.0)
                    if total_cost > 0:
                        print(f"   ✅ Found cost in config: ${total_cost:.6f}")
                    else:
                        print(f"   ⚠️  No cost found in config, using 0.0")
                except Exception:
                    print(f"   ⚠️  Could not read config, using cost 0.0")
                    total_cost = 0.0
            
            return create_original_query_result(
                success=True,
                cost=total_cost,
                output_file_path=str(baseline_json_path),
                sample_output=sample_output
            )
        
        print(f"🔄 No existing results found, executing original query for {experiment_name}")
        
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
def generate_comparison_plots_remote(dataset: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Generate Pareto frontier comparison plots for all three methods in Modal."""
    
    matplotlib.use('Agg')  # Use non-interactive backend for Modal
    
    try:
        print(f"\n📊 Generating comparison plots for {dataset} in Modal...")
        
        # Set up Modal environment
        os.environ["EXPERIMENT_OUTPUT_DIR"] = str(Path(VOLUME_MOUNT_PATH) / "outputs")
        
        # Set up paths to evaluation files in Modal volume
        base_output_dir = str(Path(VOLUME_MOUNT_PATH) / "outputs")
        
        evaluation_file_baseline = f"{base_output_dir}/{dataset}_baseline/evaluation_metrics.json"
        evaluation_file_mcts = f"{base_output_dir}/{dataset}_mcts/evaluation_metrics.json"  
        evaluation_file_simple = f"{base_output_dir}/{dataset}_simple_baseline/evaluation_metrics.json"
        
        # Check if dataset is supported
        if dataset not in dataset_metrics:
            return {
                "success": False,
                "error": f"Dataset '{dataset}' not supported for plotting. Supported datasets: {list(dataset_metrics.keys())}"
            }
            
        print(f"Looking for evaluation files:")
        print(f"  Baseline: {evaluation_file_baseline}")
        print(f"  MCTS: {evaluation_file_mcts}")
        print(f"  Simple Baseline: {evaluation_file_simple}")
        
        # Check if files exist
        files_exist = []
        for file_path in [evaluation_file_baseline, evaluation_file_mcts, evaluation_file_simple]:
            if Path(file_path).exists():
                files_exist.append(file_path)
            else:
                print(f"⚠️ File not found: {file_path}")
        
        if len(files_exist) < 2:
            return {
                "success": False,
                "error": f"Need at least 2 evaluation files to generate comparison plots. Found {len(files_exist)} files."
            }
            
        # Find Pareto frontiers for available methods
        pareto_points_baseline = []
        pareto_points_mcts = []
        pareto_points_simple = []
        
        if Path(evaluation_file_baseline).exists():
            pareto_points_baseline = find_pareto_frontier(evaluation_file_baseline, dataset)
            print(f"✅ Found {len(pareto_points_baseline)} baseline Pareto points")
            
        if Path(evaluation_file_mcts).exists():
            pareto_points_mcts = find_pareto_frontier(evaluation_file_mcts, dataset)
            print(f"✅ Found {len(pareto_points_mcts)} MCTS Pareto points")
            
        if Path(evaluation_file_simple).exists():
            pareto_points_simple = find_pareto_frontier(evaluation_file_simple, dataset)
            print(f"✅ Found {len(pareto_points_simple)} simple baseline Pareto points")
        
        # Calculate hypervolumes if we have enough data
        plot_path = None
        if len(files_exist) >= 2:
            print(f"\n📐 Calculating hypervolumes...")
            
            # Use available files for hypervolume calculation
            baseline_file = evaluation_file_baseline if Path(evaluation_file_baseline).exists() else files_exist[0]
            mcts_file = evaluation_file_mcts if Path(evaluation_file_mcts).exists() else files_exist[0] 
            simple_file = evaluation_file_simple if Path(evaluation_file_simple).exists() else files_exist[0]
            
            baseline_hv, mcts_hv, simple_hv, reference_point = calculate_hypervolume_comparison(
                baseline_file, mcts_file, simple_file, dataset,
                pareto_points_baseline, pareto_points_mcts, pareto_points_simple
            )
            
            print(f"📊 Hypervolume Results:")
            print(f"   Baseline: {baseline_hv:.4f}")
            print(f"   MCTS: {mcts_hv:.4f}")
            print(f"   Simple Baseline: {simple_hv:.4f}")
            
            # Save plots to the original output directory (same location as original query results)
            plot_output_dir = Path(base_output_dir) / f"{dataset}_original"
            plot_output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n🎨 Generating comparison plot...")
            plot_pareto_frontier_comparison(
                baseline_file, mcts_file, simple_file, dataset,
                pareto_points_baseline, pareto_points_mcts, pareto_points_simple,
                output_path=str(plot_output_dir), reference_point=reference_point
            )
            
            plot_path = str(plot_output_dir / f"pareto_frontier_comparison_{dataset}.png")
            print(f"📈 Comparison plot saved to: {plot_path}")
            
            # Save hypervolume results to text file
            hypervolume_summary_path = plot_output_dir / f"hypervolume_summary_{dataset}.txt"
            with open(hypervolume_summary_path, 'w') as f:
                f.write(f"Hypervolume Comparison Results - {dataset.upper()} Dataset\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Dataset: {dataset}\n")
                f.write(f"Accuracy Metric: {dataset_metrics[dataset]}\n")
                f.write(f"Reference Point: accuracy={reference_point['accuracy']:.4f}, cost=${reference_point['cost']:.6f}\n\n")
                
                f.write("HYPERVOLUME RESULTS:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Baseline Hypervolume:       {baseline_hv:.6f}\n")
                f.write(f"MCTS Hypervolume:           {mcts_hv:.6f}\n")
                f.write(f"Simple Baseline Hypervolume: {simple_hv:.6f}\n\n")
                
                f.write("PARETO FRONTIER POINTS:\n")
                f.write("-" * 25 + "\n")
                
                f.write(f"\nBaseline ({len(pareto_points_baseline)} points):\n")
                for i, (iteration, accuracy, cost) in enumerate(pareto_points_baseline):
                    f.write(f"  {i+1:2d}. Iteration {iteration}: {dataset_metrics[dataset]}={accuracy:.4f}, cost=${cost:.6f}\n")
                
                f.write(f"\nMCTS ({len(pareto_points_mcts)} points):\n")
                for i, (iteration, accuracy, cost) in enumerate(pareto_points_mcts):
                    f.write(f"  {i+1:2d}. Iteration {iteration}: {dataset_metrics[dataset]}={accuracy:.4f}, cost=${cost:.6f}\n")
                
                f.write(f"\nSimple Baseline ({len(pareto_points_simple)} points):\n")
                for i, (iteration, accuracy, cost) in enumerate(pareto_points_simple):
                    f.write(f"  {i+1:2d}. Iteration {iteration}: {dataset_metrics[dataset]}={accuracy:.4f}, cost=${cost:.6f}\n")
                
                f.write("\nFILES PROCESSED:\n")
                f.write("-" * 16 + "\n")
                if Path(evaluation_file_baseline).exists():
                    f.write(f"✓ Baseline: {evaluation_file_baseline}\n")
                else:
                    f.write(f"✗ Baseline: {evaluation_file_baseline} (not found)\n")
                    
                if Path(evaluation_file_mcts).exists():
                    f.write(f"✓ MCTS: {evaluation_file_mcts}\n")
                else:
                    f.write(f"✗ MCTS: {evaluation_file_mcts} (not found)\n")
                    
                if Path(evaluation_file_simple).exists():
                    f.write(f"✓ Simple Baseline: {evaluation_file_simple}\n")
                else:
                    f.write(f"✗ Simple Baseline: {evaluation_file_simple} (not found)\n")
                
                # Add timestamp
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
                    "baseline": baseline_hv,
                    "mcts": mcts_hv,
                    "simple_baseline": simple_hv
                },
                "files_processed": len(files_exist)
            }
        else:
            return {
                "success": False,
                "error": "Not enough valid evaluation files to generate plots"
            }
            
    except Exception as e:
        print(f"❌ Error generating comparison plots: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


def generate_comparison_plots(dataset: str, output_dir: Optional[str] = None) -> None:
    """Generate Pareto frontier comparison plots for all three methods."""
    try:
        print(f"\n📊 Generating comparison plots for {dataset}...")
        result = generate_comparison_plots_remote.remote(dataset, output_dir)
        
        if result["success"]:
            print(f"✅ Comparison plots generated successfully!")
            print(f"📈 Plot saved to: {result['plot_path']}")
            print(f"📄 Summary saved to: {result['summary_path']}")
            print(f"📊 Hypervolume Results:")
            hv = result["hypervolumes"]
            print(f"   Baseline: {hv['baseline']:.4f}")
            print(f"   MCTS: {hv['mcts']:.4f}")
            print(f"   Simple Baseline: {hv['simple_baseline']:.4f}")
        else:
            print(f"❌ Plot generation failed: {result['error']}")
            
    except Exception as e:
        print(f"❌ Error generating comparison plots: {e}")
        traceback.print_exc()


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
            generate_comparison_plots(dataset, output_dir)
    
    return 2 if failures else 0


@app.local_entrypoint()
def main() -> None:
    run_from_config(CONFIG)