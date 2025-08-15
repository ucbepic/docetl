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
import yaml
from docetl.runner import DSLRunner
from docetl.utils import extract_output_from_json


# Import the existing Modal functions and shared volume/mount from the experiment runners
from experiments.reasoning import run_mcts as mcts_mod
from experiments.reasoning import run_baseline as baseline_mod
from experiments.reasoning import run_simple_baseline as simple_baseline_mod
from experiments.reasoning.evaluation.utils import run_dataset_evaluation, get_evaluate_func, dataset_accuracy_metrics
from experiments.reasoning.utils import app, create_original_query_result, volume, VOLUME_MOUNT_PATH, image  # use the same App as the runners

# Known defaults for dataset YAMLs and sample dataset inputs
DEFAULT_YAML_PATHS: Dict[str, str] = {
    "cuad": "experiments/reasoning/pipelines/cuad.yaml",
    "blackvault": "experiments/reasoning/pipelines/blackvault.yaml",
    "game_reviews": "experiments/reasoning/pipelines/game_reviews.yaml",
    "sustainability": "experiments/reasoning/pipelines/sustainability.yaml",
    "biodex": "experiments/reasoning/pipelines/biodex.yaml",
    "medec": "experiments/reasoning/pipelines/medec.yaml",
}

# Always prefer train-split defaults for MCTS sample inputs
DEFAULT_DATASET_PATHS: Dict[str, str] = {
    "cuad": "experiments/reasoning/data/train/cuad.json",
    "blackvault": "experiments/reasoning/data/train/blackvault.json",
    "game_reviews": "experiments/reasoning/data/train/reviews.json",
    "sustainability": "experiments/reasoning/data/train/sustainability.json",
    "biodex": "experiments/reasoning/data/train/biodex.json",
    "medec": "experiments/reasoning/data/train/medec.json",
}

# Users can edit this CONFIG dict directly before running via Modal
CONFIG: Dict[str, Any] = {
    "experiments": [
        {
            "dataset": "sustainability",
            "baseline": {"iterations": 2},
            "mcts": {"max_iterations": 3},
            "simple_baseline": {"model": "o3"}
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
                             data_dir: Optional[str] = None, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Execute the original query plan once in Modal and return results."""
    import os
    import yaml
    from pathlib import Path
    from docetl.runner import DSLRunner
    from docetl.utils import extract_output_from_json
    from experiments.reasoning.utils import create_original_query_result
    
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
        
        # Load original YAML
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Redirect output path to experiment folder in Modal volume
        baseline_json_path = output_path / "original_output.json"
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
                      data_dir: Optional[str] = None, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Execute the original query plan once using Modal and return results."""
    return run_original_query_remote.remote(yaml_path, dataset, experiment_name, data_dir, output_dir)


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
        
        # Execute original query plan once
        print(f"🔄 Executing original query plan for {dataset}...")
        print("output_dir", output_dir)
        original_result = run_original_query(
            yaml_path=yaml_path,
            dataset=dataset,
            experiment_name=f"{dataset}_original",
            data_dir=data_dir,
            output_dir=output_dir
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
        return 2

    print("All experiments completed successfully")
    return 0


@app.local_entrypoint()
def main() -> None:
    run_from_config(CONFIG)