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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal

# Import the existing Modal functions and shared volume/mount from the experiment runners
from experiments.reasoning import run_mcts as mcts_mod
from experiments.reasoning import run_baseline as baseline_mod
from experiments.reasoning.utils import app  # use the same App as the runners

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
            "dataset": "cuad",
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
    )


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
            )
            results.append((dataset, f"mcts:{mc_name}", call))
            print(f"Spawned MCTS for {dataset} as {mc_name}")

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