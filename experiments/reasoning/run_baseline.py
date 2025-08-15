#!/usr/bin/env python3
"""
Baseline Agent Experiment Runner

This script runs the baseline reasoning agent that suggests directive optimizations
for DocETL pipelines. It's extracted from the reasoning_optimizer/agent.py to
provide a clean experiment interface.
"""

from ast import Continue
import os
import json
import argparse
import glob
from pathlib import Path
from typing import Dict, Any
from docetl.utils import extract_output_from_json
import matplotlib.pyplot as plt
from docetl.runner import DSLRunner
import yaml
import shutil
import re
from experiments.reasoning.evaluation.utils import run_dataset_evaluation


from docetl.reasoning_optimizer.agent import (
    run_single_iteration, 
    save_message_history, 
    load_message_history
)
from docetl.reasoning_optimizer.directives import DEFAULT_MODEL, DEFAULT_MAX_TPM, DEFAULT_OUTPUT_DIR

# Modal integration (mirrors experiments/reasoning/run_mcts.py)
import modal
from experiments.reasoning.utils import app, volume, VOLUME_MOUNT_PATH, image



def _resolve_in_volume(path: str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((Path(VOLUME_MOUNT_PATH) / p).resolve())


def _rewrite_pipeline_yaml_for_modal(orig_yaml_path: str, experiment_name: str) -> str:
    with open(orig_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    base_mount = Path(VOLUME_MOUNT_PATH)

    # Ensure pipeline outputs go into the mounted volume under outputs/experiment_name
    pipeline_cfg = cfg.get("pipeline", {})
    output_root = base_mount / "outputs" / experiment_name
    if isinstance(pipeline_cfg, dict):
        out = pipeline_cfg.get("output")
        if isinstance(out, dict):
            if isinstance(out.get("path"), str):
                original_name = Path(out["path"]).name
                output_root.mkdir(parents=True, exist_ok=True)
                out["path"] = str(output_root / original_name)
            if isinstance(out.get("intermediate_dir"), str):
                out["intermediate_dir"] = str(output_root / "intermediates")

    # Save rewritten YAML into volume tmp
    tmp_dir = base_mount / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    new_yaml_path = tmp_dir / f"{Path(orig_yaml_path).stem}_modal.yaml"
    with open(new_yaml_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return str(new_yaml_path)


@app.function(image=image, secrets=[modal.Secret.from_dotenv()], volumes={VOLUME_MOUNT_PATH: volume}, timeout=60 * 60 * 12)
def run_baseline_remote(
    yaml_path: str,
    data_dir: str | None = None,
    output_dir: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tpm: int = DEFAULT_MAX_TPM,
    iterations: int = 1,
    experiment_name: str = "baseline_experiment",
    dataset: str = "cuad",
    ground_truth_path: str | None = None,
    original_query_result: Dict[str, Any] | None = None,
):
    os.environ["EXPERIMENT_OUTPUT_DIR"] = str(Path(VOLUME_MOUNT_PATH) / "outputs")
    resolved_output_dir = _resolve_in_volume(output_dir) if output_dir else None
    resolved_data_dir = _resolve_in_volume(data_dir) if data_dir else None

    # Write a temporary YAML with output paths rewritten into the mounted volume
    modal_yaml_path = _rewrite_pipeline_yaml_for_modal(yaml_path, experiment_name)

    results = run_baseline_experiment(
        yaml_path=modal_yaml_path,
        data_dir=resolved_data_dir,
        output_dir=resolved_output_dir,
        model=model,
        max_tpm=max_tpm,
        iterations=iterations,
        experiment_name=experiment_name,
        dataset=dataset,
        ground_truth_path=ground_truth_path,
        original_query_result=original_query_result,
    )
    volume.commit()
    return results


@app.local_entrypoint()
def modal_main_baseline(
    yaml_path: str,
    experiment_name: str,
    data_dir: str | None = None,
    output_dir: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tpm: int = DEFAULT_MAX_TPM,
    iterations: int = 1,
    dataset: str = "cuad",
    ground_truth: str | None = None,
    original_query_result: Dict[str, Any] | None = None,
):
    run_baseline_remote.remote(
        yaml_path=yaml_path,
        data_dir=data_dir,
        output_dir=output_dir,
        model=model,
        max_tpm=max_tpm,
        iterations=iterations,
        experiment_name=experiment_name,
        dataset=dataset,
        ground_truth_path=ground_truth,
        original_query_result=original_query_result,
    )


def run_baseline_experiment(
    yaml_path: str,
    data_dir: str = None,
    output_dir: str = None,
    model: str = DEFAULT_MODEL,
    max_tpm: int = DEFAULT_MAX_TPM,
    iterations: int = 1,
    experiment_name: str = "baseline_experiment",
    dataset: str = "cuad",
    ground_truth_path: str | None = None,
    original_query_result: Dict[str, Any] | None = None,
):
    """
    Run baseline agent experiment with specified parameters.
    
    Args:
        yaml_path: Path to the input YAML pipeline file
        data_dir: Directory containing input data files  
        output_dir: Directory to save experiment outputs
        model: LLM model to use
        max_tpm: Tokens per minute limit
        iterations: Number of optimization iterations
        experiment_name: Name for this experiment run
    """
    
    # Set up environment
    if data_dir:
        os.environ['EXPERIMENT_DATA_DIR'] = data_dir
    
    # Ensure experiment name includes 'baseline' to avoid mixing with MCTS runs
    if "baseline" not in experiment_name.lower():
        experiment_name += "_baseline"

    # Determine output directory (env var, parameter, or default)
    if output_dir is None:
        output_dir = os.environ.get('EXPERIMENT_OUTPUT_DIR', DEFAULT_OUTPUT_DIR)
    
    # Create output directory
    output_path = Path(output_dir) / experiment_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Running Baseline Agent Experiment")
    print(f"=" * 50)
    print(f"Input Pipeline: {yaml_path}")
    print(f"Data Directory: {data_dir or 'default'}")
    print(f"Output Directory: {output_path}")
    print(f"Model: {model}")
    print(f"Iterations: {iterations}")
    print(f"Dataset for evaluation: {dataset}")
    print()
    
    # Initialise message history
    message_history_file = output_path / "message_history.json"
    message_history = load_message_history(str(message_history_file))

    # ------------------------------------------------------------------
    # Baseline run (iteration 0) - use original query result if available
    # ------------------------------------------------------------------
    
    if original_query_result and original_query_result["success"]:
        print("✅ Using pre-executed original query result")
        # Use the pre-executed original query result
        orig_output_sample = original_query_result["sample_output"]
        baseline_cost = original_query_result["cost"]
        
        # Copy the original output file to our experiment directory for consistency
        if original_query_result["output_file_path"]:
            import shutil
            baseline_json_path = output_path / "original_output.json"
            try:
                shutil.copy2(original_query_result["output_file_path"], baseline_json_path)
            except Exception as e:
                print(f"⚠️  Could not copy original output file: {e}")
        
        results = [{
            "iteration": 0,
            "output_file": str(output_path / "original_output.json"),
            "success": True,
            "cost": baseline_cost,
        }]
        prev_cost = baseline_cost
        print(f"✅ Baseline cost: ${baseline_cost:.4f}")
    else:
        print("▶️  Running baseline (iteration 0) - original query result not available")
        
        def run_baseline_pipeline(yaml_path: str, out_dir: Path):
            """Execute original YAML once, save JSON and return (sample_dict, cost)."""
            # Load original YAML
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)

            # Redirect output path to experiment folder
            baseline_json_path = out_dir / "original_output.json"
            try:
                config['pipeline']['output']['path'] = str(baseline_json_path)
            except Exception:
                # Fallback if structure is different
                config.setdefault('pipeline', {}).setdefault('output', {})['path'] = str(baseline_json_path)

            # Force fresh run
            config['bypass_cache'] = True

            # Save modified YAML for provenance
            baseline_yaml_path = out_dir / "baseline_config.yaml"
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
            try:
                orig_output_sample = extract_output_from_json(str(baseline_yaml_path), str(baseline_json_path))[:1]
            except Exception as e:
                print(f"⚠️  Could not load baseline output JSON: {e}")
                orig_output_sample = []

            return orig_output_sample, total_cost

        try:
            orig_output_sample, baseline_cost = run_baseline_pipeline(yaml_path, output_path)
            results = [{
                "iteration": 0,
                "output_file": str(output_path / "original_output.json"),
                "success": True,
                "cost": baseline_cost,
            }]
            prev_cost = baseline_cost
            print(f"✅ Baseline executed successfully, cost: ${baseline_cost:.4f}")
        except Exception as e:
            print(f"❌ Baseline run failed: {e}")
            orig_output_sample = []
            prev_cost = 0.0
            results = [{
                "iteration": 0,
                "success": False,
                "error": str(e),
                "cost": 0.0,
            }]
 
    # ------------------------------------------------------------------
    # Optimisation iterations
    # ------------------------------------------------------------------
    curr_yaml_path = yaml_path
    for i in range(1, iterations + 1):
        try:
            print(f"\n🔄 Running Iteration {i}/{iterations}")
            
            # Run single iteration
            # Load sample data for the dataset
            sample_data = []
            if dataset.lower() == "cuad":
                sample_data_path = Path("experiments/reasoning/data/train/cuad.json")
            elif dataset.lower() == "blackvault":
                sample_data_path = Path("experiments/reasoning/data/train/blackvault.json")
            elif dataset.lower() == "game_reviews" or dataset.lower() == "reviews":
                sample_data_path = Path("experiments/reasoning/data/train/game_reviews.json")
            elif dataset.lower() == "sustainability":
                sample_data_path = Path("experiments/reasoning/data/train/sustainability.json")
            elif dataset.lower() == "biodex":
                sample_data_path = Path("experiments/reasoning/data/train/biodex.json")
            elif dataset.lower() == "medec":
                sample_data_path = Path("experiments/reasoning/data/train/medec.json")
            else:
                sample_data_path = None
            
            if sample_data_path and sample_data_path.exists():
                try:
                    with open(sample_data_path, 'r') as f:
                        sample_data = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not load sample data from {sample_data_path}: {e}")
            print("="*100)
            print("curr_yaml_path: ", curr_yaml_path)
            output_file, updated_history, iteration_cost = run_single_iteration(
                yaml_path=curr_yaml_path,
                model=model,
                max_tpm=max_tpm,
                message_history=message_history,
                iteration_num=i,
                orig_output_sample=orig_output_sample,
                prev_plan_cost=prev_cost,
                output_dir=str(output_path),
                dataset=dataset,
                sample_data=sample_data
            )

            if not output_file:
                print(f"❌ Iteration {i} failed: No output yaml file")
                results.append({
                    "iteration": i,
                    "error": "No output file",
                    "success": False,
                    "cost": 0.0
                })
                continue

            curr_yaml_path = output_file
            print("output_file: ", output_file)
            
            # Update message history
            message_history = updated_history
            save_message_history(message_history, str(message_history_file))
            
            results.append({
                "iteration": i,
                "output_file": str(output_file),
                "success": True,
                "cost": iteration_cost
            })

            # Prepare for next iteration
            prev_cost = iteration_cost
            try:
                # The pipeline saved results to iteration_{i}_results.json – load a snippet
                result_json_path = output_path / f"iteration_{i}_results.json"
                if result_json_path.exists():
                    orig_output_sample = extract_output_from_json(output_file, result_json_path)[:1]
                else:
                    orig_output_sample = ""
            except Exception:
                orig_output_sample = ""

        except Exception as e:
            print(f"❌ Iteration {i} failed: {e}")
            results.append({
                "iteration": i,
                "error": str(e),
                "success": False,
                "cost": 0.0
            })
            print(f"🛑 Stopping optimization loop due to iteration failure. Proceeding to evaluation...")
            break
    
    # ------------------------------------------------------------------
    # Evaluation (similar to MCTS)
    # ------------------------------------------------------------------

    # Look for result JSONs produced by pipeline executions
    json_files = [
        p for p in glob.glob(str(output_path / "**/*.json"), recursive=True)
        if p.endswith("_results.json") or p.endswith("original_output.json")
    ]

    # Prepare files with cost information for evaluation
    files_for_evaluation = []
    for jf in json_files:
        # Extract iteration number from filename to get corresponding cost
        iteration_match = re.search(r'iteration_(\d+)_results\.json', jf)
        iteration_cost = 0.0  # Default cost
        if iteration_match:
            iteration_num = int(iteration_match.group(1))
        else:
            # Check for baseline file
            if jf.endswith("original_output.json"):
                iteration_num = 0
            else:
                iteration_num = None

        if iteration_num is not None:
            for result in results:
                if result["iteration"] == iteration_num and result["success"]:
                    iteration_cost = result.get("cost", 0.0)
                    break
        
        files_for_evaluation.append({
            "file_path": jf,
            "cost": iteration_cost,
            "node_id": Path(jf).stem,  # Use filename as node_id
        })

    # Use original query result cost if available, otherwise use baseline cost from first result
    root_cost = original_query_result["cost"] if (original_query_result and original_query_result["success"]) else (results[0].get("cost", 0.0) if results else 0.0)
    
    eval_results, pareto_auc = run_dataset_evaluation(
        dataset=dataset,
        nodes_or_files=files_for_evaluation,
        output_path=output_path,
        ground_truth_path=ground_truth_path,
        method_name="docetl_baseline",
        root_cost=root_cost
    )

    # Save experiment summary
    experiment_summary = {
        "experiment_name": experiment_name,
        "input_pipeline": yaml_path,
        "model": model,
        "iterations": iterations,
        "max_tpm": max_tpm,
        "data_dir": data_dir,
        "output_dir": str(output_path),
        "results": results,
        "dataset": dataset,
        "ground_truth": ground_truth_path,
        "success_rate": sum(1 for r in results if r["success"]) / len(results),
        "original_query_cost": original_query_result["cost"] if original_query_result and original_query_result["success"] else None,
        "original_query_success": original_query_result["success"] if original_query_result else None,
    }
    if eval_results:
        experiment_summary["evaluation_file"] = str(output_path / "evaluation_metrics.json")

    summary_file = output_path / "experiment_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(experiment_summary, f, indent=2)
    
    print(f"\n📋 Experiment Summary:")
    print(f"   Success Rate: {experiment_summary['success_rate']:.1%}")
    print(f"   Summary saved to: {summary_file}")
    print(f"   All outputs in: {output_path}")
    
    return experiment_summary

def main():
    parser = argparse.ArgumentParser(
        description="Run baseline reasoning agent experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run
  python run_baseline.py --yaml_path ./pipeline.yaml --experiment_name my_test
  
  # With custom data directory and output location  
  python run_baseline.py --yaml_path ./pipeline.yaml --data_dir ./data --output_dir ./results --experiment_name experiment_1
  
  # Multiple iterations with different model
  python run_baseline.py --yaml_path ./pipeline.yaml --iterations 3 --model gpt-4o --experiment_name multi_iter
        """
    )
    
    parser.add_argument("--yaml_path", type=str, required=True,
                       help="Path to the input YAML pipeline file")
    parser.add_argument("--data_dir", type=str, 
                       help="Directory containing input data files (sets EXPERIMENT_DATA_DIR)")
    parser.add_argument("--output_dir", type=str, 
                       help=f"Directory to save experiment outputs (default: EXPERIMENT_OUTPUT_DIR env var or {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                       help=f"LLM model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--max_tpm", type=int, default=DEFAULT_MAX_TPM,
                       help=f"Tokens per minute limit (default: {DEFAULT_MAX_TPM})")
    parser.add_argument("--iterations", type=int, default=1,
                       help="Number of optimization iterations (default: 1)")
    parser.add_argument("--experiment_name", type=str, required=True,
                       help="Name for this experiment run")
    parser.add_argument("--dataset", type=str, default="cuad", help="Dataset name for evaluation (default: cuad)")
    parser.add_argument("--ground_truth", type=str, help="Path to ground-truth file (if not default)")
    
    args = parser.parse_args()
    
    try:
        result = run_baseline_experiment(
            yaml_path=args.yaml_path,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model=args.model,
            max_tpm=args.max_tpm,
            iterations=args.iterations,
            experiment_name=args.experiment_name,
            dataset=args.dataset,
            ground_truth_path=args.ground_truth
        )

        evaluation_file = result["evaluation_file"]
        with open(evaluation_file, 'r') as f:
            evaluation_results = json.load(f)
        
        print("\n🎉 Baseline experiment completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n💥 Experiment failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())