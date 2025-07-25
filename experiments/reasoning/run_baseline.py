#!/usr/bin/env python3
"""
Baseline Agent Experiment Runner

This script runs the baseline reasoning agent that suggests directive optimizations
for DocETL pipelines. It's extracted from the reasoning_optimizer/agent.py to
provide a clean experiment interface.
"""

import os
import json
import argparse
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from docetl.runner import DSLRunner
import yaml as _yaml
import shutil
import re
from experiments.reasoning.evaluation.cuad import evaluate_results as cuad_evaluate


from docetl.reasoning_optimizer.agent import (
    run_single_iteration, 
    save_message_history, 
    load_message_history
)
from docetl.reasoning_optimizer.directives import DEFAULT_MODEL, DEFAULT_MAX_TPM, DEFAULT_OUTPUT_DIR

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
    # Baseline run (iteration 0)
    # ------------------------------------------------------------------

    def run_baseline_pipeline(yaml_path: str, out_dir: Path):
        """Execute original YAML once, save JSON and return (sample_dict, cost)."""
        

        # Load original YAML
        with open(yaml_path, 'r') as f:
            config = _yaml.safe_load(f)

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
            _yaml.dump(config, f, sort_keys=False)

        # Run pipeline
        runner = DSLRunner.from_yaml(str(baseline_yaml_path))
        runner.load()
        if runner.last_op_container:
            data, _, _ = runner.last_op_container.next()
            runner.save(data)
        total_cost = runner.total_cost
        runner.reset_env()

        # Load sample output (truncate if huge)
        sample_output = {}
        try:
            with open(baseline_json_path, 'r') as jf:
                sample_output = json.load(jf)
        except Exception as e:
            print(f"⚠️  Could not load baseline output JSON: {e}")

        return sample_output, total_cost


    print("▶️  Running baseline (iteration 0)")
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
        orig_output_sample = ""
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

    for i in range(1, iterations + 1):
        try:
            print(f"\n🔄 Running Iteration {i}/{iterations}")
            
            # Run single iteration
            output_file, updated_history, iteration_cost = run_single_iteration(
                yaml_path=yaml_path,
                model=model,
                max_tpm=max_tpm,
                message_history=message_history,
                iteration_num=i,
                orig_output_sample=orig_output_sample,
                prev_plan_cost=prev_cost,
                output_dir=str(output_path)
            )
            
            # Save iteration results
            iteration_output = output_path / f"iteration_{i}_output.yaml"
            if output_file and os.path.exists(output_file):
                shutil.copy2(output_file, iteration_output)
                print(f"✅ Iteration {i} output saved to: {iteration_output}")
            
            # Update message history
            message_history = updated_history
            save_message_history(message_history, str(message_history_file))
            
            results.append({
                "iteration": i,
                "output_file": str(iteration_output),
                "success": True,
                "cost": iteration_cost
            })

            # Prepare for next iteration
            prev_cost = iteration_cost
            try:
                # The pipeline saved results to iteration_{i}_results.json – load a snippet
                result_json_path = output_path / f"iteration_{i}_results.json"
                if result_json_path.exists():
                    with open(result_json_path, 'r') as rf:
                        result_json = json.load(rf)
                    orig_output_sample = json.dumps(result_json)[:2000]
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
    
    # ------------------------------------------------------------------
    # Evaluation (similar to MCTS)
    # ------------------------------------------------------------------

    eval_results = []
    if dataset.lower() == "cuad":
        if ground_truth_path is None:
            default_gt = Path("experiments/reasoning/data/CUAD-master_clauses.csv")
            ground_truth_path = str(default_gt)

        # Look for result JSONs produced by pipeline executions
        json_files = [
            p for p in glob.glob(str(output_path / "**/*.json"), recursive=True)
            if p.endswith("_results.json") or p.endswith("original_output.json")
        ]

        if json_files:
            

            for jf in json_files:
                try:
                    metrics = cuad_evaluate("docetl_baseline", jf, ground_truth_path)
                    display_path = str(Path(jf).resolve().relative_to(output_path.resolve()))
                    
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
                    
                    eval_results.append({
                        "file": display_path,
                        "precision": metrics["avg_precision"],
                        "recall": metrics["avg_recall"],
                        "f1": metrics["avg_f1"],
                        "cost": iteration_cost,
                    })
                except Exception as e:
                    print(f"⚠️  Evaluation failed for {jf}: {e}")

            if eval_results:
                eval_out_file = output_path / "evaluation_metrics.json"
                with open(eval_out_file, "w") as f:
                    json.dump(eval_results, f, indent=2)
                print(f"📊 Evaluation metrics saved to {eval_out_file}")

                # Plot Cost vs F1 (cost None -> skip)
                try:
                    xs = [row["cost"] if row["cost"] is not None else 0 for row in eval_results]
                    ys = [row["f1"] for row in eval_results]
                    plt.figure(figsize=(8,6))
                    plt.scatter(xs, ys, color="grey")
                    for row in eval_results:
                        plt.annotate(row["file"], (row["cost"] if row["cost"] is not None else 0, row["f1"]), textcoords="offset points", xytext=(4,4), fontsize=8)
                    plt.xlabel("Cost ($)")
                    plt.ylabel("F1 Score")
                    plt.title("Baseline: Cost vs F1")
                    plt.grid(True, linestyle="--", alpha=0.5)
                    plot_path = output_path / "cost_vs_f1.png"
                    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
                    plt.close()
                    print(f"📈 Scatter plot saved to: {plot_path}")
                except Exception as e:
                    print(f"⚠️  Failed to create scatter plot: {e}")

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
        "success_rate": sum(1 for r in results if r["success"]) / len(results)
    }
    if eval_results:
        experiment_summary["evaluation_file"] = str(eval_out_file)

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
    
    args = parser.parse_args()
    
    try:
        result = run_baseline_experiment(
            yaml_path=args.yaml_path,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model=args.model,
            max_tpm=args.max_tpm,
            iterations=args.iterations,
            experiment_name=args.experiment_name
        )
        
        print("\n🎉 Baseline experiment completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n💥 Experiment failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())