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
from pathlib import Path

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
    experiment_name: str = "baseline_experiment"
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
    print(f"Experiment: {experiment_name}")
    print()
    
    # Initialize message history
    message_history_file = output_path / "message_history.json"
    message_history = load_message_history(str(message_history_file))
    
    # Load original output if available for comparison
    orig_output_sample = ""
    orig_output_file = output_path / "original_output.json"
    if orig_output_file.exists():
        try:
            with open(orig_output_file, 'r') as f:
                orig_output_sample = json.load(f)
                print(f"📊 Loaded original output for comparison")
        except Exception as e:
            print(f"⚠️  Could not load original output: {e}")
    
    # Run experiment iterations
    results = []
    for i in range(1, iterations + 1):
        try:
            print(f"\n🔄 Running Iteration {i}/{iterations}")
            
            # Run single iteration
            output_file, updated_history = run_single_iteration(
                yaml_path=yaml_path,
                model=model,
                max_tpm=max_tpm,
                message_history=message_history,
                iteration_num=i,
                orig_output_sample=orig_output_sample
            )
            
            # Save iteration results
            iteration_output = output_path / f"iteration_{i}_output.yaml"
            if output_file and os.path.exists(output_file):
                import shutil
                shutil.copy2(output_file, iteration_output)
                print(f"✅ Iteration {i} output saved to: {iteration_output}")
            
            # Update message history
            message_history = updated_history
            save_message_history(message_history, str(message_history_file))
            
            results.append({
                "iteration": i,
                "output_file": str(iteration_output),
                "success": True
            })
            
        except Exception as e:
            print(f"❌ Iteration {i} failed: {e}")
            results.append({
                "iteration": i,
                "error": str(e),
                "success": False
            })
    
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
        "success_rate": sum(1 for r in results if r["success"]) / len(results)
    }
    
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