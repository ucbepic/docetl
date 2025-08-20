#!/usr/bin/env python3
"""
Script to re-evaluate plans with combined_accuracy_score of 0.0 in game_reviews evaluation.

This script:
1. Loads the evaluation_metrics.json file
2. Identifies plans with combined_accuracy_score = 0.0
3. Executes the corresponding YAML pipeline files using docetl
4. Re-evaluates the generated results
5. Updates the evaluation_metrics.json with new scores
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add the current directory to path to import evaluation modules
sys.path.append(str(Path(__file__).parent))

from experiments.reasoning.evaluation.game_reviews import evaluate_results as game_reviews_evaluate


def load_evaluation_metrics(metrics_file: Path) -> List[Dict[str, Any]]:
    """Load the current evaluation metrics."""
    with open(metrics_file, 'r') as f:
        return json.load(f)


def save_evaluation_metrics(metrics_file: Path, metrics: List[Dict[str, Any]]) -> None:
    """Save the updated evaluation metrics."""
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)


def identify_zero_score_plans(metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identify plans with combined_accuracy_score of 0.0."""
    return [plan for plan in metrics if plan.get("combined_accuracy_score", 0.0) == 0.0]


def run_docetl_pipeline(yaml_file: Path) -> bool:
    """Run a DocETL pipeline using the CLI and return success status."""
    try:
        print(f"Executing pipeline: {yaml_file}")
        result = subprocess.run(
            [sys.executable, "-m", "docetl.cli", "run", str(yaml_file)],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode == 0:
            print(f"✅ Successfully executed {yaml_file.name}")
            return True
        else:
            print(f"❌ Failed to execute {yaml_file.name}")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout executing {yaml_file.name}")
        return False
    except Exception as e:
        print(f"❌ Exception executing {yaml_file.name}: {e}")
        return False


def evaluate_result_file(result_file: Path) -> Dict[str, Any]:
    """Evaluate a result JSON file and return metrics."""
    if not result_file.exists():
        print(f"⚠️  Result file does not exist: {result_file}")
        return {
            "combined_accuracy_score": 0.0,
            "sentiment_accuracy": 0.0,
            "kendall_tau_score": 0.0,
            "weighted_score": 0.0
        }
    
    try:
        metrics = game_reviews_evaluate("docetl", str(result_file))
        return {
            "combined_accuracy_score": metrics["weighted_score"],
            "sentiment_accuracy": metrics["sentiment_accuracy"],
            "kendall_tau_score": metrics["kendall_tau_score"],
            "weighted_score": metrics["weighted_score"]
        }
    except Exception as e:
        print(f"⚠️  Failed to evaluate {result_file}: {e}")
        return {
            "combined_accuracy_score": 0.0,
            "sentiment_accuracy": 0.0,
            "kendall_tau_score": 0.0,
            "weighted_score": 0.0
        }


def main():
    """Main function to re-evaluate zero-scored plans."""
    # Define paths
    base_dir = Path(__file__).parent
    metrics_file = base_dir / "experiments/reasoning/outputs/game_reviews/mcts/evaluation_metrics.json"
    yaml_dir = base_dir / "experiments/reasoning/outputs/modal_outputs/game_reviews_mcts"
    result_dir = yaml_dir  # Results are in the same directory as YAML files
    
    print("🔍 Loading evaluation metrics...")
    if not metrics_file.exists():
        print(f"❌ Evaluation metrics file not found: {metrics_file}")
        return
    
    metrics = load_evaluation_metrics(metrics_file)
    print(f"📊 Loaded {len(metrics)} evaluation results")
    
    # Identify zero-score plans
    zero_score_plans = identify_zero_score_plans(metrics)
    print(f"🎯 Found {len(zero_score_plans)} plans with 0.0 combined_accuracy_score:")
    
    for plan in zero_score_plans:
        print(f"  - {plan['file']} (node_id: {plan['node_id']})")
    
    if not zero_score_plans:
        print("✅ No plans with zero scores found!")
        return
    
    # Process each zero-score plan
    updated_count = 0
    for plan in zero_score_plans:
        file_name = plan["file"]
        node_id = plan["node_id"]
        
        # Determine corresponding YAML and result files
        # Remove .json extension and add .yaml
        yaml_name = file_name.replace(".json", ".yaml")
        yaml_file = yaml_dir / yaml_name
        result_file = result_dir / file_name
        
        print(f"\n🔄 Processing {file_name} (node_id: {node_id})...")
        
        # Check if YAML file exists
        if not yaml_file.exists():
            print(f"⚠️  YAML file not found: {yaml_file}")
            continue
        
        # Run the DocETL pipeline
        if run_docetl_pipeline(yaml_file):
            # Evaluate the generated result
            print(f"📊 Evaluating generated result: {result_file}")
            new_metrics = evaluate_result_file(result_file)
            
            # Update the plan metrics
            plan.update(new_metrics)
            # Also update mcts_accuracy to match combined_accuracy_score
            plan["mcts_accuracy"] = new_metrics["combined_accuracy_score"]
            
            print(f"✅ Updated metrics for {file_name}:")
            print(f"   Combined accuracy: {new_metrics['combined_accuracy_score']:.4f}")
            print(f"   Sentiment accuracy: {new_metrics['sentiment_accuracy']:.4f}")
            print(f"   Kendall tau score: {new_metrics['kendall_tau_score']:.4f}")
            
            updated_count += 1
        else:
            print(f"❌ Failed to execute pipeline for {file_name}")
    
    # Save updated metrics
    if updated_count > 0:
        print(f"\n💾 Saving updated evaluation metrics...")
        save_evaluation_metrics(metrics_file, metrics)
        print(f"✅ Successfully updated {updated_count} plans in {metrics_file}")
    else:
        print(f"\n⚠️  No plans were successfully updated")
    
    print(f"\n🎉 Re-evaluation complete!")


if __name__ == "__main__":
    main()