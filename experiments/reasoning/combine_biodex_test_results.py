#!/usr/bin/env python3
"""
Combine separate test evaluation results for BioDEX methods into a single test_frontier_summary.json file.

This script can handle different formats of test results and combines them into the expected format
for the test frontier evaluation system.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

def load_test_results_from_file(file_path: Path) -> Optional[List[Dict[str, Any]]]:
    """Load test results from a file, handling different possible formats."""
    if not file_path.exists():
        print(f"⚠️  File not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle different possible formats
        if isinstance(data, list):
            # Direct list of results
            return data
        elif isinstance(data, dict):
            if "results" in data:
                # Results embedded in a dict
                return data["results"]
            elif "frontier_points" in data:
                # Pareto frontier format with test results
                points = []
                for point in data["frontier_points"]:
                    if "test_cost" in point and "test_accuracy" in point:
                        points.append({
                            "file": Path(point.get("file", "")).stem,
                            "cost": point["test_cost"],
                            "accuracy": point["test_accuracy"],
                            "accuracy_metric": point.get("test_accuracy_metric", "avg_rp_at_10")
                        })
                return points
            else:
                # Single result
                return [data]
        else:
            print(f"⚠️  Unexpected data format in {file_path}")
            return None
            
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

def load_test_results_from_directory(method_dir: Path, method_name: str) -> Dict[str, Any]:
    """Load test results from a method directory, trying different possible locations."""
    print(f"📊 Processing {method_name}...")
    
    # Possible test result file locations and names
    possible_files = [
        method_dir / "test_results.json",
        method_dir / "test_evaluation.json", 
        method_dir / "test_frontier_results.json",
        method_dir / "pareto_frontier_biodex.json",  # If it has test results embedded
        method_dir / "evaluation_metrics.json",     # If it contains test results
    ]
    
    # Also check for test_plans subdirectory
    test_plans_dir = method_dir / "test_plans" / method_name
    if test_plans_dir.exists():
        possible_files.extend([
            test_plans_dir / "test_results.json",
            test_plans_dir / "test_evaluation.json",
        ])
    
    # Try to load from each possible location
    for file_path in possible_files:
        if file_path.exists():
            print(f"  📄 Found test results at: {file_path}")
            results = load_test_results_from_file(file_path)
            if results:
                return {
                    "success": True,
                    "results": results
                }
    
    # If no test results found, try to extract from pareto frontier (for baseline comparison)
    pareto_file = method_dir / "pareto_frontier_biodex.json"
    if pareto_file.exists():
        print(f"  📄 Using pareto frontier data from: {pareto_file}")
        pareto_data = load_test_results_from_file(pareto_file)
        if pareto_data:
            return {
                "success": True,
                "results": pareto_data,
                "note": "Using training frontier data (no test results found)"
            }
    
    return {
        "success": False,
        "error": f"No test results found in {method_dir}"
    }

def create_original_baseline_result() -> Dict[str, Any]:
    """Create a placeholder for original baseline results."""
    # You can modify this to load actual original baseline test results
    # For now, creating a placeholder
    return {
        "success": False,
        "error": "Original baseline test results not found. Please run original baseline test separately."
    }

def combine_test_results() -> Dict[str, Any]:
    """Combine test results from all BioDEX methods."""
    
    # Base directory for exp_9.1 experiments
    base_dir = Path("docetl/exp_9.1")
    
    # Method directories
    methods = {
        "simple_baseline": base_dir / "biodex_simple_baseline",
        "baseline": base_dir / "biodex_baseline", 
        "mcts": base_dir / "biodex_mcts"
    }
    
    # Initialize results structure
    all_results = {}
    
    # Add original baseline placeholder
    all_results["original"] = create_original_baseline_result()
    
    # Process each method
    for method_name, method_dir in methods.items():
        result = load_test_results_from_directory(method_dir, method_name)
        all_results[method_name] = result
        
        if result["success"]:
            points = len(result.get("results", []))
            print(f"  ✅ Found {points} test points")
            if "note" in result:
                print(f"     Note: {result['note']}")
        else:
            print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
    
    # Create summary structure
    summary = {
        "dataset": "biodex",
        "timestamp": datetime.now().isoformat(),
        "methods_processed": list(all_results.keys()),
        "successful_methods": [m for m, r in all_results.items() if r.get("success", False)],
        "failed_methods": [m for m, r in all_results.items() if not r.get("success", False)],
        "results": all_results
    }
    
    return summary

def save_summary_with_custom_path(summary: Dict[str, Any], custom_path: Optional[str] = None):
    """Save the summary to a file, with option for custom path."""
    if custom_path:
        output_path = Path(custom_path)
    else:
        # Default location
        output_dir = Path("experiments/reasoning/outputs/biodex_original")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "test_frontier_summary.json"
    
    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Test frontier summary saved to: {output_path}")
    return output_path

def print_summary_statistics(summary: Dict[str, Any]):
    """Print summary statistics."""
    print(f"\n📊 Summary:")
    print(f"   Dataset: {summary['dataset']}")
    print(f"   Successful methods: {', '.join(summary['successful_methods'])}")
    if summary['failed_methods']:
        print(f"   Failed methods: {', '.join(summary['failed_methods'])}")
    
    total_points = 0
    for method, result in summary['results'].items():
        if result.get('success'):
            points = len(result.get('results', []))
            total_points += points
            print(f"   - {method}: {points} test points")
    
    print(f"   Total test points: {total_points}")

def main():
    """Main function to combine and save test results."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Combine BioDEX test evaluation results")
    parser.add_argument("--output", "-o", help="Custom output path for the summary file")
    parser.add_argument("--original-results", help="Path to original baseline test results file")
    
    args = parser.parse_args()
    
    print("🚀 Combining BioDEX test evaluation results...")
    
    # If original results file is provided, load it
    if args.original_results:
        original_file = Path(args.original_results)
        if original_file.exists():
            print(f"📄 Loading original baseline results from: {original_file}")
            original_results = load_test_results_from_file(original_file)
            if original_results:
                # Update the original baseline result
                def create_original_baseline_result():
                    return {
                        "success": True,
                        "results": original_results
                    }
    
    # Generate summary
    summary = combine_test_results()
    
    # Save summary file
    output_path = save_summary_with_custom_path(summary, args.output)
    
    # Print summary statistics
    print_summary_statistics(summary)
    
    print(f"\n💡 Next steps:")
    print(f"   1. If you have original baseline test results, provide them with --original-results")
    print(f"   2. Run the test frontier plot generation:")
    print(f"      modal run experiments/reasoning/run_test_frontier.py --dataset biodex --plot-only")

if __name__ == "__main__":
    main()
