#!/usr/bin/env python3
"""
Generate test_frontier_summary.json for BioDEX from pareto frontier files.

This script reads the pareto_frontier_biodex.json files from the three method directories
(simple_baseline, baseline, mcts) and creates a summary file that can be used for
test frontier evaluation.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

def load_pareto_frontier(method_dir: Path) -> Dict[str, Any]:
    """Load pareto frontier data from a method directory."""
    pareto_file = method_dir / "pareto_frontier_biodex.json"
    if not pareto_file.exists():
        print(f"⚠️  Pareto frontier file not found: {pareto_file}")
        return None
    
    with open(pareto_file, 'r') as f:
        return json.load(f)

def extract_frontier_points(pareto_data: Dict[str, Any], method: str) -> List[Dict[str, Any]]:
    """Extract frontier points from pareto data and format them for test summary."""
    if not pareto_data or "frontier_points" not in pareto_data:
        return []
    
    points = []
    for point in pareto_data["frontier_points"]:
        # Extract the base name without .json extension
        file_name = point.get("file", "")
        base_name = Path(file_name).stem
        
        # Get cost and accuracy
        cost = point.get("cost", 0.0)
        accuracy = point.get("avg_rp_at_10", 0.0)  # BioDEX uses avg_rp_at_10 as accuracy metric
        
        points.append({
            "file": base_name,
            "cost": cost,
            "accuracy": accuracy,
            "accuracy_metric": "avg_rp_at_10"
        })
    
    return points

def generate_test_summary() -> Dict[str, Any]:
    """Generate the test frontier summary for BioDEX."""
    
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
    
    # Process each method
    for method_name, method_dir in methods.items():
        print(f"📊 Processing {method_name}...")
        
        # Load pareto frontier data
        pareto_data = load_pareto_frontier(method_dir)
        if not pareto_data:
            all_results[method_name] = {
                "success": False,
                "error": f"Could not load pareto frontier data from {method_dir}"
            }
            continue
        
        # Extract frontier points
        frontier_points = extract_frontier_points(pareto_data, method_name)
        
        if frontier_points:
            all_results[method_name] = {
                "success": True,
                "results": frontier_points
            }
            print(f"  ✅ Found {len(frontier_points)} frontier points")
        else:
            all_results[method_name] = {
                "success": False,
                "error": "No frontier points found"
            }
            print(f"  ⚠️  No frontier points found")
    
    # Create summary structure
    summary = {
        "dataset": "biodex",
        "timestamp": datetime.now().isoformat(),
        "methods_processed": list(methods.keys()),
        "successful_methods": [m for m, r in all_results.items() if r.get("success", False)],
        "failed_methods": [m for m, r in all_results.items() if not r.get("success", False)],
        "results": all_results
    }
    
    return summary

def main():
    """Main function to generate and save the test frontier summary."""
    print("🚀 Generating BioDEX test frontier summary...")
    
    # Generate summary
    summary = generate_test_summary()
    
    # Create output directory
    output_dir = Path("experiments/reasoning/outputs/biodex_original")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary file
    summary_file = output_dir / "test_frontier_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Test frontier summary saved to: {summary_file}")
    
    # Print summary statistics
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
            print(f"   - {method}: {points} frontier points")
    
    print(f"   Total frontier points: {total_points}")

if __name__ == "__main__":
    main()
