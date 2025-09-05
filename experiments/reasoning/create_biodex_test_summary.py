#!/usr/bin/env python3
"""
Create test_frontier_summary.json for BioDEX from existing test evaluation results.

This script creates the expected test_frontier_summary.json format by combining
test results from the three BioDEX methods (simple_baseline, baseline, mcts).
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

def create_test_summary_from_pareto_frontiers() -> Dict[str, Any]:
    """
    Create test frontier summary using the existing pareto frontier files.
    This assumes the pareto frontier data represents test results.
    """
    
    # Base directory for exp_9.1 experiments
    base_dir = Path("docetl/exp_9.1")
    
    # Method directories
    methods = {
        "simple_baseline": base_dir / "biodex_simple_baseline",
        "baseline": base_dir / "biodex_baseline", 
        "mcts": base_dir / "biodex_mcts"
    }
    
    all_results = {}
    
    # Process each method
    for method_name, method_dir in methods.items():
        print(f"📊 Processing {method_name}...")
        
        pareto_file = method_dir / "pareto_frontier_biodex.json"
        if not pareto_file.exists():
            print(f"  ❌ Pareto frontier file not found: {pareto_file}")
            all_results[method_name] = {
                "success": False,
                "error": f"Pareto frontier file not found: {pareto_file}"
            }
            continue
        
        try:
            with open(pareto_file, 'r') as f:
                pareto_data = json.load(f)
            
            # Extract frontier points and convert to test result format
            frontier_points = pareto_data.get("frontier_points", [])
            test_results = []
            
            for point in frontier_points:
                file_name = point.get("file", "")
                base_name = Path(file_name).stem
                
                # Extract cost and accuracy
                cost = point.get("cost", 0.0)
                accuracy = point.get("avg_rp_at_10", 0.0)  # BioDEX uses avg_rp_at_10
                
                test_results.append({
                    "file": base_name,
                    "cost": cost,
                    "accuracy": accuracy,
                    "accuracy_metric": "avg_rp_at_10"
                })
            
            if test_results:
                all_results[method_name] = {
                    "success": True,
                    "results": test_results
                }
                print(f"  ✅ Found {len(test_results)} frontier points")
            else:
                all_results[method_name] = {
                    "success": False,
                    "error": "No frontier points found"
                }
                print(f"  ⚠️  No frontier points found")
                
        except Exception as e:
            print(f"  ❌ Error processing {method_name}: {e}")
            all_results[method_name] = {
                "success": False,
                "error": str(e)
            }
    
    # Add original baseline placeholder (you can modify this if you have actual original results)
    all_results["original"] = {
        "success": False,
        "error": "Original baseline test results not found. Please run original baseline test separately."
    }
    
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

def create_test_summary_from_custom_results(
    simple_baseline_results: Optional[List[Dict[str, Any]]] = None,
    baseline_results: Optional[List[Dict[str, Any]]] = None,
    mcts_results: Optional[List[Dict[str, Any]]] = None,
    original_results: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Create test frontier summary from custom test results.
    
    Args:
        simple_baseline_results: List of test results for simple_baseline method
        baseline_results: List of test results for baseline method
        mcts_results: List of test results for mcts method
        original_results: List of test results for original baseline
    """
    
    all_results = {}
    
    # Add original baseline results
    if original_results:
        all_results["original"] = {
            "success": True,
            "results": original_results
        }
    else:
        all_results["original"] = {
            "success": False,
            "error": "Original baseline test results not provided"
        }
    
    # Add method results
    if simple_baseline_results:
        all_results["simple_baseline"] = {
            "success": True,
            "results": simple_baseline_results
        }
    else:
        all_results["simple_baseline"] = {
            "success": False,
            "error": "Simple baseline test results not provided"
        }
    
    if baseline_results:
        all_results["baseline"] = {
            "success": True,
            "results": baseline_results
        }
    else:
        all_results["baseline"] = {
            "success": False,
            "error": "Baseline test results not provided"
        }
    
    if mcts_results:
        all_results["mcts"] = {
            "success": True,
            "results": mcts_results
        }
    else:
        all_results["mcts"] = {
            "success": False,
            "error": "MCTS test results not provided"
        }
    
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

def save_summary(summary: Dict[str, Any], output_path: Optional[str] = None):
    """Save the summary to a file."""
    if output_path:
        output_file = Path(output_path)
    else:
        # Default location
        output_dir = Path("experiments/reasoning/outputs/biodex_original")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "test_frontier_summary.json"
    
    # Create parent directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Test frontier summary saved to: {output_file}")
    return output_file

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
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create BioDEX test frontier summary")
    parser.add_argument("--output", "-o", help="Custom output path for the summary file")
    parser.add_argument("--from-pareto", action="store_true", 
                       help="Create summary from existing pareto frontier files")
    parser.add_argument("--simple-baseline-file", help="Path to simple baseline test results")
    parser.add_argument("--baseline-file", help="Path to baseline test results")
    parser.add_argument("--mcts-file", help="Path to MCTS test results")
    parser.add_argument("--original-file", help="Path to original baseline test results")
    
    args = parser.parse_args()
    
    print("🚀 Creating BioDEX test frontier summary...")
    
    if args.from_pareto:
        # Create from pareto frontier files
        summary = create_test_summary_from_pareto_frontiers()
    else:
        # Load custom results from files
        def load_results_from_file(file_path):
            if not file_path:
                return None
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
                return None
        
        simple_baseline_results = load_results_from_file(args.simple_baseline_file)
        baseline_results = load_results_from_file(args.baseline_file)
        mcts_results = load_results_from_file(args.mcts_file)
        original_results = load_results_from_file(args.original_file)
        
        summary = create_test_summary_from_custom_results(
            simple_baseline_results=simple_baseline_results,
            baseline_results=baseline_results,
            mcts_results=mcts_results,
            original_results=original_results
        )
    
    # Save summary file
    output_file = save_summary(summary, args.output)
    
    # Print summary statistics
    print_summary_statistics(summary)
    
    print(f"\n💡 Next steps:")
    print(f"   1. If you have original baseline test results, provide them with --original-file")
    print(f"   2. Run the test frontier plot generation:")
    print(f"      modal run experiments/reasoning/run_test_frontier.py --dataset biodex --plot-only")

if __name__ == "__main__":
    main()
