#!/usr/bin/env python3
"""
Test script for Pareto frontier identification
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.reasoning.evaluation.utils import identify_pareto_frontier, print_pareto_frontier_summary

def test_pareto_frontier():
    """Test the Pareto frontier identification with sample data"""
    
    # Sample evaluation results for CUAD dataset
    sample_results = [
        {"file": "plan1.json", "cost": 0.5, "f1": 0.7, "precision": 0.8, "recall": 0.6},
        {"file": "plan2.json", "cost": 1.0, "f1": 0.8, "precision": 0.9, "recall": 0.7},
        {"file": "plan3.json", "cost": 0.8, "f1": 0.75, "precision": 0.85, "recall": 0.65},
        {"file": "plan4.json", "cost": 1.2, "f1": 0.9, "precision": 0.95, "recall": 0.85},
        {"file": "plan5.json", "cost": 0.3, "f1": 0.6, "precision": 0.7, "recall": 0.5},
        {"file": "plan6.json", "cost": 1.5, "f1": 0.85, "precision": 0.9, "recall": 0.8},
    ]
    
    print("🧪 Testing Pareto Frontier Identification")
    print("=" * 50)
    print("Sample data:")
    for result in sample_results:
        print(f"  {result['file']}: Cost=${result['cost']:.2f}, F1={result['f1']:.2f}")
    
    print("\n" + "=" * 50)
    
    # Test Pareto frontier identification
    updated_results = identify_pareto_frontier(sample_results, "cuad")
    
    print("\n" + "=" * 50)
    print("Results after Pareto frontier identification:")
    for result in updated_results:
        frontier_status = "🏆 FRONTIER" if result.get("on_frontier", False) else "  dominated"
        print(f"  {result['file']}: Cost=${result['cost']:.2f}, F1={result['f1']:.2f} - {frontier_status}")
    
    print("\n" + "=" * 50)
    
    # Test summary printing
    print_pareto_frontier_summary(updated_results, "cuad")
    
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    test_pareto_frontier() 