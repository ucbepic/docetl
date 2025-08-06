#!/usr/bin/env python3
"""
Evaluation script for blackvault results using the evaluate_results function.
"""

import argparse
import sys
import os
import json

from experiments.reasoning.evaluation.blackvault import evaluate_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate blackvault results')
    parser.add_argument('json_file', help='Path to the JSON results file to evaluate')
    
    args = parser.parse_args()
    
    # Check if results file exists
    if not os.path.exists(args.json_file):
        print(f"Error: Results file '{args.json_file}' not found!")
        sys.exit(1)
    
    try:
        # Evaluate the results
        evaluation_metrics = evaluate_results(args.method_name, args.json_file)
        
        # Display the results
        print("EVALUATION RESULTS:")
        print(f"Method Name: {evaluation_metrics['method_name']}")
        print(f"Total Documents: {evaluation_metrics['total_documents']}")
        print(f"Total Distinct Locations: {evaluation_metrics['total_distinct_locations']}")
        print(f"Average Distinct Locations per Document: {evaluation_metrics['avg_distinct_locations']:.2f}")
        
        if evaluation_metrics['per_document_counts']:
            print(f"\nPer-Document Location Counts:")
            for i, count in enumerate(evaluation_metrics['per_document_counts'], 1):
                print(f"  Document {i}: {count} distinct locations")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()