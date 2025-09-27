#!/usr/bin/env python3
"""
Test script to demonstrate how to use the validate_pareto_frontier.py script.
Run this with: modal run test_validate_frontier.py
"""

from validate_pareto_frontier import main

if __name__ == "__main__":
    # Example usage:
    # Replace with your actual folder path and dataset name
    folder_path = "outputs/cuad_mcts"  # Path relative to Modal volume
    dataset = "cuad"
    
    print(f"Testing validation for {dataset} in folder: {folder_path}")
    main(folder_path, dataset)
