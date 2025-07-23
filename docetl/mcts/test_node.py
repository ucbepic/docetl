#!/usr/bin/env python3
"""
Simple test script for the Node class.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .Node import Node


def test_node_basic():
    """Test basic Node functionality."""
    # Use the existing YAML file
    yaml_path = (
        "/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/CUAD-map.yaml"
    )

    # Create root node
    root = Node(yaml_path)
    print(f"Root node: {root}")
    print(f"Operations: {len(root.get_operations())}")
    print(f"Datasets: {list(root.get_datasets().keys())}")

    # Test adding a child
    child = root.add_child(yaml_path)  # Using same file for testing
    print(f"Child node: {child}")
    print(f"Root children count: {len(root.children)}")

    # Test best_child (should return the child since it has infinite UCB)
    best = root.best_child()
    print(f"Best child: {best}")

    # Test value updates
    root.update_value(1.0)
    child.update_value(0.5)
    print(f"After updates - Root: {root}")
    print(f"After updates - Child: {child}")

    # Test UCB calculation
    print(f"Root UCB: {root.get_ucb()}")
    print(f"Child UCB: {child.get_ucb()}")

    # Test tree properties
    print(f"Root is root: {root.is_root()}")
    print(f"Child is root: {child.is_root()}")
    print(f"Root is leaf: {root.is_leaf()}")
    print(f"Child is leaf: {child.is_leaf()}")
    print(f"Root depth: {root.get_depth()}")
    print(f"Child depth: {child.get_depth()}")

    # Test path to root
    path = child.get_path_to_root()
    print(f"Path from child to root: {[node.yaml_file_path for node in path]}")

    # Test execute_plan (commented out to avoid actual execution during testing)
    print(f"\n=== Execute Plan Test ===")
    print(f"Node cost before execution: {root.cost}")
    print(
        "Note: execute_plan() is commented out to avoid actual pipeline execution during testing"
    )
    # Uncomment the following lines to test actual execution:
    # try:
    #     cost = root.execute_plan()
    #     print(f"Pipeline execution cost: ${cost:.2f}")
    #     print(f"Node cost after execution: {root.cost}")
    # except Exception as e:
    #     print(f"Execution failed: {e}")


if __name__ == "__main__":
    test_node_basic()
