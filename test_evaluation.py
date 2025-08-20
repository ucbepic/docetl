#!/usr/bin/env python3
"""
Test script to debug why some JSON files are getting 0.0 evaluation scores.
"""

import sys
from pathlib import Path

# Add the current directory to path to import evaluation modules
sys.path.append(str(Path(__file__).parent))

from experiments.reasoning.evaluation.game_reviews import evaluate_results


def test_file(file_path):
    """Test evaluation of a specific file."""
    print(f"Testing evaluation of: {file_path}")
    
    try:
        metrics = evaluate_results("docetl", str(file_path))
        print(f"Results:")
        print(f"  combined_accuracy_score: {metrics['weighted_score']:.4f}")
        print(f"  sentiment_accuracy: {metrics['sentiment_accuracy']:.4f}")
        print(f"  kendall_tau_score: {metrics['kendall_tau_score']:.4f}")
        print(f"  valid_games_processed: {metrics['valid_games_processed']}")
        return True
    except Exception as e:
        print(f"Error evaluating file: {e}")
        return False


if __name__ == "__main__":
    # Test the files that are getting 0.0 scores
    base_dir = Path("/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/experiments/reasoning/outputs/game_reviews_mcts")
    
    zero_score_files = [
        "game_reviews_modal_4.json",
        "game_reviews_modal_4_1.json",
        "game_reviews_modal_4_2.json",
        "game_reviews_modal_4_2_1.json",
        "game_reviews_modal_4_1_1.json",
        "game_reviews_modal_4_2_2.json"
    ]
    
    for file_name in zero_score_files:
        file_path = base_dir / file_name
        if file_path.exists():
            print(f"\n{'='*50}")
            test_file(file_path)
        else:
            print(f"\n{'='*50}")
            print(f"File does not exist: {file_path}")