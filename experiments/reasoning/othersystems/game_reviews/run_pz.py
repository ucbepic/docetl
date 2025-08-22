import argparse
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
import re

import palimpzest as pz
from palimpzest.constants import Model
from palimpzest.core.lib.fields import ListField, StringField
from palimpzest.policy import MaxQuality, MaxQualityAtFixedCost

from dotenv import load_dotenv
load_dotenv()

# Add the evaluation utils to the path
from experiments.reasoning.evaluation.utils import get_evaluate_func

# Budget fractions to test
FRACS = [0.75, 0.5, 0.25, 0.1]


def load_game_reviews_data(split="train"):
    """Load game reviews data from our existing JSON files."""
    file_path = f"experiments/reasoning/data/{split}/game_reviews.json"
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


class GameReviewsDataReader(pz.DataReader):
    def __init__(self, num_games: int = 100, split: str = "train", seed: int = 42):
        self.num_games = num_games
        self.split = split
        self.seed = seed

        input_cols = [
            {"name": "app_name", "type": str, "desc": "The name of the game"},
            {"name": "concatenated_reviews", "type": str, "desc": "All reviews for the game concatenated together"},
        ]
        super().__init__(input_cols)
        
        # Load the dataset
        data = load_game_reviews_data(split)
        
        # Sample games if needed
        if len(data) > num_games:
            np.random.seed(seed)
            indices = np.random.choice(len(data), num_games, replace=False)
            data = [data[i] for i in indices]
        
        self.dataset = data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        return {
            "app_name": item["app_name"],
            "concatenated_reviews": item["concatenated_reviews"]
        }
    
    def get_label_df(self):
        """Get a dataframe with labels for evaluation."""
        final_label_dataset = []
        for entry in self.dataset:
            row = {
                "app_name": entry["app_name"],
                "concatenated_reviews": entry["concatenated_reviews"],
                "positive_reviews": entry.get("positive_reviews", []),
                "negative_reviews": entry.get("negative_reviews", [])
            }
            final_label_dataset.append(row)
        
        return pd.DataFrame(final_label_dataset)


def build_game_reviews_query(dataset):
    """Build the game reviews query for sentiment analysis and review selection."""
    ds = pz.Dataset(dataset)
    
    # Define the game reviews analysis schema
    cols = [
        {
            "name": "positive_reviews", 
            "type": list[str], 
            "desc": "List of 10 positive reviews spread evenly across time, each with review_id, timestamp, and review_summary"
        },
        {
            "name": "negative_reviews", 
            "type": list[str], 
            "desc": "List of 10 negative reviews spread evenly across time, each with review_id, timestamp, and review_summary"
        }
    ]
    
    desc = """Given the following reviews for the game, analyze them and select 10 positive and 10 negative reviews that are evenly distributed across time:

From all the reviews:
1. Select 10 positive reviews spread evenly across the time range (from earliest to latest timestamps)
2. Select 10 negative reviews spread evenly across the time range (from earliest to latest timestamps)

Return two lists:
- positive_reviews: List of 10 positive reviews, sorted by timestamp
- negative_reviews: List of 10 negative reviews, sorted by timestamp

Each returned review object should contain the review ID, timestamp and a summary of the review.

The format should be:
positive_reviews = [{"review_id": "string", "timestamp": "string", "review_summary": "string"}, ...]
negative_reviews = [{"review_id": "string", "timestamp": "string", "review_summary": "string"}, ...]
"""
    
    ds = ds.sem_add_columns(cols, desc=desc, depends_on=["app_name", "concatenated_reviews"])
    
    return ds


def parse_reviews_output(output_data):
    """Parse the output from the LLM response for game reviews format"""
    if isinstance(output_data, str):
        try:
            # Try to parse as JSON directly
            result = json.loads(output_data)
            if isinstance(result, dict) and 'positive_reviews' in result and 'negative_reviews' in result:
                return result
        except json.JSONDecodeError:
            pass
        
        # If direct parsing fails, try to extract JSON from markdown code blocks
        json_match = re.search(r'```json\s*(.*?)\s*```', output_data, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                if isinstance(result, dict) and 'positive_reviews' in result and 'negative_reviews' in result:
                    return result
            except json.JSONDecodeError:
                pass
        
        # Fallback: try to extract any JSON object from the response
        json_match = re.search(r'\{.*\}', output_data, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                if isinstance(result, dict) and 'positive_reviews' in result and 'negative_reviews' in result:
                    return result
            except json.JSONDecodeError:
                pass
        
        # If all else fails, return empty structure
        print(f"Failed to parse reviews output: {output_data}")
        return {
            "positive_reviews": [],
            "negative_reviews": []
        }
    elif isinstance(output_data, dict):
        # Already parsed
        return output_data
    else:
        # Unknown format
        return {
            "positive_reviews": [],
            "negative_reviews": []
        }


def convert_predictions_to_game_reviews_format(pred_df, data_reader):
    """Convert Palimpzest predictions to the format expected by game reviews evaluation."""
    results = []
    
    for _, row in pred_df.iterrows():
        # Handle prediction values - they might be strings that need JSON parsing
        positive_reviews_raw = row.get("positive_reviews", [])
        negative_reviews_raw = row.get("negative_reviews", [])
        
        # Parse if they're strings
        if isinstance(positive_reviews_raw, str):
            parsed_data = parse_reviews_output(positive_reviews_raw)
            positive_reviews = parsed_data.get("positive_reviews", [])
            negative_reviews = parsed_data.get("negative_reviews", [])
        elif isinstance(negative_reviews_raw, str):
            parsed_data = parse_reviews_output(negative_reviews_raw)
            positive_reviews = parsed_data.get("positive_reviews", [])
            negative_reviews = parsed_data.get("negative_reviews", [])
        else:
            # Already parsed or empty
            positive_reviews = positive_reviews_raw if isinstance(positive_reviews_raw, list) else []
            negative_reviews = negative_reviews_raw if isinstance(negative_reviews_raw, list) else []
        
        # Clean up the values
        if positive_reviews is None or not isinstance(positive_reviews, list):
            positive_reviews = []
        if negative_reviews is None or not isinstance(negative_reviews, list):
            negative_reviews = []
        
        # Create the record with just predictions
        record = {
            "app_name": row["app_name"],
            "concatenated_reviews": row["concatenated_reviews"],
            "positive_reviews": positive_reviews,
            "negative_reviews": negative_reviews
        }
        results.append(record)
    
    return results


def run_experiment(data_reader, val_data_reader, policy, models, 
                   sentinel_strategy="mab", k=10, j=3, sample_budget=100, seed=42, exp_name=None):
    """Run a single experiment with given policy and return results."""
    print(f"\nRunning experiment with policy: {policy}")
    
    # Build query
    query = build_game_reviews_query(data_reader)
    
    # Configure query processor
    config = pz.QueryProcessorConfig(
        policy=policy,
        verbose=False,
        val_datasource=val_data_reader,
        processing_strategy="sentinel",
        optimizer_strategy="pareto",
        sentinel_execution_strategy=sentinel_strategy,
        execution_strategy="parallel",
        max_workers=64,
        available_models=models,
        allow_bonded_query=True,
        allow_code_synth=False,
        allow_critic=True,
        allow_mixtures=True,
        allow_rag_reduction=True,
        progress=True,
    )
    
    # Execute the query
    data_record_collection = query.run(
        config=config,
        k=k,
        j=j,
        sample_budget=sample_budget,
        seed=seed,
        exp_name=exp_name if exp_name else f"game-reviews-pz-{policy.__class__.__name__}",
        priors=None,
    )
    
    pred_df = data_record_collection.to_df()
    
    # Convert to game reviews format
    results_list = convert_predictions_to_game_reviews_format(pred_df, data_reader)
    
    # Save results as JSON
    output_file = f"experiments/reasoning/othersystems/game_reviews/{exp_name}.json" if exp_name else "experiments/reasoning/othersystems/game_reviews/pz_temp.json"
    with open(output_file, 'w') as f:
        json.dump(results_list, f, indent=2)
    
    # Evaluate using our existing evaluation framework
    evaluate_func = get_evaluate_func("game_reviews")
    metrics = evaluate_func("palimpzest", output_file)
    
    # Get execution statistics for cost
    exec_stats = data_record_collection.execution_stats
    
    return {
        "sentiment_accuracy": metrics["sentiment_accuracy"],
        "kendall_tau_score": metrics["kendall_tau_score"],
        "weighted_score": metrics["weighted_score"],
        "combined_accuracy_score": metrics["weighted_score"],
        "valid_games_processed": metrics["valid_games_processed"],
        "optimization_time": exec_stats.optimization_time if exec_stats else 0,
        "optimization_cost": exec_stats.optimization_cost if exec_stats else 0,
        "plan_execution_time": exec_stats.plan_execution_time if exec_stats else 0,
        "plan_execution_cost": exec_stats.plan_execution_cost if exec_stats else 0,
        "total_execution_time": exec_stats.total_execution_time if exec_stats else 0,
        "total_execution_cost": exec_stats.total_execution_cost if exec_stats else 0,
        "output_file": output_file,
        "sentinel_strategy": sentinel_strategy,
        "k": k,
        "j": j,
        "sample_budget": sample_budget,
    }


def main():
    parser = argparse.ArgumentParser(description="Run game reviews experiments with budget analysis using Palimpzest")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-games", type=int, default=100, help="Number of games to process")
    parser.add_argument(
        "--sentinel-execution-strategy",
        default="mab",
        type=str,
        help="The engine to use. One of mab or random",
    )
    parser.add_argument(
        "--k",
        default=3,
        type=int,
        help="Number of columns to sample in Random Sampling or MAB sentinel execution",
    )
    parser.add_argument(
        "--j",
        default=4,
        type=int,
        help="Number of rows to sample in Random Sampling or MAB sentinel execution",
    )
    parser.add_argument(
        "--sample-budget",
        default=50,
        type=int,
        help="Total sample budget in Random Sampling or MAB sentinel execution",
    )
    parser.add_argument(
        "--exp-name",
        default=None,
        type=str,
        help="The experiment name prefix.",
    )
    
    args = parser.parse_args()
    
    if os.getenv("OPENAI_API_KEY") is None:
        print("ERROR: OPENAI_API_KEY is not set")
        return
    
    # Set models - use all
    models = [
        Model.GPT_41_MINI,
        Model.GPT_41,
        Model.GPT_41_NANO,
        Model.GEMINI_25_FLASH,
        Model.GEMINI_25_FLASH_LITE,
        Model.GEMINI_25_PRO,
    ]
    
    print(f"Loading game reviews dataset...")
    
    # Create data readers: test set for main evaluation, train set for validation
    data_reader = GameReviewsDataReader(split="test", num_games=args.num_games, seed=args.seed)
    val_data_reader = GameReviewsDataReader(split="train", num_games=50, seed=args.seed)
    
    print(f"Processing {len(data_reader)} test games with Palimpzest...")
    print(f"Using {len(val_data_reader)} train games for validation...")
    
    results = {}
    
    # Step 1: Run unconstrained max quality
    print("\n=== Step 1: Running unconstrained max quality ===")
    policy = MaxQuality()
    exp_name_unconstrained = f"{args.exp_name}-unconstrained" if args.exp_name else "pz-unconstrained"
    unconstrained_result = run_experiment(
        data_reader, val_data_reader, policy, models,
        sentinel_strategy=args.sentinel_execution_strategy,
        k=args.k, j=args.j, sample_budget=args.sample_budget,
        seed=args.seed, exp_name=exp_name_unconstrained
    )
    results["unconstrained_max_quality"] = unconstrained_result
    unconstrained_cost = unconstrained_result["plan_execution_cost"]
    print(f"Unconstrained cost: ${unconstrained_cost:.4f}")
    print(f"Unconstrained Weighted Score: {unconstrained_result['weighted_score']:.4f}")
    
    # Step 2: Run at each budget fraction
    budget_targets = {}
    for i, frac in enumerate(FRACS, 2):
        budget = unconstrained_cost * frac
        budget_targets[f"budget_{int(frac*100)}_percent"] = budget
        
        print(f"\n=== Step {i}: Running max quality at {int(frac*100)}% budget (${budget:.4f}) ===")
        policy = MaxQualityAtFixedCost(max_cost=budget)
        exp_name_budget = f"{args.exp_name}-{int(frac*100)}pct" if args.exp_name else f"pz-{int(frac*100)}pct"
        
        budget_result = run_experiment(
            data_reader, val_data_reader, policy, models,
            sentinel_strategy=args.sentinel_execution_strategy,
            k=args.k, j=args.j, sample_budget=args.sample_budget,
            seed=args.seed, exp_name=exp_name_budget
        )
        results[f"budget_{int(frac*100)}_percent"] = budget_result
        print(f"{int(frac*100)}% budget cost: ${budget_result['plan_execution_cost']:.4f}")
        print(f"{int(frac*100)}% budget Weighted Score: {budget_result['weighted_score']:.4f}")
    
    # Add metadata
    results["metadata"] = {
        "seed": args.seed,
        "num_games": args.num_games,
        "unconstrained_cost": unconstrained_cost,
        "budget_fractions": FRACS,
        "budget_targets": budget_targets,
        "sentinel_execution_strategy": args.sentinel_execution_strategy,
        "k": args.k,
        "j": args.j,
        "sample_budget": args.sample_budget,
        "exp_name": args.exp_name,
        "system": "palimpzest",
    }
    
    # Save combined results
    output_path = "experiments/reasoning/othersystems/game_reviews/pz_evaluation.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📈 Combined results saved to: {output_path}")
    
    # Print summary
    print("\n=== Summary ===")
    print(f"{'Configuration':<30} {'Cost ($)':<12} {'Sentiment':<12} {'Kendall':<12} {'Weighted':<12} {'Valid Games':<12}")
    print(f"{'':30} {'':12} {'Accuracy':<12} {'Tau':<12} {'Score':<12} {'Processed':<12}")
    print("-" * 90)
    
    # Print unconstrained result
    r = results["unconstrained_max_quality"]
    print(f"{'Unconstrained':<30} {r['total_execution_cost']:<12.4f} {r['sentiment_accuracy']:<12.4f} {r['kendall_tau_score']:<12.4f} {r['weighted_score']:<12.4f} {r['valid_games_processed']:<12}")
    
    # Print budget results
    for frac in FRACS:
        key = f"budget_{int(frac*100)}_percent"
        label = f"{int(frac*100)}% Budget"
        r = results[key]
        print(f"{label:<30} {r['total_execution_cost']:<12.4f} {r['sentiment_accuracy']:<12.4f} {r['kendall_tau_score']:<12.4f} {r['weighted_score']:<12.4f} {r['valid_games_processed']:<12}")


if __name__ == "__main__":
    main()
