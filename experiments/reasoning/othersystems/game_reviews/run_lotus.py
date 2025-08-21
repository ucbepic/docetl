import pandas as pd
import json
import lotus
from lotus.models import LM
import sys
from pathlib import Path
import re

from dotenv import load_dotenv
load_dotenv()

# Add the evaluation utils to the path
from experiments.reasoning.evaluation.utils import get_evaluate_func

# Configure Lotus with the model
lm = LM(model="azure/gpt-4.1-nano", max_tokens=10000)
lotus.settings.configure(lm=lm)

def load_game_reviews_data(file_path):
    """Load game reviews data from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def parse_reviews_output(output_str):
    """Parse the output from the LLM response for game reviews format"""
    try:
        # Try to parse as JSON directly
        result = json.loads(output_str)
        if isinstance(result, dict) and 'positive_reviews' in result and 'negative_reviews' in result:
            return result
    except json.JSONDecodeError:
        pass
    
    # If direct parsing fails, try to extract JSON from markdown code blocks
    json_match = re.search(r'```json\s*(.*?)\s*```', output_str, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(1))
            if isinstance(result, dict) and 'positive_reviews' in result and 'negative_reviews' in result:
                return result
        except json.JSONDecodeError:
            pass
    
    # Fallback: try to extract any JSON object from the response
    json_match = re.search(r'\{.*\}', output_str, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            if isinstance(result, dict) and 'positive_reviews' in result and 'negative_reviews' in result:
                return result
        except json.JSONDecodeError:
            pass
    
    # If all else fails, return empty structure
    print(f"Failed to parse reviews output: {output_str}")
    return {
        "positive_reviews": [],
        "negative_reviews": []
    }

def main():
    datasets = [
        {"name": "train", "path": "experiments/reasoning/data/train/game_reviews.json"},
        {"name": "test", "path": "experiments/reasoning/data/test/game_reviews.json"}
    ]
    
    eval_results = []
    
    for dataset in datasets:
        lm.reset_stats()
        print(f"\n{'='*50}")
        print(f"Processing {dataset['name']} dataset...")
        print(f"{'='*50}")
        
        # Load the game reviews dataset
        df = load_game_reviews_data(dataset['path'])
        
        # Use the exact same prompt as in the YAML pipeline
        user_instruction = """Given the following reviews for the game {app_name}, analyze them and select 10 positive and 10 negative reviews that are evenly distributed across time:

{concatenated_reviews}

From all the reviews:
1. Select 10 positive reviews spread evenly across the time range (from earliest to latest timestamps)
2. Select 10 negative reviews spread evenly across the time range (from earliest to latest timestamps)

Return two lists:
- positive_reviews: List of 10 positive reviews, sorted by timestamp
- negative_reviews: List of 10 negative reviews, sorted by timestamp

Each returned review object should contain the review ID, timestamp and a summary of the review.

Please return your response as a JSON object with this structure:
{{
  "positive_reviews": [
    {{"review_id": "string", "timestamp": "string", "review_summary": "string"}},
    ...
  ],
  "negative_reviews": [
    {{"review_id": "string", "timestamp": "string", "review_summary": "string"}},
    ...
  ]
}}"""

        # Apply the semantic map operation
        print(f"Processing {len(df)} games with Lotus...")
        print(f"Avg num chars in concatenated_reviews: {df['concatenated_reviews'].apply(len).mean()}")
        df_result = df.sem_map(user_instruction, safe_mode=True)
        
        # Find the output column created by sem_map
        output_col = [col for col in df_result.columns if col not in df.columns][0]
        
        # Parse the outputs in the new column
        print("Parsing game reviews outputs...")
        parsed_outputs = df_result[output_col].apply(parse_reviews_output)
        
        # Extract the parsed fields directly into the dataframe
        df_result['positive_reviews'] = parsed_outputs.apply(lambda x: x['positive_reviews'])
        df_result['negative_reviews'] = parsed_outputs.apply(lambda x: x['negative_reviews'])
        
        # Drop the original output column since we've extracted the fields
        df_result = df_result.drop(columns=[output_col])
        
        # Save results as JSON
        output_path = f"experiments/reasoning/othersystems/game_reviews/lotus_{dataset['name']}.json"
        print(f"Saving results to {output_path}...")
        
        # Convert DataFrame to list of dictionaries and save
        results_list = df_result.to_dict('records')
        # Drop the concatenated reviews column
        df_result = df_result.drop(columns=['concatenated_reviews'])
        
        with open(output_path, 'w') as f:
            json.dump(results_list, f, indent=2)
        
        print(f"Processing complete! Results saved to {output_path}")
        print(f"Processed {len(results_list)} games")
        
        # Run evaluation using the utils function
        print(f"\n🧪 Running game reviews evaluation for {dataset['name']}...")
        try:
            eval_func = get_evaluate_func("game_reviews")
            metrics = eval_func("lotus", output_path)
            
            cost = lm.stats.virtual_usage.total_cost
            
            print(f"\n📊 Evaluation Results for {dataset['name']}:")
            print(f"   Sentiment Accuracy: {metrics['sentiment_accuracy']:.4f}")
            print(f"   Kendall Tau Score: {metrics['kendall_tau_score']:.4f}")
            print(f"   Weighted Score: {metrics['weighted_score']:.4f}")
            print(f"   Valid Games Processed: {metrics['valid_games_processed']}")
            print(f"   Cost: {cost:.4f}")
            
            # Add evaluation results for this dataset
            eval_results.append({
                "file": output_path,
                "sentiment_accuracy": metrics['sentiment_accuracy'],
                "kendall_tau_score": metrics['kendall_tau_score'],
                "weighted_score": metrics['weighted_score'],
                "combined_accuracy_score": metrics['weighted_score'],  # Used by utils.py
                "valid_games_processed": metrics['valid_games_processed'],
                "cost": cost
            })
            
        except Exception as e:
            print(f"⚠️  Evaluation failed for {dataset['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save combined evaluation metrics
    if eval_results:
        eval_output_path = "experiments/reasoning/othersystems/game_reviews/lotus_evaluation.json"
        with open(eval_output_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"\n📈 Combined evaluation metrics saved to: {eval_output_path}")
        print(f"📊 Total datasets processed: {len(eval_results)}")

if __name__ == "__main__":
    main()
    lotus.settings.lm.print_total_usage()
