import pandas as pd
import json
import lotus
from lotus.models import LM
import sys
from pathlib import Path
import time

from dotenv import load_dotenv
load_dotenv()

# Add the evaluation utils to the path
from experiments.reasoning.evaluation.utils import get_evaluate_func

# Configure Lotus with the model
lm = LM(model="azure/gpt-4.1-nano", max_tokens=10000)
lotus.settings.configure(lm=lm)

def load_blackvault_data(file_path):
    """Load BlackVault data from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def main():
    # Record start time
    start_time = time.time()
    
    datasets = [
        {"name": "train", "path": "experiments/reasoning/data/train/blackvault.json"},
        {"name": "test", "path": "experiments/reasoning/data/test/blackvault.json"}
    ]
    
    eval_results = []
    
    for dataset in datasets:
        lm.reset_stats()
        print(f"\n{'='*50}")
        print(f"Processing {dataset['name']} dataset...")
        print(f"{'='*50}")
        
        # Load the BlackVault dataset
        df = load_blackvault_data(dataset['path'])
        
        # Step 1: Extract event type using sem_map (following the pipeline)
        print(f"Step 1: Extracting event types from {len(df)} documents...")
        event_type_prompt = """Given the following document content from the black vault, identify and extract the type of extraterrestrial or paranormal event being described:

{all_content}

Provide a single, specific event type category."""
        
        df_with_events = df.sem_map(event_type_prompt, suffix="event_type")
        
        # Find the output column created by sem_map
        event_type_col = [col for col in df_with_events.columns if col.endswith("event_type")][0]
        
        # Step 2: Aggregate locations by event type using sem_agg (following the pipeline)
        print(f"Step 2: Aggregating locations by event type...")
        locations_prompt = """For extraterrestrial/paranormal events of type "{event_type}", analyze these reports:

{all_content}

Extract and list all unique locations of observation mentioned across these reports. They should be real locations on Earth. Return the locations as a JSON array of strings."""
        
        # Group by event type and aggregate locations
        df_aggregated = df_with_events.sem_agg(
            locations_prompt,
            group_by=[event_type_col],
            suffix="_locations_raw"
        )
        
        # Parse the JSON outputs from sem_agg
        print("Parsing location outputs...")
        locations_col = [col for col in df_aggregated.columns if col.endswith("_locations_raw")][0]
        
        def parse_locations_output(output_str):
            """Parse the locations output from the LLM response"""
            try:
                # Try to parse as JSON directly
                result = json.loads(output_str)
                if isinstance(result, list):
                    return result
                elif isinstance(result, str):
                    return [result]
                else:
                    return [str(result)]
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from markdown code blocks
                import re
                # Look for JSON within ```json ``` code blocks
                json_match = re.search(r'```json\s*(.*?)\s*```', output_str, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(1))
                        if isinstance(result, list):
                            return result
                        elif isinstance(result, str):
                            return [result]
                        else:
                            return [str(result)]
                    except json.JSONDecodeError:
                        pass
                
                # Fallback: try to extract any JSON array from the response
                json_match = re.search(r'\[.*\]', output_str, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                        return result if isinstance(result, list) else [str(result)]
                    except json.JSONDecodeError:
                        pass
                
                # If all else fails, try to split by common delimiters
                print(f"Failed to parse JSON, falling back to text parsing: {output_str}")
                # Split by common delimiters and clean up
                locations = []
                for delimiter in [',', ';', '\n', '|']:
                    if delimiter in output_str:
                        locations = [loc.strip() for loc in output_str.split(delimiter) if loc.strip()]
                        break
                
                return locations if locations else [output_str.strip()]
        
        df_aggregated['locations'] = df_aggregated[locations_col].apply(parse_locations_output)
        
        # Create final output format matching the pipeline schema
        print("Creating final output structure...")
        results_list = df_aggregated.to_dict('records')
        
        # Save results as JSON
        output_path = f"experiments/reasoning/othersystems/blackvault/lotus_{dataset['name']}.json"
        print(f"Saving results to {output_path}...")
        
        with open(output_path, 'w') as f:
            json.dump(results_list, f, indent=2)
        
        print(f"Processing complete! Results saved to {output_path}")
        print(f"Processed {len(results_list)} event types")
        
        # Record execution time before evaluation
        execution_time = time.time() - start_time
        
        # Run evaluation using the utils function
        print(f"\n🧪 Running BlackVault evaluation for {dataset['name']}...")
        try:
            eval_func = get_evaluate_func("blackvault")
            metrics = eval_func("lotus", output_path)
            
            cost = lm.stats.virtual_usage.total_cost
            
            print(f"\n📊 Evaluation Results for {dataset['name']}:")
            print(f"   Average Distinct Locations: {metrics['avg_distinct_locations']:.4f}")
            print(f"   Total Documents: {metrics['total_documents']}")
            print(f"   Total Distinct Locations: {metrics['total_distinct_locations']}")
            print(f"   Cost: {cost:.4f}")
            
            # Add evaluation results for this dataset
            eval_results.append({
                "file": output_path,
                "avg_distinct_locations": metrics['avg_distinct_locations'],
                "total_documents": metrics['total_documents'],
                "total_distinct_locations": metrics['total_distinct_locations'],
                "cost": cost,
                "execution_time": execution_time
            })
            
        except Exception as e:
            print(f"⚠️  Evaluation failed for {dataset['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save combined evaluation metrics
    if eval_results:
        eval_output_path = "experiments/reasoning/othersystems/blackvault/lotus_evaluation.json"
        with open(eval_output_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"\n📈 Combined evaluation metrics saved to: {eval_output_path}")
        print(f"📊 Total datasets processed: {len(eval_results)}")

if __name__ == "__main__":
    main()
    lotus.settings.lm.print_total_usage()
