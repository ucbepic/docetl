import pandas as pd
import json
import lotus
from lotus.models import LM
from lotus.types import CascadeArgs
import sys
from pathlib import Path
import time
import re

from dotenv import load_dotenv
load_dotenv()

# Add the evaluation utils to the path
from experiments.reasoning.evaluation.utils import get_evaluate_func

# Configure Lotus with the model and helper model for cascade
gpt_41_nano = LM(model="azure/gpt-4.1-nano", max_tokens=10000)
gpt_41_mini = LM(model="azure/gpt-4.1-mini", max_tokens=10000)

lotus.settings.configure(lm=gpt_41_mini, helper_lm=gpt_41_nano)

def load_sustainability_data(file_path):
    """Load sustainability data from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def parse_json_output(output_str):
    """Parse the JSON output from the LLM response"""
    try:
        # Try to parse as JSON directly
        return json.loads(output_str)
    except json.JSONDecodeError:
        # If direct parsing fails, try to extract JSON from markdown code blocks
        # Look for JSON within ```json ``` code blocks
        json_match = re.search(r'```json\s*(.*?)\s*```', output_str, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Fallback: try to extract any JSON array from the response
        json_match = re.search(r'\[.*\]', output_str, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # If all else fails, return a default structure
        print(f"Failed to parse JSON: {output_str}")
        return [{"company_name": "Unknown", "key_findings": output_str.strip()}]

def main():
    start_time = time.time()

    datasets = [
        {"name": "train", "path": "experiments/reasoning/data/train/sustainability.json"},
        {"name": "test", "path": "experiments/reasoning/data/test/sustainability.json"}
    ]
    
    eval_results = []
    
    for dataset in datasets:
        gpt_41_mini.reset_stats()
        gpt_41_nano.reset_stats()
        print(f"\n{'='*50}")
        print(f"Processing {dataset['name']} dataset...")
        print(f"{'='*50}")
        
        # Load the sustainability dataset
        df = load_sustainability_data(dataset['path'])
        print(f"Loaded {len(df)} documents")
        
        # Print avg # chars in tot_text_raw
        print(f"Avg # chars in tot_text_raw: {df['tot_text_raw'].apply(len).mean()}")
        
        # Step 1: Filter sustainability reports using sem_filter with cascade args
        print("Step 1: Filtering sustainability reports with cascade...")
        filter_prompt = """Analyze the following document and determine if it is a sustainability report or contains significant sustainability-related content.
        
        Document text:
        {tot_text_raw}

A sustainability report typically includes information about:
- Environmental performance and initiatives
- Social responsibility and community impact
- Corporate governance and sustainability practices
- ESG (Environmental, Social, Governance) metrics
- Climate change initiatives
- Sustainable business practices
- Corporate social responsibility activities

Return true if this document is primarily a sustainability report or contains substantial sustainability-related content, false otherwise."""
        
        # Use simple filtering without cascade to avoid Lotus library bug
        try:
            df_filtered = df.sem_filter(
                user_instruction=filter_prompt, 
                suffix="_is_sustainability"
            )
            filter_stats = {"total_docs": len(df), "filtered_docs": len(df_filtered)}
        except Exception as e:
            print(f"⚠️  Filtering failed: {e}")
            print("Continuing with original dataset...")
            df_filtered = df
            filter_stats = {"total_docs": len(df), "filtered_docs": len(df), "error": str(e)}
        print(f"After filtering: {len(df_filtered)} documents remain")
        print(f"Filter cascade stats: {filter_stats}")
        
        # Step 2: Determine economic activity using sem_map
        print("Step 2: Determining economic activity...")
        economic_activity_prompt = """Analyze the following sustainability-related document and determine the primary economic sector/activity of the company.
        
        Document text:
        {tot_text_raw}

Determine the primary economic sector/activity of the company. Choose from these exact categories:
- manufacturing
- other  
- information_communication
- finance_insurance
- agriculture
- energy_supply
- health_social
- trade_repair
- real_estate
- professional_scientific
- transportation
- construction
- accommodation_food

Return only the economic activity category."""
        
        try:
            df_with_activity = df_filtered.sem_map(economic_activity_prompt, suffix="economic_activity")
        except Exception as e:
            print(f"⚠️  Economic activity mapping failed: {e}")
            print("Skipping economic activity step...")
            df_with_activity = df_filtered
        
        # Step 3: Sustainability summary using sem_agg with group_by on raw economic activity
        print("Step 3: Creating sustainability summaries by economic activity...")
        summary_prompt = """You are analyzing sustainability reports from companies in the {economic_activity} sector.

Create a comprehensive summary for the {economic_activity} sector, given the article {tot_text_raw}. Extract the company name from each document and provide key sustainability findings for each company.

For each company, provide:
- Company name (extracted from the document)
- Key findings: A concise summary of their key sustainability initiatives, achievements, or commitments (2-3 sentences max)

Return the results as a JSON array of objects, where each object has "company_name" and "key_findings" fields."""
        
        try:
            df_summary = df_with_activity.sem_agg(
                summary_prompt,
                group_by=["economic_activity"],
                suffix="_sustainability_summary_raw"
            )
        except Exception as e:
            print(f"⚠️  Sustainability summary aggregation failed: {e}")
            print("Creating fallback summary...")
            # Create a fallback structure
            df_summary = df_with_activity.copy()
            df_summary['_sustainability_summary_raw'] = '{"error": "API connection failed"}'
        
        # Parse the JSON outputs from sem_agg
        print("Parsing sustainability summary JSON outputs...")
        df_summary['economic_activity_summary'] = df_summary['_sustainability_summary_raw'].apply(parse_json_output)
        
        # Create the final output structure
        print("Creating final output structure...")
        results_list = df_summary.to_dict(orient='records')
        
        # Save results as JSON
        output_path = f"experiments/reasoning/othersystems/sustainability/lotus_{dataset['name']}.json"
        print(f"Saving results to {output_path}...")
        
        with open(output_path, 'w') as f:
            json.dump(results_list, f, indent=2)
        
        print(f"Processing complete! Results saved to {output_path}")
        print(f"Processed {len(results_list)} documents")
        
        # Record execution time before evaluation
        execution_time = time.time() - start_time
        
        # Run evaluation using the utils function
        print(f"\n🧪 Running sustainability evaluation for {dataset['name']}...")
        try:
            eval_func = get_evaluate_func("sustainability")
            metrics = eval_func("lotus", output_path)
            
            # Calculate total cost from both models
            total_cost = gpt_41_mini.stats.virtual_usage.total_cost + gpt_41_nano.stats.virtual_usage.total_cost
            
            print(f"\n📊 Evaluation Results for {dataset['name']}:")
            print(f"   Economic Activity Accuracy: {metrics['economic_activity_accuracy']:.4f}")
            print(f"   Company Name Accuracy: {metrics['company_name_accuracy']:.4f}")
            print(f"   Combined Score: {metrics['combined_score']:.4f}")
            print(f"   Total Companies Processed: {metrics['total_companies_processed']}")
            print(f"   Total Cost (GPT-4.1 + GPT-4.1-mini): {total_cost:.4f}")
            print(f"   GPT-4.1 Cost: {gpt_41_mini.stats.virtual_usage.total_cost:.4f}")
            print(f"   GPT-4.1-mini Cost: {gpt_41_nano.stats.virtual_usage.total_cost:.4f}")
            
            # Add evaluation results for this dataset
            eval_results.append({
                "file": output_path,
                "economic_activity_accuracy": metrics['economic_activity_accuracy'],
                "combined_score": metrics['combined_score'],
                "company_name_accuracy": metrics['company_name_accuracy'],
                "total_companies_processed": metrics['total_companies_processed'],
                "missing_companies": metrics['missing_companies'],
                "avg_findings_length": metrics['avg_findings_length'],
                "total_cost": total_cost,
                "gpt4o_cost": gpt_41_mini.stats.virtual_usage.total_cost,
                "gpt4o_mini_cost": gpt_41_nano.stats.virtual_usage.total_cost,
                "filter_stats": filter_stats,
                "execution_time": execution_time
            })
            
        except Exception as e:
            print(f"⚠️  Evaluation failed for {dataset['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save combined evaluation metrics
    if eval_results:
        eval_output_path = "experiments/reasoning/othersystems/sustainability/lotus_evaluation.json"
        with open(eval_output_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"\n📈 Combined evaluation metrics saved to: {eval_output_path}")
        print(f"📊 Total datasets processed: {len(eval_results)}")

if __name__ == "__main__":
    main()
    print("\n🔍 Final Usage Statistics:")
    print("GPT-4.1 Usage:")
    gpt_41_mini.print_total_usage()
    print("\nGPT-4.1-mini Usage:")
    gpt_41_nano.print_total_usage()
