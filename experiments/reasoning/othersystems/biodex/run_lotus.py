import pandas as pd
import json
import lotus
from lotus.models import LM, LiteLLMRM
from lotus.vector_store import FaissVS
from lotus.types import CascadeArgs
import time
import ast
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# NOTE: This code is provided by the LOTUS team; and we only make minimal modifications to fit in our framework.

# Add the evaluation utils to the path
from experiments.reasoning.evaluation.utils import get_evaluate_func

def load_biodex_data(file_path):
    """Load BioDEX data from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def load_biodex_labels(labels_path):
    """Load BioDEX reaction labels from text file"""
    with open(labels_path, 'r') as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    return pd.DataFrame({"reaction": labels})

def to_comma_separated(val):
    """
    Safely convert val (which could be a list or a string representing a list)
    into a comma-separated string.
    """
    if isinstance(val, list):
        return ", ".join(val)
    elif isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return ", ".join(parsed)
            else:
                return val
        except (SyntaxError, ValueError):
            return val
    else:
        return str(val)

def remove_known_prefixes(text: str, prefixes: list) -> str:
    """
    Removes the first matching prefix from 'text' if found in 'prefixes',
    otherwise returns text unchanged.
    """
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix):]
    return text

class BioDEXPipeline:
    def __init__(self, truncation_limit=8000, model="azure/gpt-4o-mini"):
        self.truncation_limit = truncation_limit
        self.model = model
        print(f"Using truncation limit of: {self.truncation_limit}")
        print(f"Using model: {self.model}")
        
        # Configure Lotus models following the original code
        rm = LiteLLMRM(model="text-embedding-3-small", max_batch_size=1000, truncate_limit=8000)
        lm = LM(model=self.model, max_batch_size=64, temperature=0.0, max_tokens=256)
        vs = FaissVS()
        
        lotus.settings.configure(lm=lm, rm=rm, vs=vs)
        
        print(f"lotus.settings.lm.max_batch_size = {lotus.settings.lm.max_batch_size}")
        print(f"lotus.settings.lm.max_tokens = {lotus.settings.lm.max_tokens}")
        print(f"lotus.settings.lm.temperature = {lotus.settings.lm.kwargs['temperature']}")

    def prepare_queries_data(self, df):
        """Prepare the queries dataframe similar to the original code"""
        # Truncate the fulltext to truncation_limit chars
        df["patient_description"] = df["fulltext_processed"].apply(lambda x: x[:self.truncation_limit])
        
        # Convert ground_truth_reactions to string format for processing
        df["reactions"] = df["ground_truth_reactions"].apply(to_comma_separated)
        df["reactions_list"] = df["ground_truth_reactions"].copy()
        df["num_labels"] = df["reactions_list"].apply(lambda x: len(x))
        
        # Add other fields needed for compatibility
        df["title"] = ""  # Not available in our data
        df["abstract"] = ""  # Not available in our data
        
        return df

    def run_join_pipeline(self, queries_df, corpus_df, recall_target=0.95, precision_target=0.95, name="test"):
        """Run the join pipeline similar to the original Lotus code"""
        start_t = time.time()

        map_instruction = "given the {patient_description} of a medical article, identify the adverse drug reactions that are likely affecting the patient. Always write your answer as a list of 2-10 comma-separated adverse drug reactions."
        join_instruction = """Can the following condition be found in the following medical article?
        Medical article: {patient_description} 
        
        Condition we are looking for: {reaction}

        Determine if {reaction} is described in the medical article, considering the context and meaning beyond just the presence of individual words."""
        
        print(f"corpus_df = {corpus_df.shape}")

        cascade_args = CascadeArgs(
            recall_target=recall_target,
            precision_target=precision_target,
            failure_probability=0.2,
            sampling_percentage=0.00008218,  # len = 500 samples as in original
            map_instruction=map_instruction,
            cascade_IS_random_seed=42,  # fixed random seed for sampling
        )
        
        print(f"Running sem_join with cascade args...")
        answers_df, stats = queries_df.sem_join(corpus_df, join_instruction, cascade_args=cascade_args, return_stats=True)

        end_t = time.time()
        
        print(f"Time taken: {end_t - start_t}")
        print(f"stats = {stats}")

        # Save intermediate results
        answers_df.to_csv(f"experiments/reasoning/othersystems/biodex/biodex_cascade_answers_for_lm_rerank_{name}.csv", index=True)
        
        # Post process for checking answers
        answers_df["reactions_list"] = answers_df["reactions_list"].apply(to_comma_separated)
        grouped_df = (
            answers_df.groupby(["title", "abstract", "reactions", "reactions_list", "patient_description"], dropna=False)
            .apply(lambda x: x["reaction"].tolist())
            .reset_index(name="pred_reaction")
        )
        # Convert reaction list from string back to list
        grouped_df["reactions_list"] = grouped_df["reactions_list"].apply(lambda x: x.split(", "))

        return grouped_df, (end_t - start_t), stats

    def run_rerank_pipeline(self, name="test"):
        """Run the reranking pipeline similar to the original Lotus code"""
        answers_df = pd.read_csv(f"experiments/reasoning/othersystems/biodex/biodex_cascade_answers_for_lm_rerank_{name}.csv", index_col=0)

        start_t = time.time()

        # Normalize reactions_list so every row is a comma-separated string
        answers_df["reactions_list"] = answers_df["reactions_list"].apply(to_comma_separated)

        # Group by and aggregate
        grouped_df = (
            answers_df
            .groupby(["title", "abstract", "reactions", "reactions_list", "patient_description"], dropna=False)
            .apply(lambda grp: grp["reaction"].tolist())
            .reset_index(name="pred_reaction")
        )

        # Convert that comma-separated string back to a list
        grouped_df["reactions_list"] = grouped_df["reactions_list"].apply(lambda s: s.split(", "))
        
        rerank_prompt = (
            "Given the following {patient_description} of a medical article, rank the predicted drug reactions {pred_reaction} in order of most confident to least confident that the medical article is describing the drug reaction\n"
            "\n\n"
            "There may be conditions described in the medical article that are not in the list of predicted drug reactions, pred_reaction. Do not include them in the ranked list. Only focus on the conditions in the list."
            "Always write your answer as a list of comma-separated drug reactions only and nothing else."
        )

        print("Running reranking with sem_map...")
        grouped_df = grouped_df.sem_map(rerank_prompt)

        end_t = time.time()

        grouped_df.to_csv(f"experiments/reasoning/othersystems/biodex/biodex_reranked_answers_{name}.csv", index=True)

        # Parse output with known prefixes from the original code
        known_prefixes = [
            "Based on the patient description, the most applicable adverse drug reactions are:\n\n",
            "Based on the Patient_description, the most applicable adverse drug reactions from the Combined_reaction_list are:\n\n",
            "Based on the Patient_description, the most applicable adverse drug reactions are:\n\n",
            "Based on the provided Patient_description, the most applicable adverse drug reactions from the Combined_reaction_list are:\n\n",
            "Here is the list of most applicable adverse drug reactions:\n\n",
            "Here is the answer:\n\n",
            "Here is the list of the most applicable adverse drug reactions:\n\n",
            "Here is the list of most applicable adverse drug reactions:\n\n",
            "Here is the list of most applicable adverse drug reactions from the options, ranked from most applicable to least applicable:"
        ]

        grouped_df["_map"] = grouped_df["_map"].fillna("").apply(
            lambda x: remove_known_prefixes(x, known_prefixes)
        )
        grouped_df.rename(columns={"pred_reaction": "pred_reaction_norank"}, inplace=True)
        grouped_df["pred_reaction"] = grouped_df["_map"].apply(
            lambda x: [reaction.strip() for reaction in x.split(",") if reaction.strip()]
        )

        return grouped_df, (end_t - start_t)

    def run_simple_map_pipeline(self, queries_df, corpus_df, name="test"):
        """Run a simple map-only pipeline similar to biodex.yaml"""
        start_t = time.time()

        # Create the possible_labels field for each query
        all_labels = "\n".join(corpus_df["reaction"].tolist())
        queries_df["possible_labels"] = all_labels

        # Use the same prompt as in biodex.yaml
        map_prompt = """You are an expert medical reviewer specializing in adverse drug reactions (ADR).

Your task:
1. Read the medical paper and identify every ADR or medical condition reported by patients.
2. Match each identified reaction to the single best-fitting official medical reaction term from the provided list.
3. Produce a ranked JSON array named ranked_reaction_labels ordered from most to least relevant.

<<MEDICAL_PAPER>>
PMID: {id}
{patient_description}
</MEDICAL_PAPER>>

<<OFFICIAL_REACTION_TERMS>>
(24,312 terms, ONE per line)
{possible_labels}
</OFFICIAL_REACTION_TERMS>>

Guidelines:
• Only return official terms that appear in the list above.
• If multiple official terms could match, choose the most specific one.
• Exclude generic or overly broad terms unless no better term exists.
• Return a valid JSON array, for example:
  ["term-A", "term-B", "term-C"]
• Do not include any additional keys, commentary, or markdown in the output."""

        print("Running simple map pipeline...")
        result_df = queries_df.sem_map(map_prompt)

        end_t = time.time()

        # Parse the JSON outputs
        def parse_json_output(output_str):
            """Parse the JSON output from the LLM response"""
            try:
                # Try to parse as JSON directly
                import json
                return json.loads(output_str)
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from markdown code blocks
                import re
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
                
                # If all else fails, return empty list
                print(f"Failed to parse JSON: {output_str}")
                return []

        # Find the output column created by sem_map
        output_col = [col for col in result_df.columns if col not in queries_df.columns][0]
        
        # Parse the JSON outputs in the new column
        print("Parsing JSON outputs...")
        result_df["ranked_reaction_labels"] = result_df[output_col].apply(parse_json_output)
        
        # Drop the original output column since we've extracted the field
        result_df = result_df.drop(columns=[output_col])

        return result_df, (end_t - start_t)

def main():
    start_time = time.time()
    
    datasets = [
        {"name": "train", "path": "experiments/reasoning/data/train/biodex.json"},
        {"name": "test", "path": "experiments/reasoning/data/test/biodex.json"}
    ]
    
    # Load the corpus (reaction labels)
    labels_path = "experiments/reasoning/data/biodex_labels.txt"
    corpus_df = load_biodex_labels(labels_path)
    print(f"Loaded {len(corpus_df)} reaction labels")
    
    eval_results = []
    
    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"Processing {dataset['name']} dataset...")
        print(f"{'='*50}")
        
        # Load the BioDEX dataset
        df = load_biodex_data(dataset['path'])
        print(f"Loaded {len(df)} documents")
        
        # Take only first 250 samples for testing (like the original)
        df_sample = df.head(250)
        print(f"Using {len(df_sample)} samples for processing")
        
        # Pipeline 1: Join Only - using gpt-4o-mini
        print("\n🔗 Running Join Pipeline...")
        join_pipeline = BioDEXPipeline(truncation_limit=128000, model="azure/gpt-4o-mini")
        queries_df_join = join_pipeline.prepare_queries_data(df_sample.copy())
        
        lotus.settings.lm.reset_stats()
        join_result_df, join_time, join_stats = join_pipeline.run_join_pipeline(
            queries_df_join.copy(), corpus_df, 
            recall_target=0.95, 
            precision_target=0.95, 
            name=f"join_{dataset['name']}"
        )
        
        # Prepare results for evaluation - join only
        join_results_list = []
        for _, row in join_result_df.iterrows():
            result_item = {
                "id": row.get("id", "unknown"),
                "fulltext_processed": row["patient_description"],
                "ground_truth_reactions": row["reactions_list"],
                "ranked_reaction_labels": row["pred_reaction"],
                "possible_labels": corpus_df["reaction"].tolist()
            }
            join_results_list.append(result_item)
        
        # Save join results
        join_output_path = f"experiments/reasoning/othersystems/biodex/lotus_join_{dataset['name']}.json"
        with open(join_output_path, 'w') as f:
            json.dump(join_results_list, f, indent=2)
        
        join_cost = lotus.settings.lm.stats.virtual_usage.total_cost
        print(f"Join Only: {len(join_results_list)} docs, {join_time:.2f}s, cost: {join_cost:.4f}")
        
        # Evaluate join results
        try:
            eval_func = get_evaluate_func("biodex")
            join_metrics = eval_func("lotus_join", join_output_path)
            
            print(f"📊 Join Results: RP@5={join_metrics['avg_rp_at_5']:.4f}, RP@10={join_metrics['avg_rp_at_10']:.4f}, TR={join_metrics['avg_term_recall']:.4f}")
            
            eval_results.append({
                "method": "join_only",
                "file": join_output_path,
                "avg_rp_at_5": join_metrics['avg_rp_at_5'],
                "avg_rp_at_10": join_metrics['avg_rp_at_10'],
                "avg_term_recall": join_metrics['avg_term_recall'],
                "total_documents": join_metrics['total_documents'],
                "cost": join_cost,
                "processing_time": join_time,
                "join_stats": join_stats
            })
        except Exception as e:
            print(f"⚠️  Join evaluation failed: {e}")
        
        # Pipeline 2: Join + Rerank (Full Pipeline) - using gpt-4o-mini
        print("\n🎯 Running Full Pipeline (Join + Rerank)...")
        full_pipeline = BioDEXPipeline(truncation_limit=128000, model="azure/gpt-4o-mini")
        queries_df_full = full_pipeline.prepare_queries_data(df_sample.copy())
        
        lotus.settings.lm.reset_stats()
        
        # Re-run join for full pipeline (fresh stats)
        join_result_df_full, join_time_full, join_stats_full = full_pipeline.run_join_pipeline(
            queries_df_full.copy(), corpus_df, 
            recall_target=0.95, 
            precision_target=0.95, 
            name=f"full_{dataset['name']}"
        )
        
        # Run rerank pipeline
        final_result_df, rerank_time = full_pipeline.run_rerank_pipeline(name=f"full_{dataset['name']}")
        
        # Prepare results for evaluation - full pipeline
        full_results_list = []
        for _, row in final_result_df.iterrows():
            result_item = {
                "id": row.get("id", "unknown"),
                "fulltext_processed": row["patient_description"],
                "ground_truth_reactions": row["reactions_list"],
                "ranked_reaction_labels": row["pred_reaction"],
                "possible_labels": corpus_df["reaction"].tolist()
            }
            full_results_list.append(result_item)
        
        # Save full pipeline results
        full_output_path = f"experiments/reasoning/othersystems/biodex/lotus_full_{dataset['name']}.json"
        with open(full_output_path, 'w') as f:
            json.dump(full_results_list, f, indent=2)
        
        full_cost = lotus.settings.lm.stats.virtual_usage.total_cost
        total_time = join_time_full + rerank_time
        print(f"Full Pipeline: {len(full_results_list)} docs, {total_time:.2f}s (join: {join_time_full:.2f}s, rerank: {rerank_time:.2f}s), cost: {full_cost:.4f}")
        
        # Evaluate full pipeline results
        try:
            eval_func = get_evaluate_func("biodex")
            full_metrics = eval_func("lotus_full", full_output_path)
            
            print(f"📊 Full Pipeline Results: RP@5={full_metrics['avg_rp_at_5']:.4f}, RP@10={full_metrics['avg_rp_at_10']:.4f}, TR={full_metrics['avg_term_recall']:.4f}")
            
            eval_results.append({
                "method": "join_and_rerank",
                "file": full_output_path,
                "avg_rp_at_5": full_metrics['avg_rp_at_5'],
                "avg_rp_at_10": full_metrics['avg_rp_at_10'],
                "avg_term_recall": full_metrics['avg_term_recall'],
                "total_documents": full_metrics['total_documents'],
                "cost": full_cost,
                "join_time": join_time_full,
                "rerank_time": rerank_time,
                "total_time": total_time,
                "join_stats": join_stats_full
            })
        except Exception as e:
            print(f"⚠️  Full pipeline evaluation failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Pipeline 3: Simple Map (like biodex.yaml) - using gpt-4.1-mini
        print("\n📝 Running Simple Map Pipeline...")
        simple_pipeline = BioDEXPipeline(truncation_limit=128000, model="azure/gpt-4.1-mini")
        queries_df_simple = simple_pipeline.prepare_queries_data(df_sample.copy())
        
        lotus.settings.lm.reset_stats()
        simple_result_df, simple_time = simple_pipeline.run_simple_map_pipeline(
            queries_df_simple.copy(), corpus_df, name=f"simple_{dataset['name']}"
        )
        
        # Prepare results for evaluation - simple map
        simple_results_list = []
        for _, row in simple_result_df.iterrows():
            result_item = {
                "id": row.get("id", "unknown"),
                "fulltext_processed": row["patient_description"],
                "ground_truth_reactions": row["reactions_list"],
                "ranked_reaction_labels": row["ranked_reaction_labels"],
                "possible_labels": corpus_df["reaction"].tolist()
            }
            simple_results_list.append(result_item)
        
        # Save simple map results
        simple_output_path = f"experiments/reasoning/othersystems/biodex/lotus_simple_{dataset['name']}.json"
        with open(simple_output_path, 'w') as f:
            json.dump(simple_results_list, f, indent=2)
        
        simple_cost = lotus.settings.lm.stats.virtual_usage.total_cost
        print(f"Simple Map: {len(simple_results_list)} docs, {simple_time:.2f}s, cost: {simple_cost:.4f}")
        
        # Evaluate simple map results
        try:
            eval_func = get_evaluate_func("biodex")
            simple_metrics = eval_func("lotus_simple", simple_output_path)
            
            print(f"📊 Simple Map Results: RP@5={simple_metrics['avg_rp_at_5']:.4f}, RP@10={simple_metrics['avg_rp_at_10']:.4f}, TR={simple_metrics['avg_term_recall']:.4f}")
            
            eval_results.append({
                "method": "simple_map",
                "file": simple_output_path,
                "avg_rp_at_5": simple_metrics['avg_rp_at_5'],
                "avg_rp_at_10": simple_metrics['avg_rp_at_10'],
                "avg_term_recall": simple_metrics['avg_term_recall'],
                "total_documents": simple_metrics['total_documents'],
                "cost": simple_cost,
                "processing_time": simple_time,
            })
        except Exception as e:
            print(f"⚠️  Simple evaluation failed: {e}")
    
    # Save combined evaluation metrics
    if eval_results:
        eval_output_path = "experiments/reasoning/othersystems/biodex/lotus_evaluation.json"
        with open(eval_output_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"\n📈 Combined evaluation metrics saved to: {eval_output_path}")
        print(f"📊 Total datasets processed: {len(eval_results)}")

if __name__ == "__main__":
    main()
    print("\n🔍 Final Usage Statistics:")
    lotus.settings.lm.print_total_usage()
