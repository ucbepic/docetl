import argparse
import json
import os
import numpy as np
import pandas as pd

import palimpzest as pz
from palimpzest.constants import Model
from palimpzest.core.lib.fields import BooleanField, StringField
from palimpzest.policy import MaxQuality, MaxQualityAtFixedCost

from dotenv import load_dotenv
from experiments.reasoning.evaluation.utils import get_evaluate_func
from experiments.reasoning.evaluation.medec import jaccard_similarity

load_dotenv()

# Budget fractions to test
FRACS = [0.75, 0.5, 0.25, 0.1]


def load_medec_data(split="train"):
    """Load MEDEC data from our existing JSON files."""
    file_path = f"experiments/reasoning/data/{split}/medec.json"
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


class MEDECDataReader(pz.DataReader):
    def __init__(self, num_texts: int = 100, split: str = "train", seed: int = 42):
        self.num_texts = num_texts
        self.split = split
        self.seed = seed

        input_cols = [
            {"name": "text_id", "type": str, "desc": "The id of the medical text to be analyzed"},
            {"name": "text", "type": str, "desc": "The medical narrative text with numbered sentences"},
        ]
        super().__init__(input_cols)
        
        # Load the dataset
        data = load_medec_data(split)
        
        # Sample texts if needed
        if len(data) > num_texts:
            np.random.seed(seed)
            indices = np.random.choice(len(data), num_texts, replace=False)
            data = [data[i] for i in indices]
        
        self.dataset = data

    def compute_label(self, entry: dict) -> dict:
        """Compute the label for a MEDEC entry given its data."""
        # Handle error flag
        error_flag = entry.get("Error Flag", 0)
        if isinstance(error_flag, str):
            error_flag = int(error_flag) if error_flag.isdigit() else 0
        
        label_dict = {
            "is_error": bool(error_flag),
            "error_sentence": entry.get("Error Sentence", "") or "",
            "corrected_sentence": entry.get("Corrected Sentence", "") or ""
        }
        return label_dict

    @staticmethod
    def error_flag_accuracy_score(pred_error: bool, target_error: bool) -> float:
        """Score function for error flag prediction accuracy."""
        return 1.0 if pred_error == target_error else 0.0

    @staticmethod
    def error_sentence_jaccard_score(pred_sentence: str, target_sentence: str) -> float:
        """Score function for error sentence using Jaccard similarity."""
        if not target_sentence:  # No error sentence to compare against
            return 1.0 if not pred_sentence else 0.0
        
        if not isinstance(pred_sentence, str) or not isinstance(target_sentence, str):
            # Not sure why we get here. Sometimes the pred_sentence is None or a float.
            print(f"Predicted sentence: {pred_sentence}, Target sentence: {target_sentence}. Converting to strings.")
            pred_sentence = str(pred_sentence)
            target_sentence = str(target_sentence)
        
        return jaccard_similarity(pred_sentence, target_sentence)

    @staticmethod
    def corrected_sentence_jaccard_score(pred_sentence: str, target_sentence: str) -> float:
        """Score function for corrected sentence using Jaccard similarity."""
        if not target_sentence:  # No corrected sentence to compare against
            return 1.0 if not pred_sentence else 0.0
        
        if not isinstance(pred_sentence, str) or not isinstance(target_sentence, str):
            # Not sure why we get here. Sometimes the pred_sentence is None or a float.
            print(f"Predicted sentence: {pred_sentence}, Target sentence: {target_sentence}. Converting to strings.")
            pred_sentence = str(pred_sentence)
            target_sentence = str(target_sentence)
        
        return jaccard_similarity(pred_sentence, target_sentence)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        
        # Create the basic item structure
        result_item = {"fields": {}, "labels": {}, "score_fn": {}}
        result_item["fields"]["text_id"] = item["Text ID"]
        result_item["fields"]["text"] = item["Sentences"]
        
        if self.split == "train":
            # Add label info for training split
            result_item["labels"] = self.compute_label(item)
            
            # Add scoring functions
            result_item["score_fn"]["is_error"] = MEDECDataReader.error_flag_accuracy_score
            result_item["score_fn"]["error_sentence"] = MEDECDataReader.error_sentence_jaccard_score
            result_item["score_fn"]["corrected_sentence"] = MEDECDataReader.corrected_sentence_jaccard_score
        
        return result_item

def build_medec_query(dataset):
    """Build the MEDEC query for medical error detection."""
    ds = pz.Dataset(dataset)
    
    # Define the medical error detection schema
    cols = [
        {
            "name": "is_error", 
            "type": BooleanField, 
            "desc": "Whether the medical text contains an error (true/false)"
        },
        {
            "name": "error_sentence", 
            "type": StringField, 
            "desc": "The original sentence containing the medical error, if any"
        },
        {
            "name": "corrected_sentence", 
            "type": StringField, 
            "desc": "The corrected version of the error sentence, if any"
        }
    ]
    
    desc = """The text is a medical narrative about a patient. You are a skilled medical doctor reviewing the clinical text.
The text is either correct or contains one error. The text has one sentence per line. Each line starts with the sentence ID,
followed by a space, then the sentence to check. Check every sentence of the text. If the text is correct, return the
following output: is_error=false, error_sentence="", corrected_sentence="". If the text has a medical error related to treatment,
management, cause, or diagnosis, return the sentence id of the sentence containing the error, the original sentence, and a corrected version of the sentence.
Finding and correcting the error requires medical knowledge and reasoning.

Here is an example:
0 A 35-year-old woman presents to her physician with a complaint of pain and stiffness in her hands.
1 She says that the pain began 6 weeks ago a few days after she had gotten over a minor upper respiratory infection.
2 She has no history of trauma.
3 She has no significant past medical history.
4 On examination, there is swelling and tenderness of the metacarpophalangeal and proximal interphalangeal joints of both hands.
5 The wrists are also swollen and tender.
6 There is no evidence of joint deformity.
7 Laboratory studies show a positive rheumatoid factor and elevated erythrocyte sedimentation rate.
8 The C-reactive protein is elevated.
9 Bilateral radiographs of the hands demonstrate mild periarticular osteopenia around the left fifth metacarpophalangeal joint.
10 Methotrexate is given.

In this example, the error is in sentence 10: "Methotrexate is given." The correction is: "Prednisone is given."
The output is:
is_error=true, error_sentence="Methotrexate is given.", corrected_sentence="Prednisone is given."

End of Example.
"""
    
    ds = ds.sem_add_columns(cols, desc=desc, depends_on=["text"])
    
    return ds


def convert_predictions_to_medec_format(pred_df, data_reader):
    """Convert Palimpzest predictions to the format expected by MEDEC evaluation."""
    results = []
    
    # Create a mapping from text_id to original data for ground truth
    original_data_map = {item["Text ID"]: item for item in data_reader.dataset}
    
    for _, row in pred_df.iterrows():
        # Handle prediction values
        is_error = row.get("is_error", False)
        error_sentence = row.get("error_sentence", "")
        corrected_sentence = row.get("corrected_sentence", "")
        
        # Clean up the values
        if pd.isna(is_error):
            is_error = False
        if pd.isna(error_sentence):
            error_sentence = ""
        if pd.isna(corrected_sentence):
            corrected_sentence = ""
        
        # Get original data for ground truth
        text_id = row["text_id"]
        original_data = original_data_map.get(text_id, {})
        
        # Create the record in the format matching the evaluation
        # Include both predictions and ground truth data
        record = {
            "Text ID": text_id,
            "Text": original_data.get("Text", ""),
            "Sentences": row["text"],  # The numbered sentences
            "Error Flag": original_data.get("Error Flag", "0"),
            "Error Type": original_data.get("Error Type"),
            "Error Sentence ID": original_data.get("Error Sentence ID", "-1"),
            "Error Sentence": original_data.get("Error Sentence"),
            "Corrected Sentence": original_data.get("Corrected Sentence"),
            "Corrected Text": original_data.get("Corrected Text"),
            # Predictions
            "is_error": bool(is_error),
            "error_sentence": str(error_sentence).strip(),
            "corrected_sentence": str(corrected_sentence).strip()
        }
        results.append(record)
    
    return results


def run_experiment(data_reader, val_data_reader, policy, models, 
                   sentinel_strategy="mab", k=10, j=3, sample_budget=100, seed=42, exp_name=None):
    """Run a single experiment with given policy and return results."""
    print(f"\nRunning experiment with policy: {policy}")
    
    # Build query
    query = build_medec_query(data_reader)
    
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
        exp_name=exp_name if exp_name else f"medec-pz-{policy.__class__.__name__}",
        priors=None,
    )
    
    pred_df = data_record_collection.to_df()
    
    # Convert to MEDEC format
    results_list = convert_predictions_to_medec_format(pred_df, data_reader)
    
    # Save results as JSON
    output_file = f"experiments/reasoning/othersystems/medec/{exp_name}.json" if exp_name else "experiments/reasoning/othersystems/medec/pz_temp.json"
    with open(output_file, 'w') as f:
        json.dump(results_list, f, indent=2)
    
    # Evaluate using our existing evaluation framework
    evaluate_func = get_evaluate_func("medec")
    metrics = evaluate_func("palimpzest", output_file)
    
    # Get execution statistics for cost
    exec_stats = data_record_collection.execution_stats
    
    return {
        "combined_score": metrics["combined_score"],
        "error_flag_accuracy": metrics["error_flag_accuracy"],
        "avg_error_sentence_jaccard": metrics["avg_error_sentence_jaccard"],
        "avg_corrected_sentence_jaccard": metrics["avg_corrected_sentence_jaccard"],
        "total_cases": metrics["total_cases"],
        "num_error_cases": metrics["num_error_cases"],
        "num_corrected_cases": metrics["num_corrected_cases"],
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
    parser = argparse.ArgumentParser(description="Run MEDEC experiments with budget analysis using Palimpzest")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-texts", type=int, default=100, help="Number of medical texts to process")
    parser.add_argument(
        "--sentinel-execution-strategy",
        default="mab",
        type=str,
        help="The engine to use. One of mab or random",
    )
    parser.add_argument(
        "--k",
        default=6,
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
        Model.GPT_4o,
        Model.GPT_4o_MINI,
        Model.GPT_41_MINI,
        Model.GPT_41,
        Model.GPT_41_NANO,
        Model.GPT_5_MINI,
        Model.GPT_5,
        Model.GPT_5_NANO,
        Model.GEMINI_25_FLASH,
        Model.GEMINI_25_FLASH_LITE,
        Model.GEMINI_25_PRO,
    ]
    
    print("Loading MEDEC dataset...")
    
    # Create data readers: test set for main evaluation, train set for validation
    data_reader = MEDECDataReader(split="test", num_texts=args.num_texts, seed=args.seed)
    val_data_reader = MEDECDataReader(split="train", seed=args.seed)
    
    print(f"Processing {len(data_reader)} test medical texts with Palimpzest...")
    print(f"Using {len(val_data_reader)} train medical texts for validation...")
    
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
    print(f"Unconstrained Combined Score: {unconstrained_result['combined_score']:.4f}")
    
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
        print(f"{int(frac*100)}% budget Combined Score: {budget_result['combined_score']:.4f}")
    
    # Add metadata
    results["metadata"] = {
        "seed": args.seed,
        "num_texts": args.num_texts,
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
    output_path = "experiments/reasoning/othersystems/medec/pz_evaluation.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📈 Combined results saved to: {output_path}")
    
    # Print summary
    print("\n=== Summary ===")
    print(f"{'Configuration':<30} {'Cost ($)':<12} {'Combined':<12} {'Error Flag':<12} {'Error Sent':<12} {'Corrected':<12}")
    print(f"{'':30} {'':12} {'Score':<12} {'Accuracy':<12} {'Jaccard':<12} {'Jaccard':<12}")
    print("-" * 90)
    
    # Print unconstrained result
    r = results["unconstrained_max_quality"]
    print(f"{'Unconstrained':<30} {r['total_execution_cost']:<12.4f} {r['combined_score']:<12.4f} {r['error_flag_accuracy']:<12.4f} {r['avg_error_sentence_jaccard']:<12.4f} {r['avg_corrected_sentence_jaccard']:<12.4f}")
    
    # Print budget results
    for frac in FRACS:
        key = f"budget_{int(frac*100)}_percent"
        label = f"{int(frac*100)}% Budget"
        r = results[key]
        print(f"{label:<30} {r['total_execution_cost']:<12.4f} {r['combined_score']:<12.4f} {r['error_flag_accuracy']:<12.4f} {r['avg_error_sentence_jaccard']:<12.4f} {r['avg_corrected_sentence_jaccard']:<12.4f}")


if __name__ == "__main__":
    main()
