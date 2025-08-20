import json
import re
import pandas as pd
import numpy as np
from Levenshtein import distance
import random

def evaluate_results(method_name, results_file, ground_truth_file):
    # Read the DocETL results JSON file
    with open(results_file, "r") as f:
        docetl_results = json.load(f)
        docetl_results = pd.DataFrame(docetl_results)

    # Variables to track total clause length for calculating average
    all_text_spans = []

    # Read the ground truth CSV file
    ground_truth_df = pd.read_csv(ground_truth_file)
    ground_truth_df.columns = ground_truth_df.columns.map(
        lambda x: x.replace(" ", "_").lower()
    )

    # Sort ground truth dataframe
    ground_truth_df["filename"] = ground_truth_df["filename"].apply(
        lambda x: x.upper()
        .replace(".", "")
        .replace(",", "")
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("'", "")
        .replace(r'[^a-zA-Z0-9]$', '')
    )

    # Sort DocETL results
    filename_key = "name" if "name" in list(docetl_results.columns) else "filename"
    docetl_results["filename"] = docetl_results[filename_key].apply(
        lambda x: x.split("/")[-1]
        .upper()
        .rstrip(".TXT")
        .replace(".", "")
        .replace(",", "")
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("'", "")
        .replace(r'[^a-zA-Z0-9]$', '')
    )

    # Function to calculate precision and recall
    def calculate_metrics(metric, predicted_results, ground_truth):
        true_positives = 0
        false_positives = 0 
        false_negatives = 0
        true_negatives = 0

        # For all predicted results, strip the non-alphanumeric characters
        def clean_text(text):
            if isinstance(text, list):
                text = ' '.join(str(t) for t in text)
            elif not isinstance(text, str):
                text = str(text)
            cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text)
            return '' if len(cleaned) < 1 else cleaned
            
        predicted_results_cleaned = {k: clean_text(v) for k, v in predicted_results.items()}
        ground_truth_cleaned = {k: clean_text(v) for k, v in ground_truth.items()}

        # Calculate metrics across all filenames
        #print(f"----- {metric} -----")
        for filename in predicted_results_cleaned:
            pred = predicted_results_cleaned[filename]
            truth = ground_truth_cleaned.get(filename, "")
            
            # Calculate Jaccard similarity between truth and pred
            def get_jaccard_sim(str1, str2):
                words1 = set(str1.lower().split())
                words2 = set(str2.lower().split())
                intersection = words1.intersection(words2)
                union = words1.union(words2)
                return len(intersection) / len(union) if union else 0

            if truth != "" and pred != "":
                if get_jaccard_sim(truth, pred) > 0.15:
                    true_positives += 1
                else:
                    if method_name == "gemini_gemini_opt":
                        print(f"False positive (incorrect span) for {method_name}, {pred}")
                    false_positives += 1
            elif truth == "" and pred != "":
                if method_name == "gemini_gemini_opt":
                    print(f"False positive (no span) for {method_name}, {pred}")
                false_positives += 1
            elif truth != "" and pred == "":
                false_negatives += 1
            elif truth == "" and pred == "":
                true_negatives += 1

        # Calculate precision and recall
        total_predictions = true_positives + false_positives
        total_ground_truth = true_positives + false_negatives

        if total_predictions == 0:
            precision = 0.0
        else:
            precision = true_positives / total_predictions

        if total_ground_truth == 0:
            recall = float("nan")
        else:
            recall = true_positives / total_ground_truth

        return precision, recall

    # List of metrics to evaluate
    metrics = [
        "document_name", "parties", "agreement_date", "effective_date", 
        "expiration_date", "renewal_term", "notice_to_terminate_renewal",
        "governing_law", "most_favored_nation", "non_compete", "exclusivity",
        "no_solicit_of_customers", "competitive_restriction_exception",
        "no_solicit_of_employees", "non_disparagement", "termination_for_convenience",
        "right_of_first_refusal", "change_of_control", "anti_assignment",
        "revenue_profit_sharing", "price_restriction", "minimum_commitment",
        "volume_restriction", "ip_ownership_assignment", "joint_ip_ownership",
        "license_grant", "non_transferable_license", "affiliate_ip_license_licensor",
        "affiliate_ip_license_licensee", "unlimited_license",
        "irrevocable_or_perpetual_license", "source_code_escrow",
        "post_termination_services", "audit_rights", "uncapped_liability",
        "cap_on_liability", "liquidated_damages", "warranty_duration",
        "insurance", "covenant_not_to_sue", "third_party_beneficiary",
    ]

    # Reindex the dataframes and join them
    docetl_results = docetl_results.sort_values(by="filename").reset_index()

    # print(f"Number of documents in results: {len(docetl_results)}")

    # Find closest matching filename for each docetl result
    matched_filenames = []
    for docetl_filename in docetl_results["filename"]:
        closest_match = max(
            ground_truth_df["filename"],
            key=lambda x: sum(a == b for a, b in zip(x, docetl_filename))
        )
        matched_filenames.append(closest_match)
    

    ground_truth_df = ground_truth_df[ground_truth_df["filename"].isin(matched_filenames)]
    ground_truth_df = ground_truth_df.sort_values(by="filename").reset_index()
    docetl_results = docetl_results.sort_values(by="filename").reset_index()
    # Drop any of the metric cols from docetl_results if they exist
    existing_metrics = [col for col in metrics if col in docetl_results.columns]
    if existing_metrics:
        docetl_results = docetl_results.drop(columns=existing_metrics)

    # Merge the dataframes on the index
    merged_df = pd.merge(docetl_results, ground_truth_df, left_index=True, right_index=True, how="inner")
    # print(merged_df.head())

    # Calculate precision and recall for DocETL results
    docetl_metrics = {}
    for metric in metrics:
        if metric in merged_df.columns:
            predicted_clauses = dict(zip(merged_df["filename_x"], merged_df["clauses"]))
            predicted_results = {}
            for k, clauses in predicted_clauses.items():
                if isinstance(clauses, dict):
                    clauses = clauses.get("clauses", [])

                if not clauses:
                    predicted_results[k] = ""
                    continue
                
                # Print out problematic c values
                for c in clauses:
                    if not (isinstance(c, dict) and "clause_type" in c and "text_span" in c):
                        pass
                        #print(f"Problematic clause entry: {c} (type: {type(c)})")
                clauses = [{
                    "clause_type": c["clause_type"].lower().strip().replace(" ", "_").replace("-", "_"),
                    "text_span": c["text_span"]
                } for c in clauses if isinstance(c, dict) and "clause_type" in c and "text_span" in c]
                clause_types = [c["clause_type"] for c in clauses]

                if len(clause_types) == 0:
                    predicted_results[k] = ""
                    continue
                
                closest_match = min(
                    clause_types,
                    key=lambda x: distance(x, metric)
                )
                
                # If the closest match doesn't share any words with the metric, set the closest match to ""
                metric_words = metric.split('_')
                closest_match_words = closest_match.split('_')
                if not any(word in closest_match_words for word in metric_words):
                    predicted_results[k] = ""
                    continue
                
                match_for_clause_type = [c["text_span"] for c in clauses if c["clause_type"] == closest_match][0]
                predicted_results[k] = match_for_clause_type

            precision, recall = calculate_metrics(
                metric, predicted_results, merged_df[["filename_x", metric]].set_index("filename_x")[metric].to_dict()
            )
            docetl_metrics[metric] = {"precision": precision, "recall": recall}
            
            # Track all extracted text spans for average length calculation
            for key, text_span in predicted_results.items():
                if text_span and isinstance(text_span, str) and len(text_span.strip()) > 0:
                    all_text_spans.append(text_span)
        else:
            docetl_metrics[metric] = {"precision": np.nan, "recall": np.nan}

    # Calculate average clause length
    avg_clause_length = 0
    if len(all_text_spans) > 0:
        avg_clause_length = sum(len(span) for span in all_text_spans) / len(all_text_spans)

    # Calculate average metrics
    precisions = [v["precision"] for v in docetl_metrics.values()]
    recalls = [v["recall"] for v in docetl_metrics.values()]

    avg_precision = np.nanmean(precisions)
    avg_recall = np.nanmean(recalls)
    nan_fraction = (np.isnan(precisions) | np.isnan(recalls)).mean()

    # Calculate F1 score for non-nan values
    f1_scores = [
        (
            2 * (p * r) / (p + r)
            if not (np.isnan(p) or np.isnan(r)) and (p + r) != 0
            else 0 if (p + r) == 0 else np.nan
        )
        for p, r in zip(precisions, recalls)
    ]
    avg_f1 = np.nanmean(f1_scores)

    return {
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "nan_fraction": nan_fraction,
        "avg_f1": avg_f1,
        "avg_clause_length": avg_clause_length,
        "per_metric": docetl_metrics
    }


if __name__ == "__main__":
    import argparse
    import sys
    import os
    from docetl.utils import extract_output_from_json
    
    parser = argparse.ArgumentParser(description="Evaluate CUAD results from a single JSON file")
    parser.add_argument("results_file", help="Path to the results JSON file to evaluate")
    parser.add_argument("--ground_truth", "-gt", default="/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/experiments/reasoning/data/CUAD-master_clauses.csv", 
                       help="Path to the ground truth CSV file (default: experiments/reasoning/data/train/cuad_train.csv)")
    parser.add_argument("--method_name", "-m", default="evaluation", 
                       help="Name of the method being evaluated (default: evaluation)")
    
    args = parser.parse_args()
    
    # yaml_file_path = "/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/docetl/test_pipeline.yaml"
    # result = extract_output_from_json(yaml_file_path, args.results_file)
    # print(result)
    # exit()

    # Check if results file exists
    if not os.path.exists(args.results_file):
        print(f"Error: Results file '{args.results_file}' not found.")
        sys.exit(1)
    
    # Check if ground truth file exists
    if not os.path.exists(args.ground_truth):
        print(f"Error: Ground truth file '{args.ground_truth}' not found.")
        print("Please provide the correct path using --ground_truth or -gt")
        sys.exit(1)
    
    try:
        print(f"Evaluating results from: {args.results_file}")
        print(f"Using ground truth from: {args.ground_truth}")
        print(f"Method name: {args.method_name}")
        print("-" * 50)
        
        # Evaluate the results
        results = evaluate_results(args.method_name, args.results_file, args.ground_truth)
        
        # Print results in a formatted way
        print(f"Evaluation Results for {args.method_name}:")
        print(f"Average Precision: {results['avg_precision']:.4f}")
        print(f"Average Recall: {results['avg_recall']:.4f}")
        print(f"Average F1 Score: {results['avg_f1']:.4f}")
        print(f"NaN Fraction: {results['nan_fraction']:.4f}")
        print(f"Average Clause Length: {results['avg_clause_length']:.1f} characters")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

