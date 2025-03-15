import json
import re
import pandas as pd
import numpy as np
from Levenshtein import distance

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
            cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text)
            return '' if len(cleaned) < 1 else cleaned
            
        predicted_results_cleaned = {k: clean_text(v) for k, v in predicted_results.items()}
        ground_truth_cleaned = {k: clean_text(v) for k, v in ground_truth.items()}

        # Calculate metrics across all filenames
        print(f"----- {metric} -----")
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
    print(merged_df.head())

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
                
                clauses = [{
                    "clause_type": c["clause_type"].lower().strip().replace(" ", "_").replace("-", "_"),
                    "text_span": c["text_span"]
                } for c in clauses if "clause_type" in c and "text_span" in c]
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

# Evaluate both results files
results_3_7 = evaluate_results(
    "docetl_3_7",
    "agenticpreprint/cuad/results/extracted_contract_info.json",
    "agenticpreprint/cuad/master_clauses.csv"
)

results_preprint = evaluate_results(
    "docetl_preprint",
    "agenticpreprint/cuad/results/extracted_contract_info_preprint.json",
    "agenticpreprint/cuad/master_clauses.csv"
)

# Create results DataFrame with both evaluations
results_df = pd.DataFrame(
    {
        "Model": ["DocETL 3-7", "DocETL Preprint"],
        "Avg Precision": [results_3_7["avg_precision"], results_preprint["avg_precision"]],
        "Avg Recall": [results_3_7["avg_recall"], results_preprint["avg_recall"]],
        "Avg F1": [results_3_7["avg_f1"], results_preprint["avg_f1"]],
        "Avg Clause Length (chars)": [results_3_7["avg_clause_length"], results_preprint["avg_clause_length"]]
    }
)

# Display per-metric results for both evaluations
print("\nPer-Metric Results for DocETL 3-7:")
metrics_df_3_7 = pd.DataFrame.from_dict(results_3_7["per_metric"], orient='index')
print(metrics_df_3_7)

print("\nPer-Metric Results for DocETL Preprint:")
metrics_df_preprint = pd.DataFrame.from_dict(results_preprint["per_metric"], orient='index')
print(metrics_df_preprint)

# Display the overall results
print("\nEvaluation Results:")
print(results_df.to_string(index=False))

