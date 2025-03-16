import json
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from Levenshtein import distance
from scipy.stats import spearmanr, kendalltau

# Constants
RESULTS_DIR = "agenticpreprint/testrewriteability/results"
PLANS_DIR = "agenticpreprint/testrewriteability/proj_synth_plans"
GROUND_TRUTH_FILE = "agenticpreprint/cuad/master_clauses.csv"
EVALUATION_RESULTS_FILE = os.path.join(PLANS_DIR, "evaluation_results.json")
OUTPUT_DIR = "agenticpreprint/testrewriteability/metrics"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def evaluate_plan_results(plan_name, results_file, ground_truth_file):
    """
    Evaluate the results of a single plan against ground truth data.
    
    Args:
        plan_name: Name of the plan
        results_file: Path to the results file for this plan
        ground_truth_file: Path to the ground truth data
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Read the plan results JSON file
    with open(results_file, "r") as f:
        plan_results_data = json.load(f)
        # Extract results from the structure
        if "results" in plan_results_data:
            plan_results = plan_results_data["results"]
        else:
            plan_results = plan_results_data
        plan_results = pd.DataFrame(plan_results)

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

    # Sort plan results
    filename_key = "name" if "name" in list(plan_results.columns) else "filename"
    if filename_key not in plan_results.columns:
        # Try to generate a filename column from document field
        if "document" in plan_results.columns:
            plan_results["filename"] = plan_results["document"].apply(
                lambda x: x[:50] if isinstance(x, str) else "unknown"
            )
        else:
            # Just assign a sequential number to each row
            plan_results["filename"] = [f"doc_{i}" for i in range(len(plan_results))]
    else:
        plan_results["filename"] = plan_results[filename_key].apply(
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

    # Function to calculate precision and recall for a specific metric
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
                    false_positives += 1
            elif truth == "" and pred != "":
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
            
        # Calculate F1 score
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall) if not np.isnan(recall) else 0.0

        return precision, recall, f1

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
    plan_results = plan_results.sort_values(by="filename").reset_index()

    # Find closest matching filename for each plan result
    matched_filenames = []
    for plan_filename in plan_results["filename"]:
        closest_match = max(
            ground_truth_df["filename"],
            key=lambda x: sum(a == b for a, b in zip(x, plan_filename))
        )
        matched_filenames.append(closest_match)

    ground_truth_df = ground_truth_df[ground_truth_df["filename"].isin(matched_filenames)]
    ground_truth_df = ground_truth_df.sort_values(by="filename").reset_index()
    plan_results = plan_results.sort_values(by="filename").reset_index()
    
    # Drop any of the metric cols from plan_results if they exist
    existing_metrics = [col for col in metrics if col in plan_results.columns]
    if existing_metrics:
        plan_results = plan_results.drop(columns=existing_metrics)

    # Merge the dataframes on the index
    merged_df = pd.merge(plan_results, ground_truth_df, left_index=True, right_index=True, how="inner")

    # Calculate precision and recall for plan results
    plan_metrics = {}
    for metric in metrics:
        if metric in merged_df.columns:
            predicted_clauses = dict(zip(merged_df["filename_x"], merged_df["clauses"]))
            predicted_results = {}
            for k, clauses in predicted_clauses.items():
                if isinstance(clauses, dict):
                    clauses = clauses.get("clauses", [])
                elif isinstance(clauses, str):
                    try:
                        clauses = json.loads(clauses).get("clauses", [])
                    except:
                        clauses = []

                if not clauses:
                    predicted_results[k] = ""
                    continue
                
                try:
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
                
                    # Track all extracted text spans for average length calculation
                    if match_for_clause_type and isinstance(match_for_clause_type, str) and len(match_for_clause_type.strip()) > 0:
                        all_text_spans.append(match_for_clause_type)
                except Exception as e:
                    # Handle any errors with this plan's clause structure
                    predicted_results[k] = ""

            precision, recall, f1 = calculate_metrics(
                metric, predicted_results, merged_df[["filename_x", metric]].set_index("filename_x")[metric].to_dict()
            )
            plan_metrics[metric] = {"precision": precision, "recall": recall, "f1": f1}
        else:
            plan_metrics[metric] = {"precision": np.nan, "recall": np.nan, "f1": np.nan}

    # Calculate average clause length
    avg_clause_length = 0
    if len(all_text_spans) > 0:
        avg_clause_length = sum(len(span) for span in all_text_spans) / len(all_text_spans)

    # Calculate average metrics
    precisions = [v["precision"] for v in plan_metrics.values()]
    recalls = [v["recall"] for v in plan_metrics.values()]
    f1_scores = [v["f1"] for v in plan_metrics.values()]

    avg_precision = np.nanmean(precisions)
    avg_recall = np.nanmean(recalls)
    avg_f1 = np.nanmean(f1_scores)
    
    # Calculate runtime and cost if available in metadata
    runtime = plan_results_data.get("metadata", {}).get("runtime", 0)
    cost = plan_results_data.get("metadata", {}).get("cost", 0)

    return {
        "plan_name": plan_name,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1,
        "avg_clause_length": avg_clause_length,
        "runtime": runtime,
        "cost": cost,
        "per_metric": plan_metrics
    }

def evaluate_all_plans():
    """
    Evaluate all plans in the results directory and compare with pairwise rankings.
    """
    # Get all result files
    result_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.json') and not f.endswith('_error.json')]
    
    # Evaluate each plan
    all_metrics = []
    for result_file in result_files:
        plan_name = os.path.splitext(result_file)[0]
        file_path = os.path.join(RESULTS_DIR, result_file)
        
        try:
            print(f"Evaluating {plan_name}...")
            metrics = evaluate_plan_results(plan_name, file_path, GROUND_TRUTH_FILE)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Error evaluating {plan_name}: {str(e)}")
    
    # Sort plans by F1 score
    sorted_metrics = sorted(all_metrics, key=lambda x: x["avg_f1"], reverse=True)
    
    # Create a DataFrame for comparison
    comparison_df = pd.DataFrame([
        {
            "Plan": m["plan_name"],
            "Avg Precision": m["avg_precision"],
            "Avg Recall": m["avg_recall"],
            "F1 Score": m["avg_f1"],
            "Clause Length": m["avg_clause_length"],
            "Runtime (s)": m["runtime"],
            "API Cost ($)": m["cost"]
        }
        for m in sorted_metrics
    ])
    
    # Save CSV of all plan metrics
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, "plan_metrics.csv"), index=False)
    
    # Create a DataFrame with per-metric performance for each plan
    metric_columns = sorted_metrics[0]["per_metric"].keys()
    per_metric_rows = []
    
    for metric in metric_columns:
        metric_data = {"Metric": metric}
        for m in sorted_metrics:
            plan_name = m["plan_name"]
            f1 = m["per_metric"][metric]["f1"]
            metric_data[plan_name] = f1
        per_metric_rows.append(metric_data)
    
    per_metric_df = pd.DataFrame(per_metric_rows)
    per_metric_df.to_csv(os.path.join(OUTPUT_DIR, "per_metric_performance.csv"), index=False)
    
    # Variable to store top optimizer plan
    top_optimizer_plan = None
    
    # Load pairwise rankings
    if os.path.exists(EVALUATION_RESULTS_FILE):
        with open(EVALUATION_RESULTS_FILE, 'r') as f:
            eval_results = json.load(f)
            pairwise_rankings = eval_results.get("pairwise_rankings", {})
            optimizer_scores = eval_results.get("results", {})
        
        if pairwise_rankings:
            # Find the plan with the highest win count (LLM judge's Top Pick (from controlled generations))
            if pairwise_rankings:
                top_optimizer_plan = max(pairwise_rankings.items(), key=lambda x: x[1])[0]
                print(f"Optimizer's top-ranked plan: {top_optimizer_plan}")
            
            # Create mapping from plan name to F1 score (true performance)
            true_rankings = {m["plan_name"]: m["avg_f1"] for m in sorted_metrics}
            
            # Compare pairwise rankings with true rankings
            compare_rankings(pairwise_rankings, true_rankings, optimizer_scores)
    
    # Create visualizations
    create_visualizations(sorted_metrics)
    
    return comparison_df, sorted_metrics

def compare_rankings(pairwise_rankings, true_rankings, optimizer_scores):
    """
    Compare the optimizer's pairwise rankings with true rankings.
    
    Args:
        pairwise_rankings: Dictionary mapping plan names to their pairwise win counts
        true_rankings: Dictionary mapping plan names to their F1 scores
        optimizer_scores: Dictionary containing scores from the optimizer for each plan
    """
    # Create a dictionary of all optimization plan scores
    opt_plan_scores = {}
    for plan_name, plan_info in optimizer_scores.items():
        if isinstance(plan_info, dict) and "score" in plan_info:
            opt_plan_scores[plan_name] = plan_info["score"]
    
    # Print optimizer scores for debugging
    print(f"Found {len(opt_plan_scores)} plan scores in evaluation_results.json")
    
    # Sort both rankings by their respective scores
    # For optimizer rankings, use pairwise win counts when available
    optimizer_ordered_plans = []
    for plan_name in pairwise_rankings.keys():
        optimizer_ordered_plans.append((plan_name, pairwise_rankings[plan_name]))
    
    # For plans not in pairwise_rankings but in optimizer_scores, use their score
    for plan_name, score in opt_plan_scores.items():
        if plan_name not in pairwise_rankings:
            optimizer_ordered_plans.append((plan_name, score))
    
    # Sort by score (higher is better)
    optimizer_ordered_plans.sort(key=lambda x: x[1], reverse=True)
    
    # Create a sorted list of true plans by F1 score
    true_ordered_plans = [(name, score) for name, score in true_rankings.items()]
    true_ordered_plans.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Optimizer plans: {len(optimizer_ordered_plans)}")
    print(f"True ranking plans: {len(true_ordered_plans)}")
    
    # Check for any 'glean' plan types in both lists (for debugging)
    opt_glean_plans = [p for p, _ in optimizer_ordered_plans if "glean" in p]
    true_glean_plans = [p for p, _ in true_ordered_plans if "glean" in p]
    
    print(f"Optimizer has {len(opt_glean_plans)} glean plans: {opt_glean_plans}")
    print(f"True rankings has {len(true_glean_plans)} glean plans: {true_glean_plans}")
    
    # Improved function to extract plan type and number for matching
    def extract_plan_info(plan_name):
        """
        Extract the plan type and number from a plan name.
        
        Args:
            plan_name: String name of the plan
            
        Returns:
            Tuple of (plan_type, plan_number)
        """
        plan_type = None
        plan_number = None
        
        # Handle specific plan name formats directly
        if plan_name.startswith("glean_plan_"):
            plan_type = "glean"
            try:
                plan_number = int(plan_name.replace("glean_plan_", ""))
                return plan_type, plan_number
            except ValueError:
                pass
        
        elif plan_name.startswith("chain_plan_"):
            plan_type = "chain"
            try:
                plan_number = int(plan_name.replace("chain_plan_", ""))
                return plan_type, plan_number
            except ValueError:
                pass
        
        elif plan_name.startswith("parallel_plan_"):
            plan_type = "parallel"
            try:
                plan_number = int(plan_name.replace("parallel_plan_", ""))
                return plan_type, plan_number
            except ValueError:
                pass
        
        # General extraction if specific formats don't match
        parts = plan_name.split('_')
        
        # Extract plan type (prefix)
        if len(parts) >= 1:
            if parts[0] == "parallel" and len(parts) >= 2 and parts[1] == "map":
                plan_type = "parallel"
            elif parts[0] in ["chain", "glean", "parallel"]:
                plan_type = parts[0]
        
        # Special handling for plans with "with_transform" in the name
        if "with_transform" in plan_name:
            # For parallel_map_with_transform plans, find number after "transform_"
            parts = plan_name.split("transform_")
            if len(parts) > 1:
                try:
                    plan_number = int(parts[1])
                    return plan_type, plan_number
                except ValueError:
                    pass
        
        # For other plans, try to extract number from suffix
        # Start from the end to find the first numeric part
        for part in reversed(plan_name.split('_')):
            try:
                plan_number = int(part)
                break
            except ValueError:
                continue
        
        # Debug output for troublesome plans
        if plan_type is None or plan_number is None:
            print(f"Warning: Could not extract type or number from plan: {plan_name}")
        
        return plan_type, plan_number
    
    # Try to match plans between the two lists
    matched_plans = []
    unmatched_opt_plans = []
    unmatched_true_plans = []
    
    # Print all plan info for debugging
    print("\nOptimizer plan info:")
    opt_plan_details = {}
    for name, _ in optimizer_ordered_plans:
        plan_type, plan_number = extract_plan_info(name)
        opt_plan_details[(plan_type, plan_number)] = name
        print(f"  {name}: Type={plan_type}, Number={plan_number}")
    
    print("\nTrue plan info:")
    true_plan_details = {}
    for name, _ in true_ordered_plans:
        plan_type, plan_number = extract_plan_info(name)
        true_plan_details[(plan_type, plan_number)] = name
        print(f"  {name}: Type={plan_type}, Number={plan_number}")
    
    # Double-check specific plan types for debugging
    glean_plan_10_key = None
    for key, name in true_plan_details.items():
        if name == "glean_plan_10":
            glean_plan_10_key = key
            print(f"Found glean_plan_10 with key: {glean_plan_10_key}")
            if glean_plan_10_key in opt_plan_details:
                print(f"This key IS in optimizer details with value: {opt_plan_details[glean_plan_10_key]}")
            else:
                print(f"This key is NOT in optimizer details")
                
                # Search for any keys with type 'glean' and number 10
                for (type_, num_), name_ in opt_plan_details.items():
                    if type_ == "glean" and num_ == 10:
                        print(f"Found potential match: {name_} with key ({type_}, {num_})")
    
    # Find matched plans
    for key, opt_name in opt_plan_details.items():
        if key in true_plan_details:
            matched_plans.append((opt_name, true_plan_details[key]))
        else:
            unmatched_opt_plans.append((opt_name, key))
    
    # Track unmatched true plans
    for key, true_name in true_plan_details.items():
        if key not in opt_plan_details:
            unmatched_true_plans.append((true_name, key))
    
    print(f"\nFound {len(matched_plans)} matched plans")
    print(f"Unmatched optimizer plans: {len(unmatched_opt_plans)}")
    print(f"Unmatched true plans: {len(unmatched_true_plans)}")
    
    # List unmatched true plans for debugging
    print("\nUnmatched true plans:")
    for name, key in unmatched_true_plans:
        print(f"  {name}: Type={key[0]}, Number={key[1]}")
    
    # Print unmatched optimizer plans for debugging
    print("\nUnmatched optimizer plans:")
    for name, key in unmatched_opt_plans:
        print(f"  {name}: Type={key[0]}, Number={key[1]}")
    
    if not matched_plans:
        print("No matched plans found between optimizer rankings and true rankings")
        print("Example optimizer plans:", [p[0] for p in unmatched_opt_plans[:5]])
        print("Example true plans:", [p[0] for p in unmatched_true_plans[:5]])
        return
    
    # Print matched plans for verification
    print("\nMatched plans:")
    for opt_name, true_name in matched_plans:
        print(f"  {opt_name} <-> {true_name}")
    
    # Extract just the matched plans and get their rankings (position in the sorted list)
    matched_opt_plans = [p[0] for p in matched_plans]
    matched_true_plans = [p[1] for p in matched_plans]
    
    # Get win counts/scores and F1 scores for matched plans
    opt_scores = []
    f1_scores = []
    
    for opt_name, true_name in matched_plans:
        # Find score for this optimizer plan
        for name, score in optimizer_ordered_plans:
            if name == opt_name:
                opt_scores.append(score)
                break
        
        # Find F1 score for this true plan
        f1_scores.append(true_rankings[true_name])
    
    # Get ranks for matched plans
    opt_ranks = {}
    for i, (plan, _) in enumerate(optimizer_ordered_plans):
        opt_ranks[plan] = i + 1
    
    true_ranks = {}
    for i, (plan, _) in enumerate(true_ordered_plans):
        true_ranks[plan] = i + 1
    
    # Extract ranks for matched plans
    opt_rank_order = [opt_ranks[plan] for plan in matched_opt_plans]
    true_rank_order = [true_ranks[plan] for plan in matched_true_plans]
    
    # Extract raw scores for matched plans
    raw_opt_scores = []
    raw_f1_scores = []
    
    for opt_name, true_name in matched_plans:
        # Get optimizer score from original data
        for name, score in optimizer_ordered_plans:
            if name == opt_name:
                raw_opt_scores.append(score)
                break
        
        # Get F1 score from true rankings
        raw_f1_scores.append(true_rankings[true_name])
    
    # Create DataFrame for the rankings
    ranking_data = []
    for i, (opt_name, true_name) in enumerate(matched_plans):
        ranking_data.append({
            "Plan": true_name,
            "Optimizer Plan": opt_name,
            "Optimizer Score": opt_scores[i],
            "Optimizer Rank": opt_rank_order[i],
            "F1 Score": f1_scores[i],
            "F1 Rank": true_rank_order[i]
        })
    
    # Sort by F1 score
    ranking_df = pd.DataFrame(ranking_data)
    ranking_df = ranking_df.sort_values(by="F1 Score", ascending=False)
    ranking_df.to_csv(os.path.join(OUTPUT_DIR, "ranking_comparison.csv"), index=False)
    
    # Calculate correlation between optimizer rank and F1 rank
    rank_corr, rank_p = spearmanr(opt_rank_order, true_rank_order)
    
    # Calculate Kendall's tau for rank concordance
    kendall_corr, kendall_p = kendalltau(opt_rank_order, true_rank_order)
    
    # Calculate Kendall's tau using raw scores
    raw_score_kendall, raw_score_p = kendalltau(raw_opt_scores, raw_f1_scores)
    
    correlation_results = {
        "rank_correlation": rank_corr,
        "rank_p_value": rank_p,
        "kendall_correlation": kendall_corr,
        "kendall_p_value": kendall_p,
        "raw_score_kendall_correlation": raw_score_kendall,
        "raw_score_kendall_p_value": raw_score_p
    }
    
    with open(os.path.join(OUTPUT_DIR, "rank_correlation.json"), 'w') as f:
        json.dump(correlation_results, f, indent=2)
    
    # Output correlation results
    print(f"\nRank Correlation: {rank_corr:.3f} (p={rank_p:.3f})")
    print(f"Kendall's Tau (ranks): {kendall_corr:.3f} (p={kendall_p:.3f})")
    print(f"Kendall's Tau (raw scores): {raw_score_kendall:.3f} (p={raw_score_p:.3f})")
    
    # Create scatter plot of rankings
    plt.figure(figsize=(10, 8))
    plt.scatter(opt_rank_order, true_rank_order, alpha=0.7)
    
    # Add plan labels
    for i, plan in enumerate(matched_true_plans):
        plt.annotate(plan, (opt_rank_order[i], true_rank_order[i]), 
                    fontsize=10, alpha=0.8, xytext=(5, 5), 
                    textcoords='offset points')
    
    # Add line of perfect correlation
    min_rank = min(min(opt_rank_order), min(true_rank_order))
    max_rank = max(max(opt_rank_order), max(true_rank_order))
    plt.plot([min_rank, max_rank], [min_rank, max_rank], 'r--')
    
    # Add labels and title
    plt.xlabel('Optimizer Rank', fontsize=14)
    plt.ylabel('F1 Score Rank', fontsize=14)
    plt.title('Optimizer Ranks vs. F1 Score Ranks', fontsize=18)
    
    # Add correlation statistics to plot
    plt.annotate(f"Rank Correlation: {rank_corr:.3f} (p={rank_p:.3f})\nKendall's Tau (ranks): {kendall_corr:.3f} (p={kendall_p:.3f})\nKendall's Tau (raw scores): {raw_score_kendall:.3f} (p={raw_score_p:.3f})",
                xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ranking_correlation.png"), dpi=300)
    plt.close()

def create_visualizations(metrics_list, top_optimizer_plan=None):
    """
    Create visualizations of the plan performance metrics.
    
    Args:
        metrics_list: List of dictionaries containing metrics for each plan
        top_optimizer_plan: The name of the plan that the optimizer ranked first (optional)
    """
    # Create DataFrame from metrics
    df = pd.DataFrame([
        {
            "Plan": m["plan_name"],
            "Type": m["plan_name"].split("_")[0] if "_" in m["plan_name"] else "unknown",
            "Precision": m["avg_precision"],
            "Recall": m["avg_recall"],
            "F1": m["avg_f1"],
            "Runtime": m["runtime"],
            "Cost": m["cost"]
        }
        for m in metrics_list
    ])
    
    # Replace "parallel" with "isolation" for display purposes
    df["Type"] = df["Type"].replace("parallel", "isolation")
    
    # Sort by F1 score
    df = df.sort_values(by="F1", ascending=False)
    
    # Set plot style
    sns.set(style="whitegrid")
    
    # Function to find the equivalent plan in the results for the top optimizer plan
    def find_equivalent_plan(optimizer_plan_name):
        if optimizer_plan_name is None:
            return None
            
        # Extract type and number to match
        parts = optimizer_plan_name.split('_')
        if len(parts) >= 2:
            if parts[0] == "parallel" and parts[1] == "map":
                plan_type = "parallel"
            else:
                plan_type = parts[0]
                
            try:
                plan_number = int(parts[-1])
            except ValueError:
                try:
                    plan_number = int(parts[-2])
                except (ValueError, IndexError):
                    plan_number = 0
                    
            # Find matching plan in results
            for plan_name in df["Plan"]:
                plan_parts = plan_name.split('_')
                if len(plan_parts) >= 2:
                    result_type = plan_parts[0]
                    try:
                        result_number = int(plan_parts[-1])
                    except ValueError:
                        try:
                            result_number = int(plan_parts[-2])
                        except (ValueError, IndexError):
                            result_number = 0
                            
                    if result_type == plan_type and result_number == plan_number:
                        return plan_name
                        
        return None
    
    # Find the equivalent plan in our results dataset
    top_plan_in_results = find_equivalent_plan(top_optimizer_plan)
    if top_plan_in_results:
        print(f"Matched optimizer's top plan to: {top_plan_in_results}")
    
    # Reference benchmarks
    docetl_unopt = {"Precision": 0.341, "Recall": 0.430, "F1": 0.379}
    docetl_opt = {"Precision": 0.401, "Recall": 0.719, "F1": 0.477}
    
    # Set paper-quality figure parameters for consistent appearance
    plt.rcParams.update({
        'font.size': 16,
        'font.weight': 'bold',
        'axes.labelsize': 22,
        'axes.titlesize': 24,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'figure.titlesize': 28
    })
    
    # Bar chart of F1 scores by plan type with enhanced appearance for paper
    plt.figure(figsize=(16, 10))
    ax = sns.barplot(x="Plan", y="F1", hue="Type", data=df, palette="viridis", linewidth=1.5)
    
    # Add star to top optimizer plan if found - make it larger and more prominent
    if top_plan_in_results:
        plan_idx = df["Plan"].tolist().index(top_plan_in_results)
        top_plan_f1 = df.iloc[plan_idx]["F1"]
        ax.scatter(plan_idx, top_plan_f1, marker='*', s=500, color='red', 
                   edgecolor='black', linewidth=1.5, zorder=10, 
                   label="LLM judge's Top Pick")
    
    plt.title("Effectiveness of Rewrite Implementations by F1 Score", fontsize=24, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Implementation", fontsize=22, fontweight='bold')
    plt.ylabel("F1 Score", fontsize=22, fontweight='bold')
    plt.ylim(0, max(df["F1"]) * 1.1)
    
    # Enhanced legend
    plt.legend(
        title="Implementation Type", 
        loc="lower right", 
        fontsize=16, 
        title_fontsize=18,
        framealpha=0.9,
        edgecolor='gray', 
        fancybox=True,
        shadow=True
    )
    
    plt.grid(axis='y', linestyle='--', alpha=0.3, linewidth=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "f1_scores.png"), dpi=400, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    # Plot runtime vs. F1 score - enhanced for academic paper visibility
    plt.figure(figsize=(14, 10))
    
    # Create color mapping for consistency
    types = df["Type"].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(types)))
    type_color_dict = dict(zip(types, colors))
    
    # Plot points with consistent colors and larger size
    for plan_type in types:
        type_data = df[df["Type"] == plan_type]
        plt.scatter(type_data["Runtime"], type_data["F1"], 
                    color=type_color_dict[plan_type],
                    label=plan_type, alpha=0.8, s=180)
    
    # Add labels for each point with better visibility
    for i, row in df.iterrows():
        # Shorten name for better readability
        short_name = row["Plan"].split("_")[-1] if "_" in row["Plan"] else row["Plan"]
        plt.annotate(short_name, (row["Runtime"], row["F1"]), 
                     fontsize=14, fontweight='bold', alpha=0.9,
                     xytext=(5, 5), textcoords='offset points')

    # Add star to top optimizer plan if found
    if top_plan_in_results:
        top_plan_data = df[df["Plan"] == top_plan_in_results].iloc[0]
        plt.scatter(top_plan_data["Runtime"], top_plan_data["F1"], 
                   marker='*', s=600, color='red', edgecolor='black', linewidth=1.5,
                   zorder=10, label="LLM judge's Top Pick")
    
    # Add DocETL benchmark with enhanced visibility
    unopt_line = plt.axhline(y=docetl_unopt["F1"], color='red', linestyle='--', linewidth=2.5, alpha=0.8)
    opt_line = plt.axhline(y=docetl_opt["F1"], color='blue', linestyle='-', linewidth=3, alpha=0.8)
    
    # Add text in better position
    text_x_position = df["Runtime"].min() + (df["Runtime"].max() - df["Runtime"].min()) * 0.05  # 5% from left
    plt.text(text_x_position, docetl_unopt["F1"] + 0.01, "DocETL (Unopt.)", 
             fontsize=14, fontweight='bold', color='red')
    plt.text(text_x_position, docetl_opt["F1"] + 0.01, "DocETL (Opt.)", 
             fontsize=14, fontweight='bold', color='blue')
    
    # Improve axis limits to reduce whitespace
    plt.xlim(max(0, df["Runtime"].min() * 0.9), df["Runtime"].max() * 1.1)
    plt.ylim(max(0, df["F1"].min() - 0.05), min(1.0, df["F1"].max() + 0.1))
    
    plt.xlabel("Runtime (seconds)", fontsize=22, fontweight='bold')
    plt.ylabel("F1 Score", fontsize=22, fontweight='bold')
    plt.title("Runtime Performance of Rewrite Implementations", fontsize=24, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Enhanced legend
    plt.legend(
        title="Implementation Type", 
        loc="lower right", 
        fontsize=16, 
        title_fontsize=18,
        framealpha=0.9,
        edgecolor='gray', 
        fancybox=True,
        shadow=True
    )
    
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "runtime_vs_f1.png"), dpi=400, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    # Create individual cost plots (enhanced for paper visibility)
    
    # Plot F1 vs. Cost
    plt.figure(figsize=(14, 10))
    
    # Plot points with consistent colors and larger size
    for plan_type in types:
        type_data = df[df["Type"] == plan_type]
        plt.scatter(type_data["Cost"], type_data["F1"], 
                    color=type_color_dict[plan_type],
                    label=plan_type, alpha=0.8, s=180)
    
    # Add labels for each point with better visibility
    for i, row in df.iterrows():
        # Shorten name for better readability
        short_name = row["Plan"].split("_")[-1] if "_" in row["Plan"] else row["Plan"]
        plt.annotate(short_name, (row["Cost"], row["F1"]), 
                     fontsize=14, fontweight='bold', alpha=0.9,
                     xytext=(5, 5), textcoords='offset points')
    
    # Add star to top optimizer plan if found
    if top_plan_in_results:
        top_plan_data = df[df["Plan"] == top_plan_in_results].iloc[0]
        plt.scatter(top_plan_data["Cost"], top_plan_data["F1"], 
                   marker='*', s=600, color='red', edgecolor='black', linewidth=1.5,
                   zorder=10, label="LLM judge's Top Pick")
    
    # Add DocETL benchmark with enhanced visibility
    unopt_line = plt.axhline(y=docetl_unopt["F1"], color='red', linestyle='--', linewidth=2.5, alpha=0.8)
    opt_line = plt.axhline(y=docetl_opt["F1"], color='blue', linestyle='-', linewidth=3, alpha=0.8)
    
    # Add text in better position
    text_x_position = df["Cost"].min() + (df["Cost"].max() - df["Cost"].min()) * 0.05  # 5% from left
    plt.text(text_x_position, docetl_unopt["F1"] + 0.01, "DocETL (Unopt.)", 
             fontsize=14, fontweight='bold', color='red')
    plt.text(text_x_position, docetl_opt["F1"] + 0.01, "DocETL (Opt.)", 
             fontsize=14, fontweight='bold', color='blue')
    
    # Improve axis limits to reduce whitespace
    plt.xlim(max(0, df["Cost"].min() * 0.9), df["Cost"].max() * 1.1)
    plt.ylim(max(0, df["F1"].min() - 0.05), min(1.0, df["F1"].max() + 0.1))
    
    plt.xlabel("API Cost ($)", fontsize=22, fontweight='bold')
    plt.ylabel("F1 Score", fontsize=22, fontweight='bold')
    plt.title("Effectiveness of Rewrite Implementations by F1 Score", fontsize=24, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Enhanced legend
    plt.legend(
        title="Implementation Type", 
        loc="lower right", 
        fontsize=16, 
        title_fontsize=18,
        framealpha=0.9,
        edgecolor='gray', 
        fancybox=True,
        shadow=True
    )
    
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cost_vs_f1.png"), dpi=400, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    # Plot Precision vs. Cost (New) with enhanced appearance
    plt.figure(figsize=(14, 10))
    
    # Plot points with consistent colors and larger size
    for plan_type in types:
        type_data = df[df["Type"] == plan_type]
        plt.scatter(type_data["Cost"], type_data["Precision"], 
                    color=type_color_dict[plan_type],
                    label=plan_type, alpha=0.8, s=180)
    
    # Add labels for each point with better visibility
    for i, row in df.iterrows():
        # Shorten name for better readability
        short_name = row["Plan"].split("_")[-1] if "_" in row["Plan"] else row["Plan"]
        plt.annotate(short_name, (row["Cost"], row["Precision"]), 
                     fontsize=14, fontweight='bold', alpha=0.9,
                     xytext=(5, 5), textcoords='offset points')
    
    # Add star to top optimizer plan if found
    if top_plan_in_results:
        top_plan_data = df[df["Plan"] == top_plan_in_results].iloc[0]
        plt.scatter(top_plan_data["Cost"], top_plan_data["Precision"], 
                   marker='*', s=600, color='red', edgecolor='black', linewidth=1.5,
                   zorder=10, label="LLM judge's Top Pick")
    
    # Add DocETL benchmark with enhanced visibility
    unopt_line = plt.axhline(y=docetl_unopt["Precision"], color='red', linestyle='--', linewidth=2.5, alpha=0.8)
    opt_line = plt.axhline(y=docetl_opt["Precision"], color='blue', linestyle='-', linewidth=3, alpha=0.8)
    
    # Add text in better position
    text_x_position = df["Cost"].min() + (df["Cost"].max() - df["Cost"].min()) * 0.05  # 5% from left
    plt.text(text_x_position, docetl_unopt["Precision"] + 0.01, "DocETL (Unopt.)", 
             fontsize=14, fontweight='bold', color='red')
    plt.text(text_x_position, docetl_opt["Precision"] + 0.01, "DocETL (Opt.)", 
             fontsize=14, fontweight='bold', color='blue')
    
    # Improve axis limits to reduce whitespace
    plt.xlim(max(0, df["Cost"].min() * 0.9), df["Cost"].max() * 1.1)
    plt.ylim(max(0, df["Precision"].min() - 0.05), min(1.0, df["Precision"].max() + 0.1))
    
    plt.xlabel("API Cost ($)", fontsize=22, fontweight='bold')
    plt.ylabel("Precision", fontsize=22, fontweight='bold')
    plt.title("Effectiveness of Rewrite Implementations by Precision", fontsize=24, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Enhanced legend
    plt.legend(
        title="Implementation Type", 
        loc="lower right", 
        fontsize=16, 
        title_fontsize=18,
        framealpha=0.9,
        edgecolor='gray', 
        fancybox=True,
        shadow=True
    )
    
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cost_vs_precision.png"), dpi=400, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    # Plot Recall vs. Cost with enhanced appearance
    plt.figure(figsize=(14, 10))
    
    # Plot points with consistent colors and larger size
    for plan_type in types:
        type_data = df[df["Type"] == plan_type]
        plt.scatter(type_data["Cost"], type_data["Recall"], 
                    color=type_color_dict[plan_type],
                    label=plan_type, alpha=0.8, s=180)
    
    # Add labels for each point with better visibility
    for i, row in df.iterrows():
        # Shorten name for better readability
        short_name = row["Plan"].split("_")[-1] if "_" in row["Plan"] else row["Plan"]
        plt.annotate(short_name, (row["Cost"], row["Recall"]), 
                     fontsize=14, fontweight='bold', alpha=0.9,
                     xytext=(5, 5), textcoords='offset points')
    
    # Add star to top optimizer plan if found
    if top_plan_in_results:
        top_plan_data = df[df["Plan"] == top_plan_in_results].iloc[0]
        plt.scatter(top_plan_data["Cost"], top_plan_data["Recall"], 
                   marker='*', s=600, color='red', edgecolor='black', linewidth=1.5,
                   zorder=10, label="LLM judge's Top Pick")
    
    # Add DocETL benchmark with enhanced visibility
    unopt_line = plt.axhline(y=docetl_unopt["Recall"], color='red', linestyle='--', linewidth=2.5, alpha=0.8)
    opt_line = plt.axhline(y=docetl_opt["Recall"], color='blue', linestyle='-', linewidth=3, alpha=0.8)
    
    # Add text in better position
    text_x_position = df["Cost"].min() + (df["Cost"].max() - df["Cost"].min()) * 0.05  # 5% from left
    plt.text(text_x_position, docetl_unopt["Recall"] + 0.01, "DocETL (Unopt.)", 
             fontsize=14, fontweight='bold', color='red')
    plt.text(text_x_position, docetl_opt["Recall"] + 0.01, "DocETL (Opt.)", 
             fontsize=14, fontweight='bold', color='blue')
    
    # Improve axis limits to reduce whitespace
    plt.xlim(max(0, df["Cost"].min() * 0.9), df["Cost"].max() * 1.1)
    plt.ylim(max(0, df["Recall"].min() - 0.05), min(1.0, df["Recall"].max() + 0.1))
    
    plt.xlabel("API Cost ($)", fontsize=22, fontweight='bold')
    plt.ylabel("Recall", fontsize=22, fontweight='bold')
    plt.title("Effectiveness of Rewrite Implementations by Recall", fontsize=24, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Enhanced legend
    plt.legend(
        title="Implementation Type", 
        loc="lower right", 
        fontsize=16, 
        title_fontsize=18,
        framealpha=0.9,
        edgecolor='gray', 
        fancybox=True,
        shadow=True
    )
    
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cost_vs_recall.png"), dpi=400, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    # Create combined plot for all three metrics vs cost (for academic paper)
    create_combined_cost_plots(df, type_color_dict, types, docetl_unopt, docetl_opt, top_plan_in_results)
    
    # Create a metrics by type distribution plot
    plt.figure(figsize=(16, 10))
    
    # Reshape data for plotting distributions
    plot_data = []
    for _, row in df.iterrows():
        plot_data.append({
            "Plan Type": row["Type"],
            "Metric": "Precision",
            "Value": row["Precision"],
            "Plan": row["Plan"]
        })
        plot_data.append({
            "Plan Type": row["Type"],
            "Metric": "Recall",
            "Value": row["Recall"],
            "Plan": row["Plan"]
        })
        plot_data.append({
            "Plan Type": row["Type"],
            "Metric": "F1",
            "Value": row["F1"],
            "Plan": row["Plan"]
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Set paper-quality figure parameters
    plt.rcParams.update({
        'font.size': 16,
        'font.weight': 'bold',
        'axes.labelsize': 22,
        'axes.titlesize': 24,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'figure.titlesize': 28,
        'figure.constrained_layout.use': True
    })
    
    # Create subplot grid - one row, three columns with minimal spacing
    fig, axes = plt.subplots(1, 3, figsize=(20, 10), sharey=False)
    # fig.suptitle("Distribution of Metrics by Implementation Type", fontsize=28, fontweight='bold', y=0.98)
    
    
    # For academic paper readability, define large font sizes
    TITLE_SIZE = 24
    AXIS_LABEL_SIZE = 22 
    TICK_SIZE = 18
    LEGEND_SIZE = 18
    BENCHMARK_FONT_SIZE = 16
    
    metrics = ["Precision", "Recall", "F1"]
    
    for i, metric in enumerate(metrics):
        # Filter data for this metric
        metric_data = plot_df[plot_df["Metric"] == metric]
        
        # Create boxplot with points overlay - make boxes more prominent
        sns.boxplot(x="Plan Type", y="Value", data=metric_data, ax=axes[i], 
                   palette="viridis", linewidth=2.5, width=0.6)
        
        # Add strip plot points - make them larger
        ax_stripplot = sns.stripplot(x="Plan Type", y="Value", data=metric_data, 
                      ax=axes[i], color="black", alpha=0.6, jitter=True, 
                      size=10, edgecolor='gray', linewidth=0.5)
        
        # Add star to top optimizer plan if found
        if top_plan_in_results:
            top_plan_row = metric_data[metric_data["Plan"] == top_plan_in_results]
            if not top_plan_row.empty:
                top_plan_type = top_plan_row["Plan Type"].iloc[0]
                top_plan_value = top_plan_row["Value"].iloc[0]
                
                # Find the position in the stripplot
                type_pos = list(metric_data["Plan Type"].unique()).index(top_plan_type)
                
                # Add star - make it larger and more prominent
                axes[i].scatter(type_pos, top_plan_value, marker='*', s=500, 
                          color='red', edgecolor='black', linewidth=1.5, zorder=10)
        
        # Add DocETL benchmark lines with clearly distinguishable styles
        # Unoptimized: dashed red line
        unopt_line = axes[i].axhline(y=docetl_unopt[metric], color='red', 
                       linestyle='--', linewidth=2.5, alpha=0.8, 
                       label="DocETL (Unopt.)")
        
        # Optimized: solid blue line
        opt_line = axes[i].axhline(y=docetl_opt[metric], color='blue', 
                     linestyle='-', linewidth=3, alpha=0.8,
                     label="DocETL (Opt.)")
        
        # Add bold, larger labels
        axes[i].set_title(f"{metric}", fontsize=TITLE_SIZE, fontweight='bold')
        axes[i].set_xlabel("Implementation Type", fontsize=AXIS_LABEL_SIZE, fontweight='bold')
        if i == 0:  # Only add y-label to first plot
            axes[i].set_ylabel("Score", fontsize=AXIS_LABEL_SIZE, fontweight='bold')
        else:
            axes[i].set_ylabel("", fontsize=0)  # Hide other y-labels
            
        # Increase tick size
        axes[i].tick_params(axis='both', labelsize=TICK_SIZE, width=1.5, length=6)
        axes[i].grid(True, linestyle='--', linewidth=0.8, alpha=0.3)
        
        # Improve y-axis limits to reduce unnecessary whitespace
        y_values = metric_data["Value"].dropna()
        if len(y_values) > 0:
            y_min = max(0, y_values.min() - 0.05)  # Add small padding, but never go below 0
            y_max = min(1.0, y_values.max() + 0.1)  # Add padding, but never exceed 1.0
            axes[i].set_ylim(y_min, y_max)
            
        # Add large, high-contrast text labels for benchmark lines
        if i == 0:  # Only add text to first plot to avoid crowding
            # Calculate a better position for the text
            y_range = axes[i].get_ylim()
            text_offset = (y_range[1] - y_range[0]) * 0.03  # Small vertical offset
            
            axes[i].text(
                -0.15, docetl_unopt[metric] + text_offset, 
                "DocETL (Unopt.)", 
                fontsize=BENCHMARK_FONT_SIZE, fontweight='bold', ha='right', va='bottom', 
                color='red', transform=axes[i].get_yaxis_transform()
            )
            axes[i].text(
                -0.15, docetl_opt[metric] + text_offset, 
                "DocETL (Opt.)", 
                fontsize=BENCHMARK_FONT_SIZE, fontweight='bold', ha='right', va='bottom',
                color='blue', transform=axes[i].get_yaxis_transform()
            )
    
    # Create a compact and visible legend outside the plots
    all_handles = []
    all_labels = []
    
    # Get handles for implementation types (from the boxes)
    box_handles, box_labels = axes[0].get_legend_handles_labels()
    for handle, label in zip(box_handles, box_labels):
        if label not in all_labels:  # Avoid duplicates
            all_handles.append(handle)
            all_labels.append(label)
    
    # Add star for optimizer's top pick
    if top_plan_in_results:
        star_patch = plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                             markersize=18, markeredgecolor='black', markeredgewidth=1.5,
                             label="LLM judge's Top Pick")
        all_handles.append(star_patch)
        all_labels.append("LLM judge's Top Pick")
    
    # Add manual entries for DocETL lines
    unopt_line = plt.Line2D([0], [1], color='red', linestyle='--', linewidth=2.5, alpha=0.8)
    opt_line = plt.Line2D([0], [1], color='blue', linestyle='-', linewidth=3, alpha=0.8)
    all_handles.extend([unopt_line, opt_line])
    all_labels.extend(["DocETL (Unopt.)", "DocETL (Opt.)"])
    
    # Place a more compact legend at the bottom 
    fig.legend(
        all_handles, all_labels,
        loc='lower center', 
        bbox_to_anchor=(0.5, 0.00),  # Position right at the bottom
        ncol=len(all_handles),  # Put all items in one row to save vertical space
        fontsize=LEGEND_SIZE,
        frameon=True,
        fancybox=True,
        shadow=True,
        labelspacing=1.0,
        handletextpad=0.5,
        borderpad=0.5
    )
    
    # Adjust layout to maximize space usage and minimize whitespace
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.10, wspace=0.15)  # Reduce space between subplots
    
    # Save with high DPI and tight bounding box
    plt.savefig(os.path.join(OUTPUT_DIR, "metrics_distribution_by_type.png"), 
               dpi=400, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    # Reset rcParams to default for other plots
    plt.rcParams.update(plt.rcParamsDefault)

def create_combined_cost_plots(df, type_color_dict, types, docetl_unopt, docetl_opt, top_plan_in_results):
    """
    Create a combined figure with three cost-related plots in a row (Precision, Recall, F1).
    
    Args:
        df: DataFrame containing all metrics data
        type_color_dict: Dictionary mapping plan types to colors for consistency
        types: Unique plan types
        docetl_unopt: DocETL unoptimized benchmark values
        docetl_opt: DocETL optimized benchmark values
        top_plan_in_results: The name of the top ranked plan in results dataset
    """
    # Set paper-quality figure parameters
    plt.rcParams.update({
        'font.size': 16,
        'font.weight': 'bold',
        'axes.labelsize': 22,
        'axes.titlesize': 26,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'figure.titlesize': 28
    })
    
    # Create a figure with 3 subplots in a row
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), constrained_layout=True)
    # fig.suptitle("Effectiveness of LLM Implementations of Rewrite Directives", fontsize=28, fontweight='bold', y=1.02)
    
    # Define metrics and titles - focus on assessment rather than cost trade-offs
    metrics = ["Precision", "Recall", "F1"]
    titles = metrics


    # For academic paper readability, increase all font sizes
    TITLE_SIZE = 24
    AXIS_LABEL_SIZE = 22
    TICK_SIZE = 18
    LEGEND_SIZE = 20
    ANNOTATION_SIZE = 14
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        
        # Plot points with consistent colors
        for plan_type in types:
            type_data = df[df["Type"] == plan_type]
            ax.scatter(type_data["Cost"], type_data[metric], 
                      color=type_color_dict[plan_type],
                      label=plan_type, alpha=0.8, s=180)
        
        # Add labels for each point - use smaller font and shorter names for readability
        for _, row in df.iterrows():
            # Extract short names for better visibility
            short_name = row["Plan"].split("_")[-1] if "_" in row["Plan"] else row["Plan"]
            ax.annotate(short_name, (row["Cost"], row[metric]), 
                       fontsize=ANNOTATION_SIZE, fontweight='bold', alpha=0.9,
                       xytext=(5, 5), textcoords='offset points')
        
        # Add star to top optimizer plan if found
        if top_plan_in_results and top_plan_in_results in df["Plan"].values:
            top_plan_data = df[df["Plan"] == top_plan_in_results].iloc[0]
            ax.scatter(top_plan_data["Cost"], top_plan_data[metric], 
                     marker='*', s=600, color='red', edgecolor='black', linewidth=1.5,
                     zorder=10, label="LLM judge's Top Pick" if i == 0 else "")
        
        # Add DocETL benchmark
        unopt_line = ax.axhline(y=docetl_unopt[metric], color='red', linestyle='--', 
                              linewidth=2.5, alpha=0.8, label="DocETL (Unopt.)" if i == 0 else "")
        opt_line = ax.axhline(y=docetl_opt[metric], color='blue', linestyle='-', 
                            linewidth=3, alpha=0.8, label="DocETL (Opt.)" if i == 0 else "")
        
        # Add benchmark text in better positions
        text_x_position = df["Cost"].max() * 0.05  # Position near left side
        
        # Only add benchmark text to the first plot to avoid clutter
        if i == 0:
            ax.text(text_x_position, docetl_unopt[metric] + 0.01, 
                   "DocETL (Unopt.)", fontsize=ANNOTATION_SIZE, fontweight='bold',
                   color='red', ha='left', va='bottom')
            ax.text(text_x_position, docetl_opt[metric] + 0.01, 
                   "DocETL (Opt.)", fontsize=ANNOTATION_SIZE, fontweight='bold',
                   color='blue', ha='left', va='bottom')
        
        # Add labels and title with larger fonts
        ax.set_xlabel("API Cost ($)", fontsize=AXIS_LABEL_SIZE, fontweight='bold')
        ax.set_ylabel(metric, fontsize=AXIS_LABEL_SIZE, fontweight='bold')
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')
        
        # Improve axes appearance
        ax.tick_params(axis='both', labelsize=TICK_SIZE, width=1.5, length=6)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.0)
        
        # Improve axis limits to reduce whitespace
        y_values = df[metric].dropna()
        if len(y_values) > 0:
            y_min = max(0, y_values.min() - 0.05)
            y_max = min(1.0, y_values.max() + 0.1)
            ax.set_ylim(y_min, y_max)
        
        # Determine cost limits
        cost_values = df["Cost"].dropna()
        if len(cost_values) > 0:
            x_min = max(0, cost_values.min() * 0.9)
            x_max = cost_values.max() * 1.1
            ax.set_xlim(x_min, x_max)
        
        # Only add the legend to the first plot
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            
            # Ensure no duplicates in legend
            unique_labels = []
            unique_handles = []
            for handle, label in zip(handles, labels):
                if label not in unique_labels:
                    unique_labels.append(label)
                    unique_handles.append(handle)
            
            ax.legend(handles=unique_handles, labels=unique_labels, 
                     title="Plan Type", title_fontsize=LEGEND_SIZE,
                     loc="lower right", fontsize=LEGEND_SIZE - 2,
                     framealpha=0.9, edgecolor='gray', fancybox=True)
    
    # Adjust layout to maximize space
    plt.tight_layout()
    
    # Save with high quality
    plt.savefig(os.path.join(OUTPUT_DIR, "cost_vs_metrics_combined.pdf"), 
               bbox_inches='tight')
    plt.close()
    
    # Reset rcParams to default for other plots
    plt.rcParams.update(plt.rcParamsDefault)

if __name__ == "__main__":
    comparison_df, metrics_list = evaluate_all_plans()
    
    # Print top 5 plans
    print("\nTop 5 Plans by F1 Score:")
    print(comparison_df.head(5).to_string(index=False))
    
    # Print bottom 5 plans
    print("\nBottom 5 Plans by F1 Score:")
    print(comparison_df.tail(5).to_string(index=False))
    
    # Print average metrics by plan type
    plan_types = comparison_df["Plan"].apply(lambda x: x.split("_")[0] if "_" in x else "unknown")
    by_type = pd.concat([comparison_df, pd.Series(plan_types, name="Type")], axis=1)
    
    # Explicitly select only numeric columns for averaging
    numeric_columns = ["Avg Precision", "Avg Recall", "F1 Score", "Runtime (s)", "API Cost ($)"]
    type_averages = by_type.groupby("Type")[numeric_columns].mean()
    
    print("\nAverage Metrics by Plan Type:")
    print(type_averages.to_string())
    
    print(f"\nResults saved to {OUTPUT_DIR}") 