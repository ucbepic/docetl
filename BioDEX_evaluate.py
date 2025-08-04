import argparse
import json
import os
import time
from functools import partial

import datasets


# Specify target pmids
target_pmids = [
    "34685674", "34040174", "33206300", "34631085", "34382462",
    "34066952", "34176836", "33463607", "34377045", "34646840"
]

seed=42
test_dataset = datasets.load_dataset("BioDEX/BioDEX-Reactions", split="test").to_pandas()

# Filter rows with specified pmids
test_dataset = test_dataset[test_dataset["pmid"].isin(target_pmids)].to_dict(orient="records")

def compute_target_record(entry):
    reactions_lst = [
        reaction.strip().lower().replace("'", "").replace("^", "")
        for reaction in entry["reactions"].split(",")
    ]
    label_dict = {"ranked_reaction_labels": reactions_lst}
    return label_dict

label_fields_to_values = {
    entry["pmid"]: compute_target_record(entry) for entry in test_dataset
}

def extract_predictions(json_file_path):
    """
    Extract pmid and ranked_reaction_labels from prediction JSON file
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        predictions_data = json.load(f)
    
    # Extract required fields
    predictions = {}
    for entry in predictions_data:
        pmid = entry["pmid"]
        ranked_reaction_labels = entry["ranked_reaction_labels"]
        predictions[pmid] = {"ranked_reaction_labels": ranked_reaction_labels}
    
    return predictions

# Load predictions
json_file_path = "/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/BioDEX/output/orig_plan.json"  
predictions = extract_predictions(json_file_path)

# Evaluation functions from the original code
def rank_precision_at_k(preds: list, targets: list, k: int):
    if preds is None:
        return 0.0

    try:
        # lower-case each list
        preds = [pred.strip().lower().replace("'", "").replace("^", "") for pred in preds]
        targets = set([target.strip().lower().replace("'", "").replace("^", "") for target in targets])

        # compute rank-precision at k
        rn = len(targets)
        denom = min(k, rn)
        total = 0.0
        for i in range(k):
            total += preds[i] in targets if i < len(preds) else 0.0

        return total / denom

    except Exception as e:
        print(f"Error in rank_precision_at_k: {e}")
        return 0.0

def term_recall(preds: list, targets: list):
    if preds is None:
        return 0.0

    try:
        # normalize terms in each list
        pred_terms = set([
            term.strip()
            for pred in preds
            for term in pred.lower().replace("'", "").replace("^", "").split(" ")
        ])
        target_terms = set([
            term.strip()
            for target in targets
            for term in target.lower().replace("'", "").replace("^", "").split(" ")
        ])

        # compute term recall and return
        intersect = pred_terms.intersection(target_terms)
        if len(target_terms) == 0:
            return 0.0
        term_recall = len(intersect) / len(target_terms)

        return term_recall

    except Exception as e:
        print(f"Error in term_recall: {e}")
        return 0.0

def compute_avg_rp_at_k(predictions, label_fields_to_values, k=5):
    total_rp_at_k = 0
    bad = 0
    valid_count = 0
    
    for pmid in predictions:
        if pmid in label_fields_to_values:
            preds = predictions[pmid]['ranked_reaction_labels']
            targets = label_fields_to_values[pmid]['ranked_reaction_labels']
            try:
                rp_score = rank_precision_at_k(preds, targets, k)
                total_rp_at_k += rp_score
                valid_count += 1
            except Exception as e:
                print(f"Error evaluating PMID {pmid}: {e}")
                bad += 1
        else:
            print(f"PMID {pmid} not found in ground truth")
            bad += 1

    avg_rp_at_k = total_rp_at_k / valid_count if valid_count > 0 else 0.0
    return avg_rp_at_k, bad, valid_count

def compute_avg_term_recall(predictions, label_fields_to_values):
    total_term_recall = 0
    bad = 0
    valid_count = 0
    
    for pmid in predictions:
        if pmid in label_fields_to_values:
            preds = predictions[pmid]['ranked_reaction_labels']
            targets = label_fields_to_values[pmid]['ranked_reaction_labels']
            try:
                tr_score = term_recall(preds, targets)
                total_term_recall += tr_score
                valid_count += 1
            except Exception as e:
                print(f"Error evaluating term recall for PMID {pmid}: {e}")
                bad += 1
        else:
            print(f"PMID {pmid} not found in ground truth")
            bad += 1

    avg_term_recall = total_term_recall / valid_count if valid_count > 0 else 0.0
    return avg_term_recall, bad, valid_count

# Run evaluation
print("Evaluating predictions...")
print("="*50)

# Rank Precision @ K evaluation
rp_at_k, bad_rp, valid_rp = compute_avg_rp_at_k(predictions, label_fields_to_values, k=5)
print(f"Rank Precision @ 5: {rp_at_k:.5f}")
print(f"Valid predictions for RP@5: {valid_rp}")
print(f"Bad predictions for RP@5: {bad_rp}")
print()

# Term Recall evaluation
tr, bad_tr, valid_tr = compute_avg_term_recall(predictions, label_fields_to_values)
print(f"Term Recall: {tr:.5f}")
print(f"Valid predictions for Term Recall: {valid_tr}")
print(f"Bad predictions for Term Recall: {bad_tr}")
print()

# # Print detailed comparison for each example
# print("Detailed per-example results:")
# print("="*50)

# for pmid in predictions:
#     if pmid in label_fields_to_values:
#         preds = predictions[pmid]['ranked_reaction_labels']
#         targets = label_fields_to_values[pmid]['ranked_reaction_labels']
        
#         rp_score = rank_precision_at_k(preds, targets, 5)
#         tr_score = term_recall(preds, targets)
        
#         print(f"PMID: {pmid}")
#         print(f"  RP@5: {rp_score:.3f}, Term Recall: {tr_score:.3f}")
#         print(f"  Ground Truth ({len(targets)}): {targets}")
#         print(f"  Prediction ({len(preds)}): {preds}")
#         print()