import json
from typing import Dict, Any, List


def rank_precision_at_k(preds: List[str], targets: List[str], k: int) -> float:
    """
    Calculate rank precision at k for reaction predictions.
    
    Args:
        preds: List of predicted reactions
        targets: List of ground truth reactions
        k: Top k predictions to consider
        
    Returns:
        Rank precision at k score
    """
    if not preds:
        return 0.0

    try:
        # Normalize: lower-case and remove special characters
        preds = [pred.strip().lower().replace("'", "").replace("^", "") for pred in preds]
        targets = set([target.strip().lower().replace("'", "").replace("^", "") for target in targets])

        # Compute rank-precision at k
        rn = len(targets)
        denom = min(k, rn)
        if denom == 0:
            return 0.0
            
        total = 0.0
        for i in range(min(k, len(preds))):
            if preds[i] in targets:
                total += 1.0

        return total / denom

    except Exception as e:
        print(f"Error in rank_precision_at_k: {e}")
        return 0.0


def term_recall(preds: List[str], targets: List[str]) -> float:
    """
    Calculate term recall for reaction predictions.
    
    Args:
        preds: List of predicted reactions
        targets: List of ground truth reactions
        
    Returns:
        Term recall score
    """
    if not preds:
        return 0.0

    try:
        # Normalize terms in each list
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

        # Compute term recall
        if len(target_terms) == 0:
            return 0.0
        
        intersect = pred_terms.intersection(target_terms)
        return len(intersect) / len(target_terms)

    except Exception as e:
        print(f"Error in term_recall: {e}")
        return 0.0


def evaluate_results(method_name: str, results_file: str, ground_truth_file: str = None, original_json_file: str = None) -> Dict[str, Any]:
    """
    Evaluate BioDEX results against ground truth.
    
    Args:
        method_name: Name of the method being evaluated
        results_file: Path to the results JSON file
        ground_truth_file: Not used - ground truth is embedded in results
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Load results (which already contain ground truth)
    with open(results_file, 'r') as f:
        results = json.load(f)

    if original_json_file:
        with open(original_json_file, "r") as f:
            original_json_content = json.load(f)
        
        total_orig_docs = len(original_json_content)
    else:
        total_orig_docs = 0
    
    # Evaluation metrics
    total_docs = 0
    rp_at_5_scores = []
    rp_at_10_scores = []
    term_recall_scores = []
    
    # Process results
    for result in results:
        if isinstance(result, dict):
            # Ground truth is already in the document
            ground_truth = result.get("ground_truth_reactions", [])
            predictions = result.get("ranked_reaction_labels", [])
            
            if not ground_truth:  # Skip if no ground truth
                continue
                
            total_docs += 1
            
            # Calculate metrics
            rp5 = rank_precision_at_k(predictions, ground_truth, 5)
            rp10 = rank_precision_at_k(predictions, ground_truth, 10)
            tr = term_recall(predictions, ground_truth)
            
            rp_at_5_scores.append(rp5)
            rp_at_10_scores.append(rp10)
            term_recall_scores.append(tr)
    
    # Calculate averages
    avg_rp_at_5 = sum(rp_at_5_scores) / len(rp_at_5_scores) if rp_at_5_scores else 0.0
    avg_rp_at_10 = sum(rp_at_10_scores) / len(rp_at_10_scores) if rp_at_10_scores else 0.0
    avg_term_recall = sum(term_recall_scores) / len(term_recall_scores) if term_recall_scores else 0.0
    
    return {
        "method": method_name,
        "total_documents": total_docs,
        "avg_rp_at_5": avg_rp_at_5,
        "avg_rp_at_10": avg_rp_at_10,
        "avg_term_recall": avg_term_recall,
        "rp_at_5_scores": rp_at_5_scores,
        "rp_at_10_scores": rp_at_10_scores,
        "term_recall_scores": term_recall_scores
    }