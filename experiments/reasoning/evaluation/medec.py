import json
import pandas as pd
from typing import Dict, Any, List


def jaccard_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity between two texts based on words (lowercased)."""
    if not text1 or not text2:
        return 0.0
    
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if len(union) == 0:
        return 0.0
    
    return len(intersection) / len(union)


def evaluate_results(method_name: str, results_file: str, ground_truth_file: str = None, original_json_file: str = None) -> Dict[str, Any]:
    """
    Evaluate MEDEC results against ground truth.
    
    Args:
        method_name: Name of the method being evaluated
        results_file: Path to the results JSON file
        ground_truth_file: Not used - ground truth is in the results data itself
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Ground truth is embedded in the results data itself
    # Create a mapping from Text ID to ground truth from the results
    gt_mapping = {}
    for result in results:
        if isinstance(result, dict) and "Text ID" in result:
            text_id = result["Text ID"]
            
            # Extract ground truth from the result data
            # Handle both string and numeric error flags
            error_flag = result.get("Error Flag", 0)
            if isinstance(error_flag, str):
                error_flag = int(error_flag) if error_flag.isdigit() else 0
            
            gt_mapping[text_id] = {
                "error_flag": bool(error_flag),
                "error_sentence": result.get("Error Sentence", "") or "",
                "corrected_sentence": result.get("Corrected Sentence", "") or ""
            }
    
    # Evaluation metrics
    total_cases = 0
    correct_error_predictions = 0
    error_sentence_similarities = []
    corrected_sentence_similarities = []
    
    # Process results
    for result in results:
        if isinstance(result, dict) and "Text ID" in result:
            text_id = result.get("Text ID")
            if text_id not in gt_mapping:
                continue
                
            total_cases += 1
            gt = gt_mapping[text_id]
            
            # Extract predictions directly from the result
            predicted_error = result.get("is_error", False)
            predicted_error_sentence = result.get("error_sentence", "")
            predicted_corrected_sentence = result.get("corrected_sentence", "")
            
            # Convert ground truth error flag to boolean
            gt_error = bool(gt["error_flag"])
            
            # Accuracy for error flag prediction
            if predicted_error == gt_error:
                correct_error_predictions += 1
            
            # Jaccard similarity for error and corrected sentences (only when GT has error)
            if gt_error and gt["error_sentence"]:
                error_sim = jaccard_similarity(predicted_error_sentence, gt["error_sentence"])
                error_sentence_similarities.append(error_sim)
                
                if gt["corrected_sentence"]:
                    corrected_sim = jaccard_similarity(predicted_corrected_sentence, gt["corrected_sentence"])
                    corrected_sentence_similarities.append(corrected_sim)
    
    # Calculate final metrics
    error_flag_accuracy = correct_error_predictions / total_cases if total_cases > 0 else 0.0
    avg_error_sentence_jaccard = sum(error_sentence_similarities) / len(error_sentence_similarities) if error_sentence_similarities else 0.0
    avg_corrected_sentence_jaccard = sum(corrected_sentence_similarities) / len(corrected_sentence_similarities) if corrected_sentence_similarities else 0.0
    
    # Combined score: weighted average of error flag accuracy and sentence similarities
    # Weight: 50% error flag accuracy, 25% error sentence jaccard, 25% corrected sentence jaccard
    combined_score = (0.5 * error_flag_accuracy + 
                     0.25 * avg_error_sentence_jaccard + 
                     0.25 * avg_corrected_sentence_jaccard)
    
    return {
        "total_cases": total_cases,
        "error_flag_accuracy": error_flag_accuracy,
        "avg_error_sentence_jaccard": avg_error_sentence_jaccard,
        "avg_corrected_sentence_jaccard": avg_corrected_sentence_jaccard,
        "combined_score": combined_score,
        "num_error_cases": len(error_sentence_similarities),
        "num_corrected_cases": len(corrected_sentence_similarities)
    }