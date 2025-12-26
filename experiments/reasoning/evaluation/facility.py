import json
from typing import Dict, List, Any


def score_urgency(gold_urgency: str, pred_urgency: str) -> float:
    """Compute score for the urgency classification."""
    return 1.0 if gold_urgency == pred_urgency else 0.0


def score_sentiment(gold_sentiment: str, pred_sentiment: str) -> float:
    """Compute score for the sentiment classification."""
    return 1.0 if gold_sentiment == pred_sentiment else 0.0


def score_categories(gold_categories: Dict[str, bool], pred_categories: List[str]) -> float:
    """
    Compute score for the categories classification.
    Uses the same match/mismatch logic as category accuracy in the DSPy implementation.
    """
    correct = 0
    pred_categories = [p.lower() for p in pred_categories]
    for category, is_present in gold_categories.items():
        if is_present and category.lower() in pred_categories:
            correct += 1
        elif not is_present and category.lower() not in pred_categories:
            correct += 1
    
    return correct / len(gold_categories) if gold_categories else 0.0


def evaluate_results(method_name: str, results_file: str, ground_truth_file: str = None) -> Dict[str, Any]:
    """
    Evaluate facility support analysis results.
    
    Args:
        method_name: Name of the method being evaluated
        results_file: Path to the results JSON file
        ground_truth_file: Not used (ground truth is in the results file with GT prefix)
    
    Returns:
        Dictionary containing evaluation metrics
    """
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    total_urgency_score = 0
    total_sentiment_score = 0
    total_categories_score = 0
    total_combined_score = 0
    valid_samples = 0
    
    for result in results:
        # Check if we have both predictions and ground truth
        if not all(k in result for k in ['urgency', 'sentiment', 'categories', 
                                          'GT urgency', 'GT sentiment', 'GT categories']):
            continue
        
        valid_samples += 1
        
        # Get ground truth values
        gt_urgency = result['GT urgency'].lower()
        gt_sentiment = result['GT sentiment'].lower()
        gt_categories = result['GT categories']
        
        # Get predicted values
        pred_urgency = result['urgency']
        pred_sentiment = result['sentiment']
        pred_categories = result.get('categories', [])
        
        # Ensure pred_categories is a list
        if not isinstance(pred_categories, list):
            pred_categories = []
        
        # Calculate individual scores
        urgency_score = score_urgency(gt_urgency, pred_urgency)
        sentiment_score = score_sentiment(gt_sentiment, pred_sentiment)
        categories_score = score_categories(gt_categories, pred_categories)
        
        # Calculate combined score (average of the three, matching DSPy implementation)
        combined_score = (urgency_score + sentiment_score + categories_score) / 3
        
        # Accumulate scores
        total_urgency_score += urgency_score
        total_sentiment_score += sentiment_score
        total_categories_score += categories_score
        total_combined_score += combined_score
    
    if valid_samples == 0:
        return {
            "urgency_accuracy": 0.0,
            "sentiment_accuracy": 0.0,
            "categories_accuracy": 0.0,
            "combined_score": 0.0,
            "valid_samples_processed": 0
        }
    
    return {
        "urgency_accuracy": total_urgency_score / valid_samples,
        "sentiment_accuracy": total_sentiment_score / valid_samples,
        "categories_accuracy": total_categories_score / valid_samples,
        "combined_score": total_combined_score / valid_samples,
        "valid_samples_processed": valid_samples
    }