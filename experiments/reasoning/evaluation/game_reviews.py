import json
import argparse
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.stats import kendalltau

# Download VADER lexicon if not already present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse timestamp string to datetime object."""
    try:
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            return datetime.strptime(timestamp_str, "%Y-%m-%d")
        except ValueError:
            return datetime(1900, 1, 1)


def is_positive_review(review_text: str) -> bool:
    """Determine if a review is positive using NLTK VADER sentiment analysis."""
    try:
        sia = SentimentIntensityAnalyzer()
        sentiment_scores = sia.polarity_scores(review_text)
        return sentiment_scores['compound'] > 0.05
    except Exception:
        return 'good' in review_text.lower() or 'great' in review_text.lower()


def extract_reviews_from_input(concatenated_reviews: str) -> List[Dict[str, Any]]:
    """Extract individual reviews from concatenated review text."""
    reviews = []
    review_sections = concatenated_reviews.split("Review ID:")[1:]
    
    for section in review_sections:
        # Try to parse with newlines first
        lines = section.strip().split('\n')
        if len(lines) >= 3:
            # Original parsing logic for newline-separated text
            try:
                review_id = lines[0].strip()
                review_text = ""
                timestamp = None
                
                for line in lines[1:]:
                    if line.startswith("Review:"):
                        review_text = line.replace("Review:", "").strip()
                    elif line.startswith("Timestamp:"):
                        timestamp = line.replace("Timestamp:", "").strip()
                    elif not line.startswith("Helpful Votes:") and review_text:
                        review_text += " " + line.strip()
                
                if review_id and review_text and timestamp:
                    reviews.append({
                        'review_id': review_id,
                        'review_text': review_text.strip(),
                        'timestamp': timestamp,
                        'is_positive': is_positive_review(review_text)
                    })
                    continue
            except Exception as e:
                print(f"Error extracting reviews from input (newline parsing): {e}")
        
        try:
            # Look for section markers in the text without relying on newlines
            section_text = section.strip()
            # Extract review ID (first number/word at the beginning)
            words = section_text.split()
            review_id = words[0] if words else ""
            
            # Find section markers
            review_start = section_text.find("Review:")
            timestamp_start = section_text.find("Timestamp:")
            helpful_start = section_text.find("Helpful Votes:")
            
            review_text = ""
            timestamp = None
            
            if review_start != -1:
                # Extract review text between "Review:" and next section
                review_end = timestamp_start if timestamp_start != -1 else helpful_start if helpful_start != -1 else len(section_text)
                review_text = section_text[review_start + 7:review_end].strip()
            
            if timestamp_start != -1:
                # Extract timestamp from "Timestamp:" to end of string (it's usually last)
                timestamp = section_text[timestamp_start + 10:].strip()
            
            if review_id and review_text and timestamp:
                reviews.append({
                    'review_id': review_id,
                    'review_text': review_text.strip(),
                    'timestamp': timestamp,
                    'is_positive': is_positive_review(review_text)
                })
            else:
                print(f"Failed to extract required fields - ID: '{review_id}', Review: '{review_text}', Timestamp: '{timestamp}'")
                
        except Exception as e:
            print(f"Error extracting reviews from input (non-newline parsing): {e}")
            continue
    
    return reviews


def calculate_kendall_tau_score(all_results: List[Dict], all_gt: List[Dict]) -> float:
    """Calculate Kendall's tau correlation coefficient for temporal ordering accuracy."""
    if len(all_results) == 0 or len(all_gt) == 0:
        return 0.0
    
    # Create mappings from review_id to timestamp
    gt_id_to_timestamp = {r['review_id']: parse_timestamp(r['timestamp']) for r in all_gt}
    
    # Filter results to only include valid review IDs
    valid_results = [r for r in all_results 
                    if isinstance(r, dict) and r.get('review_id') in gt_id_to_timestamp]
    
    if len(valid_results) < 2:
        return 1.0 if len(valid_results) <= 1 else 0.0
    
    try:
        # Get predicted order (order in results) and true order (by timestamp)
        predicted_order = list(range(len(valid_results)))
        result_timestamps = [gt_id_to_timestamp[r['review_id']] for r in valid_results]
        true_order = np.argsort([t.timestamp() for t in result_timestamps])
        
        # Calculate Kendall's tau and normalize to [0, 1]
        tau, _ = kendalltau(predicted_order, true_order)
        return max(0.0, (tau + 1) / 2)
    except Exception:
        return 0.0


def calculate_sentiment_accuracy(positive_results: List[Dict], negative_results: List[Dict], 
                               positive_gt_ids: set, negative_gt_ids: set) -> float:
    """Calculate sentiment classification accuracy."""
    all_results = positive_results + negative_results
    if not all_results:
        return 0.0
    
    correct_sentiment = (
        sum(1 for rev in positive_results
            if isinstance(rev, dict) and rev.get('review_id') in positive_gt_ids) +
        sum(1 for rev in negative_results
            if isinstance(rev, dict) and rev.get('review_id') in negative_gt_ids)
    )
    
    return correct_sentiment / len(all_results)


def evaluate_results(method_name: str, results_file: str, ground_truth_file: str = None, original_json_file: str = None) -> Dict[str, Any]:
    """
    Evaluate game reviews analysis results using weighted score (50-50 Kendall's tau + sentiment).
    
    Args:
        method_name: Name of the method being evaluated
        results_file: Path to the results JSON file
        ground_truth_file: Not used for this evaluation (reviews analysis is self-contained)
    
    Returns:
        Dictionary containing evaluation metrics
    """
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    if original_json_file:
        with open(original_json_file, "r") as f:
            original_json_content = json.load(f)
        
        original_json_content_cleaned = {}
        for item in original_json_content:
            filename = item["name"].split("/")[-1].upper().rstrip(".TXT").replace(".", "").replace(",", "").replace(" ", "").replace("_", "").replace("-", "").replace("'", "").replace(r'[^a-zA-Z0-9]$', '')
            original_json_content_cleaned[filename] = item
        
        orig_valid_games = len(original_json_content_cleaned)
    else: 
        orig_valid_games = 0
    
    total_sentiment_accuracy = 0
    total_kendall_tau_score = 0
    total_weighted_score = 0
    valid_games = 0
    
    for result in results:
        if ('positive_reviews' not in result or 'negative_reviews' not in result or 
            'concatenated_reviews' not in result):
            continue
            
        # Extract ground truth from input
        ground_truth_reviews = extract_reviews_from_input(result['concatenated_reviews'])
        positive_gt = [r for r in ground_truth_reviews if r['is_positive']]
        negative_gt = [r for r in ground_truth_reviews if not r['is_positive']]

        # Get results
        positive_results = result['positive_reviews'] if isinstance(result['positive_reviews'], list) else []
        negative_results = result['negative_reviews'] if isinstance(result['negative_reviews'], list) else []

        all_results = positive_results + negative_results
        all_gt = ground_truth_reviews
        
        valid_games += 1
        
        # Calculate sentiment accuracy
        positive_gt_ids = {r['review_id'] for r in positive_gt}
        negative_gt_ids = {r['review_id'] for r in negative_gt}
        sentiment_accuracy = calculate_sentiment_accuracy(
            positive_results, negative_results, positive_gt_ids, negative_gt_ids
        )
        
        # Calculate Kendall's tau score for temporal ordering
        kendall_tau_score = calculate_kendall_tau_score(all_results, all_gt)
        
        # Calculate weighted combination (50-50)
        weighted_score = 0.5 * kendall_tau_score + 0.5 * sentiment_accuracy
        
        # Accumulate scores
        total_sentiment_accuracy += sentiment_accuracy
        total_kendall_tau_score += kendall_tau_score
        total_weighted_score += weighted_score
    
    
    
    if valid_games == 0:
        return {
            "sentiment_accuracy": 0.0,
            "kendall_tau_score": 0.0,
            "weighted_score": 0.0,
            "valid_games_processed": 0
        }
    
    if orig_valid_games > 0: valid_games = orig_valid_games
    
    return {
        "sentiment_accuracy": total_sentiment_accuracy / valid_games,
        "kendall_tau_score": total_kendall_tau_score / valid_games,
        "weighted_score": total_weighted_score / valid_games,
        "valid_games_processed": valid_games
    }