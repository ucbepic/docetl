import json
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
            # If parsing fails, return a default date
            print(f"Failed to parse timestamp: {timestamp_str}")
            return datetime(1900, 1, 1)


def is_positive_review(review_text: str) -> bool:
    """Determine if a review is positive using NLTK VADER sentiment analysis."""
    try:
        sia = SentimentIntensityAnalyzer()
        sentiment_scores = sia.polarity_scores(review_text)
        # Use compound score: positive if > 0.05, negative if < -0.05, neutral otherwise
        return sentiment_scores['compound'] > 0.05
    except Exception:
        # Fallback to simple keyword-based analysis if VADER fails
        positive_indicators = [
            'recommend', 'great', 'amazing', 'excellent', 'fantastic', 'love', 'good', 
            'fun', 'enjoy', 'beautiful', 'perfect', 'awesome', 'brilliant', 'wonderful',
            'like', 'best', 'stunning', 'impressive', 'solid', 'addicting', 'entertaining'
        ]
        
        negative_indicators = [
            'bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointing',
            'broken', 'bug', 'crash', 'unplayable', 'waste', 'refund', 'annoying',
            'frustrating', 'boring', 'repetitive', 'laggy', 'glitch', 'mess'
        ]
        
        text_lower = review_text.lower()
        positive_score = sum(1 for word in positive_indicators if word in text_lower)
        negative_score = sum(1 for word in negative_indicators if word in text_lower)
        
        return positive_score > negative_score


def extract_reviews_from_input(concatenated_reviews: str) -> List[Dict[str, Any]]:
    """Extract individual reviews from concatenated review text."""
    reviews = []
    
    # Split by "Review ID:" to separate reviews
    review_sections = concatenated_reviews.split("Review ID:")[1:]  # Skip first empty element
    
    for section in review_sections:
        lines = section.strip().split('\n')
        if len(lines) < 3:
            continue
            
        try:
            review_id = lines[0].strip()
            review_text = ""
            timestamp = None
            
            # Find the review text and timestamp
            for i, line in enumerate(lines[1:], 1):
                if line.startswith("Helpful Votes:"):
                    # Review text ends here, timestamp should be next
                    if i + 1 < len(lines) and lines[i + 1].startswith("Timestamp:"):
                        timestamp = lines[i + 1].replace("Timestamp:", "").strip()
                    break
                elif line.startswith("Review:"):
                    review_text = line.replace("Review:", "").strip()
                elif not line.startswith("Timestamp:") and not line.startswith("Helpful Votes:"):
                    review_text += " " + line.strip()
            
            if review_id and review_text and timestamp:
                reviews.append({
                    'review_id': review_id,
                    'review_text': review_text.strip(),
                    'timestamp': timestamp,
                    'is_positive': is_positive_review(review_text)
                })
        except Exception:
            continue
    
    return reviews


def evaluate_temporal_distribution(reviews: List[Dict], expected_count: int) -> float:
    """Evaluate how evenly distributed the reviews are across time."""
    if len(reviews) != expected_count:
        return 0.0
    
    timestamps = [parse_timestamp(review['timestamp']) for review in reviews]
    timestamps.sort()
    
    if len(timestamps) < 2:
        return 1.0 if len(timestamps) == expected_count else 0.0
    
    # Calculate time intervals between consecutive reviews
    intervals = []
    for i in range(1, len(timestamps)):
        interval = (timestamps[i] - timestamps[i-1]).total_seconds()
        intervals.append(interval)
    
    if not intervals:
        return 1.0
    
    # Calculate coefficient of variation (lower is better for even distribution)
    mean_interval = np.mean(intervals)
    if mean_interval == 0:
        return 1.0
    
    std_interval = np.std(intervals)
    cv = std_interval / mean_interval
    
    # Convert to score (0-1, where 1 is perfectly even)
    # CV of 0 means perfectly even, CV of 1+ means very uneven
    return max(0, 1 - cv)


def calculate_kendall_tau_score(positive_results: List[Dict], negative_results: List[Dict], 
                               positive_gt: List[Dict], negative_gt: List[Dict]) -> float:
    """
    Calculate Kendall's tau correlation coefficient for temporal ordering accuracy.
    Returns a score between 0 and 1, where 1 is perfect ordering.
    """
    try:
        # Combine all results and ground truth
        all_results = positive_results + negative_results
        all_gt = positive_gt + negative_gt
        
        if len(all_results) == 0 or len(all_gt) == 0:
            return 0.0
        
        # Create mappings from review_id to timestamp
        gt_id_to_timestamp = {r['review_id']: parse_timestamp(r['timestamp']) for r in all_gt}
        
        # Filter results to only include valid review IDs that exist in ground truth
        valid_results = [r for r in all_results 
                        if isinstance(r, dict) and r.get('review_id') in gt_id_to_timestamp]
        
        if len(valid_results) < 2:
            return 1.0 if len(valid_results) <= 1 else 0.0
        
        # Get predicted order (order in which reviews appear in results)
        predicted_order = list(range(len(valid_results)))
        
        # Get true order based on timestamps
        result_timestamps = [gt_id_to_timestamp[r['review_id']] for r in valid_results]
        true_order = np.argsort([t.timestamp() for t in result_timestamps])
        
        # Calculate Kendall's tau
        tau, p_value = kendalltau(predicted_order, true_order)
        
        # Convert tau from [-1, 1] to [0, 1] scale
        normalized_tau = (tau + 1) / 2
        
        return max(0.0, normalized_tau)
        
    except Exception:
        return 0.0


def calculate_combined_accuracy_score(positive_results: List[Dict], negative_results: List[Dict], 
                                    positive_gt: List[Dict], negative_gt: List[Dict]) -> float:
    """
    Calculate combined F1-like accuracy score that counts a review as correct only if:
    1. Not hallucinated (review ID exists in ground truth)
    2. In correct position/order (sorted by timestamp)  
    3. Correctly classified as positive/negative
    """
    
    def is_correctly_sorted(reviews_list, gt_reviews):
        """Check if reviews are sorted by timestamp and match ground truth order."""
        if len(reviews_list) == 0:
            return True
            
        try:
            # Extract valid review IDs that exist in ground truth
            gt_id_to_timestamp = {r['review_id']: parse_timestamp(r['timestamp']) for r in gt_reviews}
            
            valid_result_timestamps = []
            for rev in reviews_list:
                if isinstance(rev, dict) and rev.get('review_id') in gt_id_to_timestamp:
                    valid_result_timestamps.append(gt_id_to_timestamp[rev['review_id']])
            
            return valid_result_timestamps == sorted(valid_result_timestamps)
        except:
            return False
    
    # Check sorting for both positive and negative reviews
    positive_sorted = is_correctly_sorted(positive_results, positive_gt)
    negative_sorted = is_correctly_sorted(negative_results, negative_gt)
    
    if not (positive_sorted and negative_sorted):
        return 0.0  # If not sorted correctly, score is 0
    
    # Count correctly classified, non-hallucinated reviews
    correct_reviews = 0
    total_reviews = 0
    
    positive_gt_ids = {r['review_id'] for r in positive_gt}
    negative_gt_ids = {r['review_id'] for r in negative_gt}
    
    # Check positive reviews
    for rev in positive_results:
        total_reviews += 1
        if (isinstance(rev, dict) and 
            rev.get('review_id') in positive_gt_ids):  # Not hallucinated + correctly classified
            correct_reviews += 1
    
    # Check negative reviews  
    for rev in negative_results:
        total_reviews += 1
        if (isinstance(rev, dict) and 
            rev.get('review_id') in negative_gt_ids):  # Not hallucinated + correctly classified
            correct_reviews += 1
    
    return correct_reviews / max(1, total_reviews)


def evaluate_results(method_name: str, results_file: str, ground_truth_file: str = None) -> Dict[str, Any]:
    """
    Evaluate game reviews analysis results.
    
    Args:
        method_name: Name of the method being evaluated
        results_file: Path to the results JSON file
        ground_truth_file: Not used for this evaluation (reviews analysis is self-contained)
    
    Returns:
        Dictionary containing evaluation metrics
    """
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    total_combined_accuracy = 0
    total_temporal_score = 0
    total_sorting_score = 0
    total_hallucination_score = 0
    total_sentiment_accuracy = 0
    total_kendall_tau_score = 0
    total_weighted_score = 0
    
    valid_games = 0
    
    for result in results:
        if ('positive_reviews' not in result or 'negative_reviews' not in result or 
            'concatenated_reviews' not in result):
            continue
            
        ground_truth_reviews = extract_reviews_from_input(result['concatenated_reviews'])
        positive_gt = [r for r in ground_truth_reviews if r['is_positive']]
        negative_gt = [r for r in ground_truth_reviews if not r['is_positive']]
        
        positive_results = result['positive_reviews'] if isinstance(result['positive_reviews'], list) else []
        negative_results = result['negative_reviews'] if isinstance(result['negative_reviews'], list) else []
        
        valid_games += 1
        
        positive_gt_ids = {r['review_id'] for r in positive_gt}
        negative_gt_ids = {r['review_id'] for r in negative_gt}
        
        # Main metric: Combined accuracy score
        combined_accuracy = calculate_combined_accuracy_score(
            positive_results, negative_results, positive_gt, negative_gt
        )
        
        # Individual metrics for debugging/analysis
        temporal_score = (evaluate_temporal_distribution(positive_results, 10) + 
                         evaluate_temporal_distribution(negative_results, 10)) / 2
        
        def is_sorted_by_timestamp(reviews_list):
            if len(reviews_list) < 2:
                return 1.0
            try:
                timestamps = [parse_timestamp(rev['timestamp']) for rev in reviews_list 
                            if isinstance(rev, dict) and 'timestamp' in rev]
                return 1.0 if timestamps == sorted(timestamps) else 0.0
            except:
                return 0.0
        
        sorting_score = (is_sorted_by_timestamp(positive_results) + is_sorted_by_timestamp(negative_results)) / 2
        
        all_gt_ids = {r['review_id'] for r in ground_truth_reviews}
        total_returned = len(positive_results) + len(negative_results)
        hallucinated_count = sum(1 for rev in positive_results + negative_results
                               if isinstance(rev, dict) and rev.get('review_id') not in all_gt_ids)
        hallucination_score = 1 - (hallucinated_count / max(1, total_returned))
        
        correct_sentiment = (sum(1 for rev in positive_results
                               if isinstance(rev, dict) and rev.get('review_id') in positive_gt_ids) +
                           sum(1 for rev in negative_results
                               if isinstance(rev, dict) and rev.get('review_id') in negative_gt_ids))
        sentiment_accuracy = correct_sentiment / max(1, total_returned)
        
        # Calculate Kendall's tau score for temporal ordering
        kendall_tau_score = calculate_kendall_tau_score(
            positive_results, negative_results, positive_gt, negative_gt
        )
        
        # Calculate weighted combination (50-50) of Kendall's tau and sentiment accuracy
        weighted_score = 0.5 * kendall_tau_score + 0.5 * sentiment_accuracy
        
        # Accumulate scores
        total_combined_accuracy += combined_accuracy
        total_temporal_score += temporal_score
        total_sorting_score += sorting_score
        total_hallucination_score += hallucination_score
        total_sentiment_accuracy += sentiment_accuracy
        total_kendall_tau_score += kendall_tau_score
        total_weighted_score += weighted_score
    
    if valid_games == 0:
        return {
            "combined_accuracy_score": 0.0,
            "temporal_distribution_score": 0.0,
            "sorting_accuracy": 0.0,
            "hallucination_score": 0.0,
            "sentiment_accuracy": 0.0,
            "kendall_tau_score": 0.0,
            "weighted_score": 0.0,
            "valid_games_processed": 0
        }
    
    return {
        "combined_accuracy_score": total_combined_accuracy / valid_games,
        "temporal_distribution_score": total_temporal_score / valid_games,
        "sorting_accuracy": total_sorting_score / valid_games,
        "hallucination_score": total_hallucination_score / valid_games,
        "sentiment_accuracy": total_sentiment_accuracy / valid_games,
        "kendall_tau_score": total_kendall_tau_score / valid_games,
        "weighted_score": total_weighted_score / valid_games,
        "valid_games_processed": valid_games
    }