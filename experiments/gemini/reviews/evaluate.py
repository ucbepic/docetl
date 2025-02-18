import json
from datetime import datetime
import pandas as pd
from scipy import stats
import numpy as np

def evaluate_reviews(analyzed_reviews_path):
    # Read the analyzed reviews
    with open(analyzed_reviews_path, 'r') as f:
        analyzed_reviews = json.load(f)

    # Read the original reviews for verification
    with open('experiments/gemini/reviews/reviews.json', 'r') as f:
        original_reviews = json.load(f)

    # Read the original CSV to get review scores
    df_reviews = pd.read_csv('experiments/gemini/reviews/reviews_sample.csv')
    review_scores = dict(zip(df_reviews.review_id.astype(str), df_reviews.review_score))

    # Create lookup dict for original reviews
    original_lookup = {review['app_id']: review['concatenated_reviews'] for review in original_reviews}

    def check_timestamps_order(reviews):
        """
        Calculate Kendall's Tau-b correlation between the actual and ideal chronological ordering,
        comparing timestamps directly as strings.
        
        Returns a value between -1 and 1, where:
        - 1 means perfect chronological ordering
        - -1 means perfect reverse ordering
        - 0 means random ordering
        """
        if len(reviews) <= 1:
            return 1.0  # Single review or empty list is considered ordered
            
        # Get timestamps as strings
        timestamps = [r['timestamp'] for r in reviews]
        
        # Create sorted timestamps (ideal chronological order)
        sorted_timestamps = sorted(timestamps)
        
        # Calculate Kendall's Tau-b correlation between actual and sorted timestamps
        tau, _ = stats.kendalltau(timestamps, sorted_timestamps)
        
        return tau if not pd.isna(tau) else 1.0  # Handle edge case where tau is NaN

    def check_timestamp_spread(reviews):
        """
        Calculate how evenly spread out the timestamps are.
        Returns a value between 0 and 1, where:
        - 1 means perfectly evenly spread
        - 0 means all timestamps are clustered
        """
        if len(reviews) <= 1:
            return 1.0

        # Convert timestamps to datetime objects
        timestamps = [datetime.strptime(r['timestamp'], '%Y-%m-%d %H:%M:%S') for r in reviews]
        timestamps_sorted = sorted(timestamps)
        
        # Calculate time differences between consecutive reviews
        time_diffs = [(timestamps_sorted[i+1] - timestamps_sorted[i]).total_seconds() 
                     for i in range(len(timestamps_sorted)-1)]
        
        # Calculate coefficient of variation (lower means more evenly spread)
        cv = np.std(time_diffs) / np.mean(time_diffs) if np.mean(time_diffs) > 0 else 0
        
        # Convert to a 0-1 score where 1 is most evenly spread
        spread_score = 1 / (1 + cv)
        return spread_score

    def verify_review_id(review_id, original_text):
        """Verify if review ID appears in original concatenated reviews"""
        is_found = f"Review ID: {review_id}" in original_text
        if not is_found:
            print(f"Review ID {review_id} not found in original text")
        return is_found

    def verify_review_sentiment(review_id, is_positive):
        """Verify if the review sentiment matches the original score"""
        if review_id not in review_scores:
            return False
        actual_score = review_scores[review_id]
        return (actual_score == is_positive)

    results = []
    for game in analyzed_reviews:
        game_results = {
            'app_id': game['app_id'],
            'app_name': game['app_name'],
            'positive_reviews_ordered': check_timestamps_order(game['positive_reviews']),
            'negative_reviews_ordered': check_timestamps_order(game['negative_reviews']),
            'positive_reviews_spread': check_timestamp_spread(game['positive_reviews']),
            'negative_reviews_spread': check_timestamp_spread(game['negative_reviews']),
            'positive_reviews_found_in_text': [],
            'negative_reviews_found_in_text': [],
            'positive_sentiment_verified': [],
            'negative_sentiment_verified': []
        }
        
        original_text = original_lookup[game['app_id']]
        
        # Verify positive reviews
        for review in game['positive_reviews']:
            found_in_text = verify_review_id(review['review_id'], original_text)
            game_results['positive_reviews_found_in_text'].append(found_in_text)
            sentiment_correct = verify_review_sentiment(review['review_id'], True)
            game_results['positive_sentiment_verified'].append(sentiment_correct)
            
        # Verify negative reviews  
        for review in game['negative_reviews']:
            found_in_text = verify_review_id(review['review_id'], original_text)
            game_results['negative_reviews_found_in_text'].append(found_in_text)
            sentiment_correct = verify_review_sentiment(review['review_id'], False)
            game_results['negative_sentiment_verified'].append(sentiment_correct)
        
        results.append(game_results)

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)

    print("\nResults Summary:")
    print("-" * 50)
    print(f"Total games analyzed: {len(df)}")

    # Calculate average Kendall's Tau scores
    avg_pos_tau = df['positive_reviews_ordered'].mean()
    avg_neg_tau = df['negative_reviews_ordered'].mean()
    print(f"Average positive reviews Kendall's Tau: {avg_pos_tau:.3f}")
    print(f"Average negative reviews Kendall's Tau: {avg_neg_tau:.3f}")

    # Calculate average spread scores
    avg_pos_spread = df['positive_reviews_spread'].mean()
    avg_neg_spread = df['negative_reviews_spread'].mean()
    print(f"Average positive reviews spread score: {avg_pos_spread:.3f}")
    print(f"Average negative reviews spread score: {avg_neg_spread:.3f}")

    pos_found = sum(sum(r['positive_reviews_found_in_text']) for r in results)
    neg_found = sum(sum(r['negative_reviews_found_in_text']) for r in results)
    total_pos = sum(len(r['positive_reviews_found_in_text']) for r in results)
    total_neg = sum(len(r['negative_reviews_found_in_text']) for r in results)

    pos_sentiment = sum(sum(r['positive_sentiment_verified']) for r in results)
    neg_sentiment = sum(sum(r['negative_sentiment_verified']) for r in results)

    print(f"Positive reviews found in text: {pos_found}/{total_pos} ({pos_found/total_pos*100:.1f}%)")
    print(f"Negative reviews found in text: {neg_found}/{total_neg} ({neg_found/total_neg*100:.1f}%)")
    print(f"Verified positive sentiments: {pos_sentiment}/{pos_found} ({pos_sentiment/pos_found*100:.1f}%)")
    print(f"Verified negative sentiments: {neg_sentiment}/{neg_found} ({neg_sentiment/neg_found*100:.1f}%)")
    print("\nOverall Metrics:")
    print(f"Average Kendall's Tau: {(avg_pos_tau + avg_neg_tau)/2:.3f}")
    print(f"Average spread score: {(avg_pos_spread + avg_neg_spread)/2:.3f}")
    print(f"Reviews found in text: {pos_found + neg_found}/{total_pos + total_neg} ({(pos_found + neg_found)/(total_pos + total_neg)*100:.1f}%)")
    print(f"Verified sentiments: {pos_sentiment + neg_sentiment}/{pos_found + neg_found} ({(pos_sentiment + neg_sentiment)/(pos_found + neg_found)*100:.1f}%)")

    # Instead of printing, return metrics dictionary
    metrics = {
        'kendalls_tau': (avg_pos_tau + avg_neg_tau) / 2,
        'spread_score': (avg_pos_spread + avg_neg_spread) / 2,
        'reviews_found': (pos_found + neg_found) / (total_pos + total_neg) * 100,
        'verified_sentiments': (pos_sentiment + neg_sentiment) / (pos_found + neg_found) * 100
    }
    
    return metrics

def print_comparison_table(results):
    """Print a formatted comparison table of evaluation results."""
    headers = ['Method', "Kendall's Tau", 'Reviews Found', 'Verified Sentiments']
    
    # Print header
    print('\nComparison of Methods:')
    print('-' * 75)
    print(f'{headers[0]:<30} | {headers[1]:^12} | {headers[2]:^12} | {headers[3]:^18}')
    print('-' * 75)
    
    # Print each row
    for method, metrics in results.items():
        print(
            f'{method:<30} | '
            f'{metrics["kendalls_tau"]:^12.3f} | '
            f'{metrics["reviews_found"]:^11.1f}% | '
            f'{metrics["verified_sentiments"]:^17.1f}%'
        )
    print('-' * 75)

# Collect results for all methods
results = {
    'Baseline Gemini': evaluate_reviews('experiments/gemini/reviews/analyzed_reviews_baseline.json'),
    'Optimized Gemini': evaluate_reviews('experiments/gemini/reviews/analyzed_reviews_optimized.json')
}

# Print comparison table
print_comparison_table(results)
