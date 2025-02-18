import json
from collections import Counter
from pathlib import Path
import pandas as pd
import requests
import time
from typing import Dict, Optional

def load_json_file(filepath):
    """Load and parse a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
        return data
        # count_key = [k for k in data[0].keys() if k.startswith("_counts")][0]
        # return [e for e in data if e.get(count_key, 1) > 1]

def normalize_text(text):
    """Normalize text by removing commas and converting to lowercase."""
    return text.lower().replace(',', '')

# def get_location_validity(location: str) -> Optional[Dict]:
#     """
#     Check if a location is valid using Nominatim API.
#     Includes rate limiting to be respectful of the API.
#     """
#     # Rate limiting - 1 request per second
#     time.sleep(1)
    
#     base_url = "https://nominatim.openstreetmap.org/search"
#     params = {
#         "q": location,
#         "format": "json",
#         "limit": 1
#     }
#     headers = {
#         "User-Agent": "LocationValidationScript/1.0"  # Required by Nominatim
#     }
    
#     try:
#         response = requests.get(base_url, params=params, headers=headers)
#         response.raise_for_status()
#         results = response.json()
#         return results[0] if results else None
#     except Exception as e:
#         print(f"API error for location '{location}': {str(e)}")
#         return None

def evaluate_precision(data, source_text):
    """Evaluate precision of locations against source text and geocoding."""
    total_locations = 0
    found_in_text = 0
    valid_locations = 0
    
    for event in data:
        locations = event.get('locations', [])
        for location in locations:
            total_locations += 1
            location_words = normalize_text(location).split()
            
            # Check if location appears in source text
            in_text = False
            if location_words and location_words[0] in source_text:
                found_in_text += 1
                in_text = True
            
            # Validate using geocoding API
            if in_text:
            # and get_location_validity(location):
                valid_locations += 1
            else:
                print(f"Location not found via API: {location}")
                if in_text:
                    print(f"  (Note: This location was found in source text)")
    
    text_precision = (found_in_text / total_locations * 100) if total_locations > 0 else 0
    api_precision = (valid_locations / total_locations * 100) if total_locations > 0 else 0
    
    return {
        'total_locations': total_locations,
        'found_in_text': found_in_text,
        'valid_locations': valid_locations,
        'text_precision': round(text_precision, 2),
        'api_precision': round(api_precision, 2)
    }

def count_locations_per_event(data):
    """Count the number of unique locations for each event type."""
    event_counts = Counter()
    
    for event in data:
        event_type = event['event_type']
        locations = event.get('locations', [])
        unique_locations = set(locations)  # Deduplicate locations
        event_counts[event_type] += len(unique_locations)
    
    return dict(event_counts)

def main():
    # Define file paths
    base_path = Path(__file__).parent
    gemini_path = base_path / "event_locations_optimized.json"
    gemini_resolveonly_path = base_path / "event_locations_unoptimized.json"
    
    # Load JSON files

    gemini_data = load_json_file(gemini_path)
    gemini_unoptimized_data = load_json_file(gemini_resolveonly_path)   

    # Count locations per event type for both datasets
    gemini_counts = count_locations_per_event(gemini_data)
    gemini_unoptimized_counts = count_locations_per_event(gemini_unoptimized_data)
    
    # Evaluate precision
    # baseline_precision = evaluate_precision(baseline_data, source_text)
    # optimized_precision = evaluate_precision(optimized_data, source_text)

    # Create a comparison DataFrame
    all_event_types = sorted(set(gemini_counts.keys()) | set(gemini_unoptimized_counts.keys()))
    
    comparison_data = {
        'Event Type': all_event_types,
        'Gemini Locations (Optimized)': [gemini_counts.get(et, 0) for et in all_event_types],
        'Gemini Locations (Unoptimized)': [gemini_unoptimized_counts.get(et, 0) for et in all_event_types]
    }
    
    df = pd.DataFrame(comparison_data)
    
    # Add difference and percentage change columns
    df['Difference'] = df['Gemini Locations (Optimized)'] - df['Gemini Locations (Unoptimized)']
    df['% Change'] = ((df['Gemini Locations (Optimized)'] - df['Gemini Locations (Unoptimized)']) / 
                     df['Gemini Locations (Unoptimized)'] * 100).round(2)

    # Print results
    print("\nLocation Count Comparison:")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 80)
    print(f"Number of Distinct Event Types: {len(all_event_types)}")

    
    gemini_total_diff = sum(gemini_counts.values()) - sum(gemini_unoptimized_counts.values())
    gemini_total_pct = round((gemini_total_diff / sum(gemini_unoptimized_counts.values()) * 100), 2)
    print(f"Total Difference: {gemini_total_diff:+d} ({gemini_total_pct:+.2f}%)")
 

if __name__ == "__main__":      
    main()
