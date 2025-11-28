import json

def evaluate_results(method_name, results_file, ground_truth_file=None, original_json_file=None):
    """
    Evaluate blackvault results by calculating average number of distinct locations per event type.
    
    Args:
        method_name: Name of the method being evaluated
        results_file: Path to the JSON results file
        ground_truth_file: Not used for blackvault (no ground truth available)
    
    Returns:
        dict: Evaluation metrics including average distinct locations per document
    """
    # Read the DocETL results JSON file
    with open(results_file, "r") as f:
        docetl_results = json.load(f)
    

    if not original_json_file: num_files = len(docetl_results)
    else:
        with open(original_json_file, "r") as f:
            original_json_content = json.load(f)
        
        num_files = len(original_json_content)
    
    if not docetl_results:
        return {
            "avg_distinct_locations": 0.0,
            "total_documents": 0,
            "total_distinct_locations": 0,
            "per_document_counts": []
        }
    
    # Ensure we have a list of documents
    if isinstance(docetl_results, dict):
        docetl_results = [docetl_results]
    
    per_document_counts = []
    total_distinct_locations = 0
    
    # Calculate distinct locations for each document (event type)
    for doc in docetl_results:
        locations = doc.get('locations', [])
        
        # Handle different possible formats of locations data
        if isinstance(locations, str):
            # If it's a string, try to parse it or split by common delimiters
            if locations.strip():
                unique_locations = set(loc.strip() for loc in locations.split(',') if loc.strip())
            else:
                unique_locations = set()
        elif isinstance(locations, list):
            # Filter out empty strings and duplicates
            unique_locations = set(loc.strip() for loc in locations if isinstance(loc, str) and loc.strip())
        else:
            unique_locations = set()
        
        distinct_count = len(unique_locations)
        per_document_counts.append(distinct_count)
        total_distinct_locations += distinct_count
    
    # Calculate average distinct locations per document
    num_documents = num_files
    avg_distinct_locations = total_distinct_locations / num_documents if num_documents > 0 else 0.0
    
    return {
        "avg_distinct_locations": avg_distinct_locations,
        "total_documents": num_documents,
        "total_distinct_locations": total_distinct_locations,
        "per_document_counts": per_document_counts,
        "method_name": method_name
    }