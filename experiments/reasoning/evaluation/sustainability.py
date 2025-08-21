import json
import pandas as pd
import numpy as np
from collections import defaultdict

def evaluate_results(method_name, results_file, ground_truth_file):
    """
    Evaluate sustainability analysis results against ground truth.
    
    Args:
        method_name: Name of the method being evaluated
        results_file: Path to DocETL results JSON file
        ground_truth_file: Path to ground truth data (should be the original dataset)
    
    Returns:
        Dictionary containing evaluation metrics
    """
    
    # Read the DocETL results JSON file
    with open(results_file, "r") as f:
        docetl_results = json.load(f)
    
    # Read the ground truth file (original company reports dataset)
    with open(ground_truth_file, "r") as f:
        ground_truth_data = json.load(f)
    
    # Create ground truth mapping by ID
    gt_by_id = {item["id"]: item for item in ground_truth_data}
    
    # Evaluation metrics
    metrics = {
        "economic_activity_accuracy": 0.0,
        "company_name_accuracy": 0.0,
        "total_companies_processed": 0,
        "economic_activity_distribution": defaultdict(int),
        "missing_companies": 0,
        "avg_findings_length": 0.0,
        "combined_score": 0.0
    }
    
    total_processed = 0
    economic_activity_correct = 0
    company_name_correct = 0
    all_findings_lengths = []
    
    # Process each economic activity group in results
    for activity_result in docetl_results:
        if "economic_activity_summary" not in activity_result:
            continue
            
        predicted_activity = activity_result.get("economic_activity", "unknown")

        
        for company_summary in activity_result["economic_activity_summary"]:
            company_name = company_summary.get("company_name", "").strip()
            key_findings = company_summary.get("key_findings", "")
            
            if not company_name:
                continue
                
            total_processed += 1
            
            # Track findings length
            if key_findings:
                all_findings_lengths.append(len(key_findings))
            
            # Find matching ground truth by company name (fuzzy matching)
            gt_match = None
            best_match_score = 0
            best_match_company_name = None
            for gt_item in ground_truth_data:
                gt_company_name = gt_item.get("GT company_name", "").strip()
                if not gt_company_name:
                    continue
                    
                # Simple similarity check
                similarity = calculate_name_similarity(company_name.lower(), gt_company_name.lower())
                if similarity > best_match_score:  
                    best_match_score = similarity
                    if similarity >= 0.5: gt_match = gt_item
                    best_match_company_name = gt_company_name
            
            if gt_match:
                # Check economic activity accuracy (exact match, no normalization)
                gt_economic_activity = gt_match.get("GT economic_activity", "").strip()
                
                if gt_economic_activity == predicted_activity:
                    economic_activity_correct += 1
                    
                # else:
                #     print(f"Economic activity mismatch: GT {gt_economic_activity} != {predicted_activity}")
    
                # Check company name accuracy (if we found a match, name is reasonably accurate)
                company_name_correct += 1
                
                # Track economic activity distribution
                if gt_economic_activity:
                    metrics["economic_activity_distribution"][gt_economic_activity] += 1
            else:
                metrics["missing_companies"] += 1
    
    # Calculate final metrics
    if total_processed > 0:
        metrics["economic_activity_accuracy"] = economic_activity_correct / total_processed
        metrics["company_name_accuracy"] = company_name_correct / total_processed
        metrics["total_companies_processed"] = total_processed
        
        if all_findings_lengths:
            metrics["avg_findings_length"] = sum(all_findings_lengths) / len(all_findings_lengths)
    
    # Additional analysis metrics
    metrics["total_economic_activities"] = len(metrics["economic_activity_distribution"])
    metrics["most_common_activity"] = max(metrics["economic_activity_distribution"].items(), 
                                        key=lambda x: x[1])[0] if metrics["economic_activity_distribution"] else None
    metrics["combined_score"] = (economic_activity_correct + company_name_correct) / (total_processed * 2)
    
    return metrics

def calculate_name_similarity(name1, name2):
    """Calculate simple similarity between two company names with typo handling"""
    from difflib import SequenceMatcher
    
    # Remove common corporate suffixes and clean names
    suffixes = ["ltd", "limited", "inc", "corp", "corporation", "llc", "plc", "bv", "nv", "sa", "ag"]
    
    def clean_name(name):
        if not name:
            return ""
        words = name.lower().split()
        words = [w for w in words if w not in suffixes]
        return " ".join(words)
    
    clean1 = clean_name(name1)
    clean2 = clean_name(name2)
    
    if not clean1 or not clean2:
        return 0.0
    
    # Check if one name is contained in the other (for cases like "deloitte" vs "deloitte touche...")
    if clean1 in clean2 or clean2 in clean1:
        shorter = min(len(clean1), len(clean2))
        longer = max(len(clean1), len(clean2))
        return shorter / longer
    
    # Word-level similarity with typo tolerance
    words1 = clean1.split()
    words2 = clean2.split()
    
    if not words1 or not words2:
        return 0.0
    
    matched = 0
    used_words = set()
    
    for word1 in words1:
        best_match = 0
        best_word = None
        
        for word2 in words2:
            if word2 in used_words:
                continue
            
            # Exact match or similar enough (handles typos)
            similarity = SequenceMatcher(None, word1, word2).ratio()
            if similarity > best_match:
                best_match = similarity
                best_word = word2
        
        # Accept matches above 80% similarity
        if best_match >= 0.8:
            matched += 1
            used_words.add(best_word)
    
    # Return ratio of matched words to total unique words
    total_words = len(set(words1 + words2))
    return matched / total_words if total_words > 0 else 0.0


def main():
    """
    Main function to run sustainability evaluation.
    Update the file paths below to match your actual data files.
    """
    # Example file paths - update these to match your actual files
    method_name = "sustainability_analysis"
    results_file = "/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/experiments/reasoning/outputs/sustainability_baseline/original_output.json"
    ground_truth_file = "/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/experiments/reasoning/data/company_reports_gt.json"
    
    try:
        # Run evaluation
        metrics = evaluate_results(method_name, results_file, ground_truth_file)
        
        # Print results
        print(f"\n=== Evaluation Results for {method_name} ===")
        print(f"Total companies processed: {metrics['total_companies_processed']}")
        print(f"Economic activity accuracy: {metrics['economic_activity_accuracy']:.3f}")
        print(f"Company name accuracy: {metrics['company_name_accuracy']:.3f}")
        print(f"Missing companies: {metrics['missing_companies']}")
        print(f"Average findings length: {metrics['avg_findings_length']:.1f}")
        print(f"Total economic activities: {metrics['total_economic_activities']}")
        print(f"Most common activity: {metrics['most_common_activity']}")
        print(f"Combined score: {metrics['combined_score']:.3f}")
                
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please update the file paths in the main function to point to your actual data files.")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
    except Exception as e:
        print(f"Error during evaluation: {e}")


if __name__ == "__main__":
    main()

