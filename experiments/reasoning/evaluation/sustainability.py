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
        "avg_findings_length": 0.0
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
            
            for gt_item in ground_truth_data:
                gt_company_name = gt_item.get("GT company_name", "").strip()
                if not gt_company_name:
                    continue
                    
                # Simple similarity check
                similarity = calculate_name_similarity(company_name.lower(), gt_company_name.lower())
                if similarity > best_match_score and similarity > 0.6:  # 60% threshold
                    best_match_score = similarity
                    gt_match = gt_item
            
            if gt_match:
                # Check economic activity accuracy (exact match, no normalization)
                gt_economic_activity = gt_match.get("GT economic_activity", "").strip()
                
                if gt_economic_activity == predicted_activity:
                    economic_activity_correct += 1
                
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
    
    return metrics

def calculate_name_similarity(name1, name2):
    """Calculate simple similarity between two company names"""
    # Remove common corporate suffixes and clean names
    suffixes = ["ltd", "limited", "inc", "corp", "corporation", "llc", "plc", "bv", "nv", "sa", "ag"]
    
    def clean_name(name):
        words = name.lower().split()
        words = [w for w in words if w not in suffixes]
        return " ".join(words)
    
    clean1 = clean_name(name1)
    clean2 = clean_name(name2)
    
    if not clean1 or not clean2:
        return 0.0
    
    # Simple word overlap similarity
    words1 = set(clean1.split())
    words2 = set(clean2.split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)