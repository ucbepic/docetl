#!/usr/bin/env python3
"""
Script to compute words per document for each dataset test set.

This script analyzes JSON files in the test data directory and computes
word counts for each document across different datasets.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
import statistics


def count_words(text: str) -> int:
    """
    Count words in a text string.
    
    Args:
        text: Input text string
        
    Returns:
        Number of words in the text
    """
    if not text or not isinstance(text, str):
        return 0
    
    # Remove extra whitespace and split by whitespace
    words = re.findall(r'\b\w+\b', text.lower())
    return len(words)


def extract_document_content(doc: Dict[str, Any], dataset_name: str) -> str:
    """
    Extract document content based on dataset structure.
    
    Args:
        doc: Document dictionary
        dataset_name: Name of the dataset
        
    Returns:
        Document content as string
    """
    # Different datasets use different field names for content
    if dataset_name == 'biodex':
        # Special case: combine fulltext_processed and possible_labels
        content_parts = []
        if 'fulltext_processed' in doc and doc['fulltext_processed']:
            content_parts.append(str(doc['fulltext_processed']))
        if 'possible_labels' in doc and doc['possible_labels']:
            content_parts.append(str(doc['possible_labels']))
        return ' '.join(content_parts)
    
    elif dataset_name == 'blackvault':
        field_name = 'all_content'
    elif dataset_name == 'cuad':
        field_name = 'document'
    elif dataset_name == 'game_reviews':
        field_name = 'concatenated_reviews'
    elif dataset_name == 'medec':
        field_name = 'Text'
    elif dataset_name == 'sustainability':
        field_name = 'tot_text_raw'
    else:
        # Fallback for unknown datasets
        field_name = 'content'
    
    # Extract content from the specified field
    if field_name in doc and doc[field_name]:
        return str(doc[field_name])
    
    # If no content field found, return empty string
    return ""


def analyze_dataset(file_path: Path) -> Dict[str, Any]:
    """
    Analyze a single dataset file and compute word statistics.
    
    Args:
        file_path: Path to the JSON dataset file
        
    Returns:
        Dictionary containing word count statistics
    """
    dataset_name = file_path.stem
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {
            'dataset': dataset_name,
            'error': str(e),
            'total_documents': 0,
            'word_counts': [],
            'statistics': {}
        }
    
    if not isinstance(data, list):
        print(f"Warning: {file_path} does not contain a list of documents")
        return {
            'dataset': dataset_name,
            'error': 'Invalid format: not a list',
            'total_documents': 0,
            'word_counts': [],
            'statistics': {}
        }
    
    word_counts = []
    empty_docs = 0
    
    for i, doc in enumerate(data):
        if not isinstance(doc, dict):
            print(f"Warning: Document {i} in {dataset_name} is not a dictionary")
            continue
            
        content = extract_document_content(doc, dataset_name)
        word_count = count_words(content)
        
        if word_count == 0:
            empty_docs += 1
        else:
            word_counts.append(word_count)
    
    # Compute statistics
    if word_counts:
        stats = {
            'mean': round(statistics.mean(word_counts), 2),
            'median': round(statistics.median(word_counts), 2),
            'min': min(word_counts),
            'max': max(word_counts),
            'std_dev': round(statistics.stdev(word_counts), 2) if len(word_counts) > 1 else 0,
            'total_words': sum(word_counts),
            'non_empty_documents': len(word_counts),
            'empty_documents': empty_docs
        }
    else:
        stats = {
            'mean': 0,
            'median': 0,
            'min': 0,
            'max': 0,
            'std_dev': 0,
            'total_words': 0,
            'non_empty_documents': 0,
            'empty_documents': empty_docs
        }
    
    return {
        'dataset': dataset_name,
        'total_documents': len(data),
        'word_counts': word_counts,
        'statistics': stats
    }


def print_summary_table(results: List[Dict[str, Any]]) -> None:
    """
    Print a formatted summary table of results.
    
    Args:
        results: List of dataset analysis results
    """
    print("\n" + "="*100)
    print("WORDS PER DOCUMENT ANALYSIS - SUMMARY TABLE")
    print("="*100)
    
    # Header
    print(f"{'Dataset':<15} {'Total Docs':<12} {'Non-Empty':<12} {'Mean':<8} {'Median':<8} {'Min':<8} {'Max':<10} {'Std Dev':<10} {'Total Words':<12}")
    print("-" * 100)
    
    # Data rows
    for result in results:
        if 'error' in result:
            print(f"{result['dataset']:<15} {'ERROR':<12} {'-':<12} {'-':<8} {'-':<8} {'-':<8} {'-':<10} {'-':<10} {'-':<12}")
        else:
            stats = result['statistics']
            print(f"{result['dataset']:<15} {result['total_documents']:<12} {stats['non_empty_documents']:<12} "
                  f"{stats['mean']:<8} {stats['median']:<8} {stats['min']:<8} {stats['max']:<10} "
                  f"{stats['std_dev']:<10} {stats['total_words']:<12}")


def print_detailed_results(results: List[Dict[str, Any]]) -> None:
    """
    Print detailed results for each dataset.
    
    Args:
        results: List of dataset analysis results
    """
    print("\n" + "="*100)
    print("DETAILED RESULTS BY DATASET")
    print("="*100)
    
    for result in results:
        print(f"\nDataset: {result['dataset']}")
        print("-" * 50)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            continue
        
        stats = result['statistics']
        print(f"Total Documents: {result['total_documents']}")
        print(f"Non-empty Documents: {stats['non_empty_documents']}")
        print(f"Empty Documents: {stats['empty_documents']}")
        print(f"Total Words: {stats['total_words']:,}")
        print(f"Mean Words per Document: {stats['mean']}")
        print(f"Median Words per Document: {stats['median']}")
        print(f"Min Words per Document: {stats['min']}")
        print(f"Max Words per Document: {stats['max']}")
        print(f"Standard Deviation: {stats['std_dev']}")
        
        # Show word count distribution
        if result['word_counts']:
            word_counts = sorted(result['word_counts'])
            print(f"\nWord Count Distribution:")
            print(f"  25th percentile: {word_counts[len(word_counts)//4]}")
            print(f"  75th percentile: {word_counts[3*len(word_counts)//4]}")
            
            # Show some examples
            print(f"\nSample Word Counts (first 10): {word_counts[:10]}")


def save_results_to_file(results: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save results to a JSON file.
    
    Args:
        results: List of dataset analysis results
        output_file: Output file path
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")
    except Exception as e:
        print(f"Error saving results to {output_file}: {e}")


def main():
    """Main function to analyze all datasets."""
    # Define the test data directory
    test_data_dir = Path("/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/experiments/reasoning/data/test")
    
    if not test_data_dir.exists():
        print(f"Error: Test data directory not found: {test_data_dir}")
        return
    
    # Find all JSON files in the directory
    json_files = list(test_data_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {test_data_dir}")
        return
    
    print(f"Found {len(json_files)} dataset files:")
    for file_path in json_files:
        print(f"  - {file_path.name}")
    
    # Analyze each dataset
    results = []
    for file_path in sorted(json_files):
        print(f"\nAnalyzing {file_path.name}...")
        result = analyze_dataset(file_path)
        results.append(result)
    
    # Print results
    print_summary_table(results)
    print_detailed_results(results)
    
    # Save results to file
    output_file = "/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/words_per_document_analysis.json"
    save_results_to_file(results, output_file)
    
    print(f"\nAnalysis complete! Processed {len(json_files)} datasets.")


if __name__ == "__main__":
    main()
