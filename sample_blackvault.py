#!/usr/bin/env python3
"""
Extract the first 10 documents from blackvault_articles_pdfs.json
and save them to a new file for testing purposes.
"""

import json
import os

def extract_first_10_documents():
    """Extract first 10 documents from the blackvault dataset."""
    
    input_file = "/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/experiments/reasoning/data/blackvault_articles_pdfs.json"
    output_file = "/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/experiments/reasoning/data/blackvault_articles_pdfs_first_10.json"
    
    # Read the original file
    print(f"Reading from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Original dataset contains {len(data)} documents")
    
    # Extract first 10 documents
    first_10_docs = data[:10]
    
    # Save to new file
    print(f"Saving first {len(first_10_docs)} documents to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(first_10_docs, f, indent=2, ensure_ascii=False)
    
    print("✅ Successfully extracted first 10 documents!")
    

if __name__ == "__main__":
    extract_first_10_documents()