#!/usr/bin/env python3
"""
Script to split JSON files from train directory into train_split (40 entries) and val (10 entries).
Each JSON file contains 50 entries total.
"""

import json
import os
import shutil
from pathlib import Path

def split_json_files():
    # Define paths
    train_dir = Path("/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/experiments/reasoning/data/train")
    train_split_dir = Path("/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/experiments/reasoning/data/train_split")
    val_dir = Path("/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/experiments/reasoning/data/val")
    
    # Create output directories if they don't exist
    train_split_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all JSON files in the train directory
    json_files = list(train_dir.glob("*.json"))
    
    print(f"Found {len(json_files)} JSON files to process:")
    for file in json_files:
        print(f"  - {file.name}")
    
    # Process each JSON file
    for json_file in json_files:
        print(f"\nProcessing {json_file.name}...")
        
        # Load the JSON data
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"  Total entries: {len(data)}")
        
        # Split the data: first 40 entries for train_split, last 10 for val
        train_split_data = data[:40]
        val_data = data[40:50]
        
        print(f"  Train split entries: {len(train_split_data)}")
        print(f"  Val entries: {len(val_data)}")
        
        # Save train_split data
        train_split_file = train_split_dir / json_file.name
        with open(train_split_file, 'w', encoding='utf-8') as f:
            json.dump(train_split_data, f, indent=2, ensure_ascii=False)
        
        # Save val data
        val_file = val_dir / json_file.name
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Saved {train_split_file.name} to train_split/")
        print(f"  ✓ Saved {val_file.name} to val/")
    
    print(f"\n✅ Successfully split all JSON files!")
    print(f"📁 Train split directory: {train_split_dir}")
    print(f"📁 Val directory: {val_dir}")

if __name__ == "__main__":
    split_json_files()
