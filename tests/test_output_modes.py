"""
Tests for output modes: tools vs structured_output
Tests both map and reduce operations with complex schemas
"""

import pytest
import tempfile
import os
import json
from typing import List, Dict, Any, Optional

from docetl.runner import DSLRunner


# Synthetic data with fruits embedded in documents
SYNTHETIC_DOCUMENTS = [
    {
        "id": 1,
        "content": "The apple orchard was beautiful in spring. The red apples hung heavy on the branches, and nearby grew some carrots and tomatoes in the garden."
    },
    {
        "id": 2, 
        "content": "She made a delicious smoothie with bananas, strawberries, and blueberries. The farmers market also had fresh spinach and kale."
    },
    {
        "id": 3,
        "content": "The orange tree provided shade in the backyard. Orange juice is refreshing, especially when paired with celery sticks and cucumber slices."
    },
    {
        "id": 4,
        "content": "Grape vines covered the trellis, producing sweet purple grapes. The vineyard workers also grew potatoes and onions in adjacent fields."
    },
    {
        "id": 5,
        "content": "Apple pie is a classic dessert. The recipe calls for tart apples, cinnamon, and a side of roasted Brussels sprouts and sweet potatoes."
    },
    {
        "id": 6,
        "content": "Tropical fruits like mango and pineapple are exotic treats. The grocery store display also featured fresh broccoli and bell peppers."
    },
    {
        "id": 7,
        "content": "The strawberry fields stretched for miles. Workers picked ripe strawberries while others harvested lettuce and radishes nearby."
    },
    {
        "id": 8,
        "content": "Banana bread is made with overripe bananas. The recipe pairs well with a salad of arugula and cherry tomatoes."
    },
    {
        "id": 9,
        "content": "Peach cobbler smells amazing when baking. The peaches were picked fresh, along with corn and zucchini from the same farm."
    },
    {
        "id": 10,
        "content": "Watermelon is perfect for summer picnics. The vendor also sold fresh asparagus and green beans at the farmers market."
    }
]


@pytest.fixture
def temp_output_file():
    """Create a temporary output file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        pass
    yield tmp.name
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)


@pytest.fixture
def temp_dataset_file():
    """Write SYNTHETIC_DOCUMENTS to a temporary JSON file and yield its path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as tmp:
        json.dump(SYNTHETIC_DOCUMENTS, tmp)
        tmp.flush()
    try:
        yield tmp.name
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def count_extracted_items(results: List[Dict[str, Any]], operation_type: str) -> int:
    """Count the number of items extracted from results."""
    total_items = 0
    
    for result in results:
        if operation_type == "map":
            if "fruits_and_veggies" in result:
                total_items += len(result["fruits_and_veggies"])
        elif operation_type == "reduce":
            if "duplicate_fruits" in result:
                total_items += len(result["duplicate_fruits"])
    
    return total_items


def assess_accuracy(results: List[Dict[str, Any]], operation_type: str) -> Dict[str, Any]:
    """Assess the accuracy of the results."""
    if operation_type == "map":
        # For map: count unique fruits found and check if veggies are reasonable
        fruits_found = set()
        veggie_mappings = []
        
        for result in results:
            if "fruits_and_veggies" in result:
                for item in result["fruits_and_veggies"]:
                    if isinstance(item, dict) and "fruit" in item:
                        fruits_found.add(item["fruit"].lower())
                        if "most_similar_veggie" in item:
                            veggie_mappings.append({
                                "fruit": item["fruit"],
                                "veggie": item["most_similar_veggie"],
                                "reasoning": item.get("reasoning", "")
                            })
        
        return {
            "unique_fruits_found": len(fruits_found),
            "fruits": list(fruits_found),
            "veggie_mappings": veggie_mappings,
            "total_mappings": len(veggie_mappings)
        }
    
    elif operation_type == "reduce":
        # For reduce: check if duplicates are correctly identified
        duplicates_found = []
        
        for result in results:
            if "duplicate_fruits" in result:
                for item in result["duplicate_fruits"]:
                    if isinstance(item, dict) and "fruit" in item and "count" in item:
                        if item["count"] > 1:
                            duplicates_found.append({
                                "fruit": item["fruit"],
                                "count": item["count"],
                                "doc_ids": item.get("document_ids", [])
                            })
        
        return {
            "duplicate_fruits_found": len(duplicates_found),
            "duplicates": duplicates_found
        }
    
    return {}


@pytest.mark.parametrize("output_mode", ["tools", "structured_output"])
def test_map_operation_complex_schema(output_mode, temp_output_file, temp_dataset_file):
    """Test map operation with complex schema using both output modes."""
    print(f"\n=== Testing MAP operation with {output_mode} mode; path: {temp_output_file} ===")
    
    config = {
        "datasets": {
            "fruits_docs": {
                "type": "file",
                "path": temp_dataset_file
            }
        },
        "default_model": "gpt-4o-mini",
        "operations": [
            {
                "name": "extract_fruits_and_veggies",
                "type": "map",
                "bypass_cache": True,
                "prompt": "Analyze the text and extract all fruits mentioned, along with the most similar vegetable for each fruit based on characteristics like color, texture, or nutritional content.",
                "output": {
                    "schema": {
                        "fruits_and_veggies": "list[{fruit: str, most_similar_veggie: str, reasoning: str}]"
                    },
                    "mode": output_mode
                },
                "model": "gpt-4o-mini"
            }
        ],
        "pipeline": {
            "steps": [
                {
                    "name": "extract_step",
                    "input": "fruits_docs",
                    "operations": ["extract_fruits_and_veggies"]
                }
            ],
            "output": {
                "type": "file",
                "path": temp_output_file
            }
        }
    }
    
    # Create runner and execute
    runner = DSLRunner(config, max_threads=4)
    runner.load()
    
    # Execute the pipeline and get results
    if runner.last_op_container:
        results, _, _ = runner.last_op_container.next()
    else:
        results = []
    
    # Basic assertions
    assert isinstance(results, list), f"Results should be a list for {output_mode} mode"
    assert len(results) > 0, f"Should have results for {output_mode} mode"
    
    # Count extracted items
    items_count = count_extracted_items(results, "map")
    print(f"{output_mode} mode extracted {items_count} fruit-veggie mappings")
    
    # Should extract at least one item
    assert items_count >= 1, f"{output_mode} mode should extract at least one fruit-veggie mapping"
    
    # Assess accuracy
    accuracy = assess_accuracy(results, "map")
    print(f"{output_mode} mode found {accuracy['unique_fruits_found']} unique fruits: {accuracy['fruits']}")
    print(f"{output_mode} mode created {accuracy['total_mappings']} fruit-veggie mappings")
    
    # Store results for comparison
    return {
        "mode": output_mode,
        "items_count": items_count,
        "accuracy": accuracy,
        "results": results
    }


@pytest.mark.parametrize("output_mode", ["tools", "structured_output"])
def test_reduce_operation_with_folding(output_mode, temp_output_file, temp_dataset_file):
    """Test reduce operation with folding using both output modes."""
    print(f"\n=== Testing REDUCE operation with {output_mode} mode ===")
    
    config = {
        "datasets": {
            "fruits_docs": {
                "type": "file",
                "path": temp_dataset_file
            }
        },
        "default_model": "gpt-4o-mini",
        "operations": [
            {
                "name": "find_duplicate_fruits",
                "reduce_key": "_all",
                "type": "reduce",
                "bypass_cache": True,
                "prompt": "Find all fruits that are mentioned more than once across the documents in {{ inputs }}. Count the occurrences accurately.",
                "fold_batch_size": 3,
                "fold_prompt": "Given the previous output {{ output }} and the new batch of documents {{ inputs }}, summarize the fruits mentioned and their counts. Combine duplicate entries and maintain accurate counts.",
                "output": {
                    "schema": {
                        "duplicate_fruits": "list[{fruit: str, count: int, document_ids: list[int]}]"
                    },
                    "mode": output_mode
                },
                "model": "gpt-4o-mini"
            }
        ],
        "pipeline": {
            "steps": [
                {
                    "name": "reduce_step",
                    "input": "fruits_docs",
                    "operations": ["find_duplicate_fruits"]
                }
            ],
            "output": {
                "type": "file",
                "path": temp_output_file
            }
        }
    }
    
    # Create runner and execute
    runner = DSLRunner(config, max_threads=4)
    runner.load()
    
    # Execute the pipeline and get results
    if runner.last_op_container:
        results, _, _ = runner.last_op_container.next()
    else:
        results = []
    
    # Basic assertions
    assert isinstance(results, list), f"Results should be a list for {output_mode} mode"
    assert len(results) > 0, f"Should have results for {output_mode} mode"
    
    # Count extracted items
    items_count = count_extracted_items(results, "reduce")
    print(f"{output_mode} mode found {items_count} duplicate fruits")
    
    # Should find at least one duplicate fruit (apples and bananas appear multiple times)
    assert items_count >= 1, f"{output_mode} mode should find at least one duplicate fruit"
    
    # Assess accuracy
    accuracy = assess_accuracy(results, "reduce")
    print(f"{output_mode} mode found duplicate fruits: {[d['fruit'] for d in accuracy['duplicates']]}")
    
    # Store results for comparison
    return {
        "mode": output_mode,
        "items_count": items_count,
        "accuracy": accuracy,
        "results": results
    }


def test_compare_output_modes(temp_output_file, temp_dataset_file):
    """Compare the performance of both output modes."""
    print("\n" + "="*60)
    print("COMPARING OUTPUT MODES")
    print("="*60)
    
    # Test map operations
    map_results = {}
    for mode in ["tools", "structured_output"]:
        try:
            result = test_map_operation_complex_schema(mode, temp_output_file, temp_dataset_file)
            map_results[mode] = result
        except Exception as e:
            print(f"Error testing {mode} for map: {e}")
            map_results[mode] = {"error": str(e)}
    
    # Test reduce operations  
    reduce_results = {}
    for mode in ["tools", "structured_output"]:
        try:
            result = test_reduce_operation_with_folding(mode, temp_output_file, temp_dataset_file)
            reduce_results[mode] = result
        except Exception as e:
            print(f"Error testing {mode} for reduce: {e}")
            reduce_results[mode] = {"error": str(e)}
    
    # Compare map results
    print("\n--- MAP OPERATION COMPARISON ---")
    if "tools" in map_results and "structured_output" in map_results:
        tools_map = map_results["tools"]
        structured_map = map_results["structured_output"]
        
        if "error" not in tools_map and "error" not in structured_map:
            print(f"Tools mode: {tools_map['items_count']} items, {tools_map['accuracy']['unique_fruits_found']} unique fruits")
            print(f"Structured mode: {structured_map['items_count']} items, {structured_map['accuracy']['unique_fruits_found']} unique fruits")
            
            if structured_map['accuracy']['unique_fruits_found'] > tools_map['accuracy']['unique_fruits_found']:
                print("🏆 STRUCTURED OUTPUT had more accurate fruit extraction for MAP operation")
            elif tools_map['accuracy']['unique_fruits_found'] > structured_map['accuracy']['unique_fruits_found']:
                print("🏆 TOOLS mode had more accurate fruit extraction for MAP operation")
            else:
                print("🤝 Both modes performed equally for MAP operation")
    
    # Compare reduce results
    print("\n--- REDUCE OPERATION COMPARISON ---")
    if "tools" in reduce_results and "structured_output" in reduce_results:
        tools_reduce = reduce_results["tools"]
        structured_reduce = reduce_results["structured_output"]
        
        if "error" not in tools_reduce and "error" not in structured_reduce:
            print(f"Tools mode: {tools_reduce['items_count']} duplicate fruits found")
            print(f"Structured mode: {structured_reduce['items_count']} duplicate fruits found")
            
            # Check which found the expected duplicates (apples, bananas, strawberries appear multiple times)
            expected_duplicates = {"apple", "banana", "strawberry"}
            
            tools_found = {d['fruit'].lower() for d in tools_reduce['accuracy']['duplicates']}
            structured_found = {d['fruit'].lower() for d in structured_reduce['accuracy']['duplicates']}
            
            tools_correct = len(tools_found.intersection(expected_duplicates))
            structured_correct = len(structured_found.intersection(expected_duplicates))
            
            print(f"Tools mode found {tools_correct} expected duplicates: {tools_found}")
            print(f"Structured mode found {structured_correct} expected duplicates: {structured_found}")
            
            if structured_correct > tools_correct:
                print("🏆 STRUCTURED OUTPUT had more accurate duplicate detection for REDUCE operation")
            elif tools_correct > structured_correct:
                print("🏆 TOOLS mode had more accurate duplicate detection for REDUCE operation")
            else:
                print("🤝 Both modes performed equally for REDUCE operation")
    
    print("\n" + "="*60)