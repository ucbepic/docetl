# Evaluation Functions

How to write evaluation functions for MOAR optimization.

## How Evaluation Functions Work

Your evaluation function receives the pipeline output and computes metrics by comparing it to the original dataset. MOAR uses one specific metric from your returned dictionary (specified by `metric_key`) to optimize for accuracy.

!!! info "Function Signature"
    Your function must have exactly this signature:
    ```python
    def evaluate_results(dataset_file_path: str, results_file_path: str) -> Dict[str, Any]:
    ```

### What You Receive

- **`results_file_path`**: Path to JSON file containing your pipeline's output
- **`dataset_file_path`**: Path to JSON file containing the original dataset

### What You Return

A dictionary with numeric metrics. The key specified in `optimizer_config.metric_key` will be used as the accuracy metric for optimization.

!!! tip "Using Original Input Data"
    Pipeline output includes the original input data. For example, if your dataset has a `src` attribute, it will be available in the output. You can use this directly for comparison without loading the dataset file separately.

## Basic Example

```python
import json
from typing import Any, Dict
from docetl.utils_evaluation import register_eval

@register_eval
def evaluate_results(dataset_file_path: str, results_file_path: str) -> Dict[str, Any]:
    # Load pipeline output
    with open(results_file_path, 'r') as f:
        output = json.load(f)
    
    total_correct = 0
    for result in output:
        # For example, if your dataset has a 'src' attribute, it's available in the output
        original_text = result.get("src", "").lower()
        # Replace "your_extraction_key" with the actual key from your pipeline output
        extracted_items = result.get("your_extraction_key", [])
        
        # Check if extracted items appear in original text
        for item in extracted_items:
            if str(item).lower() in original_text:
                total_correct += 1
    
    return {
        "extraction_score": total_correct,  # This key is used if metric_key="extraction_score"
        "total_extracted": sum(len(r.get("your_extraction_key", [])) for r in output),
    }
```

## Requirements

!!! warning "Critical Requirements"
    - The function must be decorated with `@docetl.register_eval`
    - It must take exactly two arguments: `dataset_file_path` and `results_file_path`
    - It must return a dictionary with numeric metrics
    - The `metric_key` in your `optimizer_config` must match one of the keys in this dictionary
    - Only one function per file can be decorated with `@register_eval`

## Performance Considerations

!!! tip "Keep It Fast"
    Your evaluation function will be called many times during optimization. Make sure it's efficient:
    
    - Avoid expensive computations
    - Cache results if possible
    - Keep the function simple and fast

## Common Evaluation Patterns

### Pattern 1: Extraction Verification with Recall

Check if extracted items appear in the document text and compute recall:

```python
@register_eval
def evaluate_results(dataset_file_path: str, results_file_path: str) -> Dict[str, Any]:
    with open(results_file_path, 'r') as f:
        output = json.load(f)
    
    # For example, if your dataset has a 'src' attribute, it's available in the output
    total_correct = 0
    total_extracted = 0
    total_expected = 0
    
    for result in output:
        # Replace "src" with the actual key from your dataset
        original_text = result.get("src", "").lower()
        extracted_items = result.get("your_extraction_key", [])  # Replace with your key
        
        # Count correct extractions (items that appear in text)
        for item in extracted_items:
            total_extracted += 1
            if str(item).lower() in original_text:
                total_correct += 1
        
        # Count expected items (if you have ground truth)
        # total_expected += len(expected_items)
    
    precision = total_correct / total_extracted if total_extracted > 0 else 0.0
    recall = total_correct / total_expected if total_expected > 0 else 0.0
    
    return {
        "extraction_score": total_correct,  # Use this as metric_key
        "precision": precision,
        "recall": recall,
    }
```

### Pattern 2: Comparing Against Ground Truth

Load ground truth from the dataset file and compare:

```python
@register_eval
def evaluate_results(dataset_file_path: str, results_file_path: str) -> Dict[str, Any]:
    with open(results_file_path, 'r') as f:
        predictions = json.load(f)
    
    with open(dataset_file_path, 'r') as f:
        ground_truth = json.load(f)
    
    # Compare predictions with ground truth
    # Adjust keys based on your data structure
    correct = 0
    total = len(predictions)
    
    for pred, truth in zip(predictions, ground_truth):
        # Example: compare classification labels
        if pred.get("predicted_label") == truth.get("true_label"):
            correct += 1
    
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
    }
```

### Pattern 3: External Evaluation (File or API)

Load additional data or call an API for evaluation:

```python
import requests
from pathlib import Path

@register_eval
def evaluate_results(dataset_file_path: str, results_file_path: str) -> Dict[str, Any]:
    with open(results_file_path, 'r') as f:
        output = json.load(f)
    
    # Option A: Load ground truth from a separate file
    ground_truth_path = Path(dataset_file_path).parent / "ground_truth.json"
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    # Option B: Call an API for evaluation
    # response = requests.post("https://api.example.com/evaluate", json=output)
    # api_score = response.json()["score"]
    
    # Evaluate using ground truth
    scores = []
    for result, truth in zip(output, ground_truth):
        # Your evaluation logic here
        score = compute_score(result, truth)
        scores.append(score)
    
    return {
        "average_score": sum(scores) / len(scores) if scores else 0.0,
        "scores": scores,
    }
```

## Testing Your Function

!!! tip "Test Before Running"
    Test your evaluation function independently before running MOAR:
    
    ```python
    result = evaluate_results("dataset.json", "results.json")
    print(result)  # Check that your metric_key is present
    ```

This helps catch errors early and ensures your function works correctly.

