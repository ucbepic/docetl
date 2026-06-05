# Evaluation Functions

How to write evaluation functions for MOAR optimization.

## How Evaluation Functions Work

Your evaluation function reads the pipeline output and computes metrics. MOAR uses one specific metric from your returned dictionary (specified by `metric_key`) to optimize for accuracy.

### What You Receive

- **`results_path`**: Path to JSON file containing your pipeline's output
- Optionally, **`dataset_path`**: Path to JSON file containing the original dataset (if you use a 2-argument signature)

### What You Return

A dictionary with numeric metrics. The key specified by `metric_key` will be used as the accuracy metric for optimization.

!!! tip "Using Original Input Data"
    Pipeline output includes the original input data. For example, if your dataset has a `src` attribute, it will be available in the output. You can use this directly for comparison without loading the dataset file separately.

## Python API (Recommended)

Pass any callable directly to `pipeline.optimize()`:

```python
import json

def evaluate(results_path):
    with open(results_path) as f:
        output = json.load(f)
    
    total_correct = 0
    for result in output:
        original_text = result.get("src", "").lower()
        for item in result.get("your_extraction_key", []):
            if str(item).lower() in original_text:
                total_correct += 1
    
    return {
        "extraction_score": total_correct,
        "total_extracted": sum(len(r.get("your_extraction_key", [])) for r in output),
    }

result = pipeline.optimize(eval_fn=evaluate, metric_key="extraction_score")
```

If you need access to the original dataset, use a two-argument signature — the dataset path is passed automatically:

```python
def evaluate(dataset_path, results_path):
    with open(results_path) as f:
        output = json.load(f)
    with open(dataset_path) as f:
        dataset = json.load(f)
    # compare output to dataset...
    return {"score": computed_score}
```

## CLI (File-Based)

For CLI usage, create a Python file with a `@register_eval` decorated function:

```python
# evaluate.py
import json
from docetl.utils_evaluation import register_eval

@register_eval
def evaluate_results(dataset_file_path: str, results_file_path: str) -> dict:
    with open(results_file_path) as f:
        output = json.load(f)
    
    total_correct = 0
    for result in output:
        original_text = result.get("src", "").lower()
        for item in result.get("your_extraction_key", []):
            if str(item).lower() in original_text:
                total_correct += 1
    
    return {
        "extraction_score": total_correct,
        "total_extracted": sum(len(r.get("your_extraction_key", [])) for r in output),
    }
```

!!! warning "CLI Requirements"
    - The function must be decorated with `@register_eval`
    - It must take exactly two arguments: `dataset_file_path` and `results_file_path`
    - It must return a dictionary with numeric metrics
    - The `metric_key` must match one of the keys in this dictionary
    - Only one function per file can be decorated with `@register_eval`

## Performance Considerations

!!! tip "Keep It Fast"
    Your evaluation function will be called many times during optimization. Make sure it's efficient:
    
    - Avoid expensive computations
    - Cache results if possible
    - Keep the function simple and fast

## Common Evaluation Patterns

### Pattern 1: Extraction Verification

Check if extracted items appear in the document text:

```python
def evaluate(results_path):
    with open(results_path) as f:
        output = json.load(f)
    
    total_correct = 0
    total_extracted = 0
    
    for result in output:
        original_text = result.get("src", "").lower()
        for item in result.get("your_extraction_key", []):
            total_extracted += 1
            if str(item).lower() in original_text:
                total_correct += 1
    
    precision = total_correct / total_extracted if total_extracted > 0 else 0.0
    
    return {
        "extraction_score": total_correct,
        "precision": precision,
    }
```

### Pattern 2: Comparing Against Ground Truth

Use a two-argument signature to access the dataset:

```python
def evaluate(dataset_path, results_path):
    with open(results_path) as f:
        predictions = json.load(f)
    with open(dataset_path) as f:
        ground_truth = json.load(f)
    
    correct = sum(
        1 for pred, truth in zip(predictions, ground_truth)
        if pred.get("predicted_label") == truth.get("true_label")
    )
    total = len(predictions)
    
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
    }
```

### Pattern 3: Using a Closure

Capture external state (ground truth, config, etc.) via a closure:

```python
def make_eval(ground_truth_path):
    with open(ground_truth_path) as f:
        ground_truth = json.load(f)

    def evaluate(results_path):
        with open(results_path) as f:
            output = json.load(f)
        scores = [compute_score(r, t) for r, t in zip(output, ground_truth)]
        return {"average_score": sum(scores) / len(scores) if scores else 0.0}

    return evaluate

result = pipeline.optimize(
    eval_fn=make_eval("ground_truth.json"),
    metric_key="average_score",
)
```

## Testing Your Function

!!! tip "Test Before Running"
    Test your evaluation function independently before running MOAR:
    
    ```python
    result = evaluate("results.json")
    print(result)  # Check that your metric_key is present
    ```

This helps catch errors early and ensures your function works correctly.

