# Evaluation Functions

How to write evaluation functions for MOAR optimization.

!!! tip "No ground truth? Use an LLM judge"
    If you can't write a label function for your task, skip `eval_fn` entirely and pass `judge_model` instead — see [LLM-as-Judge Evaluation](#llm-as-judge-evaluation) below.

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

## LLM-as-Judge Evaluation

When your task has no ground truth labels, designate a judge LLM instead of writing an evaluation function. This is the same agent-guided plan evaluation introduced in the [original DocETL paper](https://arxiv.org/abs/2410.12189), adapted to MOAR's search loop:

1. **Criteria synthesis** — The rewrite agent model reads your pipeline (operation prompts, output schemas, sample inputs) and writes task-specific validation criteria. You can also supply your own with `judge_criteria`.
2. **1–5 rating** — After a candidate plan runs on the sample data, the judge model rates its outputs against the criteria on a 1–5 scale, per sample document.
3. **Ranked insertion** — The new plan is slotted into a leaderboard of all previously evaluated plans. The rating routes it to a neighborhood of similarly rated plans; judge comparisons (batched, best-first ranking calls over the candidate and its neighbors) decide its exact position.

The plan's "accuracy" is a score in (0, 1) derived from its leaderboard position, so the MCTS rewards are **ranking-based**. Two invariants hold across MOAR iterations:

- The relative order of previously evaluated plans never changes — a new plan's outputs can only be *inserted* into the current ranking, never reshuffle it.
- Scores are immutable once assigned, keeping rewards stationary for the search.

### Python API

```python
result = pipeline.optimize(
    judge_model="gpt-4.1-mini",          # instead of eval_fn/metric_key
    # judge_criteria="...",              # optional; auto-generated if omitted
)
```

### YAML / CLI

```yaml
optimizer_config:
  judge_model: gpt-4.1-mini
  # judge_criteria: |
  #   The output should list every landmark mentioned in the document...
```

### Notes

- Criteria generation runs on the **rewrite agent model** (`agent_model`); rating and comparison calls run on the **judge model**, which can be a cheaper model.
- Long outputs are handled with batching and token-budgeted truncation: comparisons are made per sample document where outputs align 1:1 with inputs, in one batched ranking call per document (reusing the `rank` operator's batch machinery). If a whole neighborhood doesn't fit the judge's context window, placement falls back to one-vs-one comparisons, and individual outputs are truncated as a last resort.
- The full audit trail — criteria, ratings, every comparison, and the leaderboard — is written to `ranking.json` in `save_dir`.
- Judge LLM costs are counted in `result.total_search_cost`.

