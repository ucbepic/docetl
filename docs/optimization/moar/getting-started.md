# Getting Started with MOAR

This guide walks you through running your first MOAR optimization step by step.

## Step 1: Create Your Pipeline YAML

Start with a standard DocETL pipeline YAML file:

```yaml
datasets:
  transcripts:
    path: data/transcripts.json
    type: file

default_model: gpt-4o-mini

operations:
  - name: extract_medications
    type: map
    output:
      schema:
        medication: list[str]
    prompt: |
      Extract all medications mentioned in: {{ input.src }}

pipeline:
  steps:
    - name: medication_extraction
      input: transcripts
      operations:
        - extract_medications
  output:
    type: file
    path: results.json
```

!!! note "Standard Pipeline"
    Your pipeline doesn't need any special configuration for MOAR. Just create a normal DocETL pipeline.

## Step 2: Create an Evaluation Function

Create a Python file with an evaluation function. This function will be called for each pipeline configuration that MOAR explores.

!!! info "How Evaluation Works"
    - Your function receives the pipeline output and the original dataset
    - You compute evaluation metrics by comparing the output to the dataset
    - You return a dictionary of metrics
    - MOAR uses one specific key from this dictionary (specified by `metric_key`) as the accuracy metric to optimize

```python
# evaluate_medications.py
import json
from typing import Any, Dict
from docetl.utils_evaluation import register_eval

@register_eval
def evaluate_results(dataset_file_path: str, results_file_path: str) -> Dict[str, Any]:
    """
    Evaluate pipeline output against the original dataset.
    """
    # Load pipeline output
    with open(results_file_path, 'r') as f:
        output = json.load(f)
    
    # Load original dataset for comparison
    with open(dataset_file_path, 'r') as f:
        dataset = json.load(f)
    
    # Compute your evaluation metrics
    correct_count = 0
    total_count = len(output)
    
    for idx, result in enumerate(output):
        # Compare result with original data
        # For example, if your dataset has a 'src' attribute, it's available in the output
        original_text = result.get("src", "").lower()
        extracted_items = result.get("medication", [])
        
        # Check if extracted items appear in original text
        for item in extracted_items:
            if item.lower() in original_text:
                correct_count += 1
    
    # Return dictionary of metrics
    return {
        "medication_extraction_score": correct_count,  # This key will be used if metric_key matches
        "total_extracted": total_count,
        "precision": correct_count / total_count if total_count > 0 else 0.0,
    }
```

!!! warning "Important Requirements"
    - The function must be decorated with `@docetl.register_eval`
    - It must take exactly two arguments: `dataset_file_path` and `results_file_path`
    - It must return a dictionary with numeric metrics
    - The `metric_key` in your `optimizer_config` must match one of the keys in this dictionary
    - Only one function per file can be decorated with `@register_eval`

For more details on evaluation functions, see the [Evaluation Functions guide](evaluation.md).

## Step 3: Configure the Optimizer

Add an `optimizer_config` section to your YAML. The `metric_key` specifies which key from your evaluation function's return dictionary will be used as the accuracy metric for optimization:

```yaml
optimizer_config:
  type: moar
  save_dir: results/moar_optimization
  available_models:  # LiteLLM model names - ensure API keys are set in your environment
    - gpt-4o-mini
    - gpt-4o
    - gpt-4.1-mini
    - gpt-4.1
  evaluation_file: evaluate_medications.py
  metric_key: medication_extraction_score  # This must match a key in your evaluation function's return dictionary
  max_iterations: 40
  model: gpt-4.1
  dataset_path: data/transcripts_sample.json  # Optional: use sample/hold-out dataset
```

!!! tip "Using Sample Datasets"
    Use `dataset_path` to specify a sample or hold-out dataset for optimization. This prevents optimizing on your test set. The main pipeline will still use the full dataset from the `datasets` section.

For complete configuration details, see the [Configuration Reference](configuration.md).

## Step 4: Run the Optimizer

Run MOAR optimization using the CLI:

```bash
docetl build pipeline.yaml --optimizer moar
```

!!! success "What Happens Next"
    MOAR will:
    1. Explore different pipeline configurations
    2. Evaluate each configuration using your evaluation function
    3. Build a cost-accuracy frontier of optimal solutions
    4. Save results to your `save_dir`

## Step 5: Review Results

After optimization completes, check your `save_dir` for:

- **`experiment_summary.json`** - High-level summary of the run
- **`pareto_frontier.json`** - List of optimal solutions
- **`evaluation_metrics.json`** - Detailed evaluation results
- **`pipeline_*.yaml`** - Optimized pipeline configurations

For details on interpreting results, see [Understanding Results](results.md).

## Next Steps

- Learn about [configuration options](configuration.md)
- See [complete examples](examples.md)
- Read [troubleshooting tips](troubleshooting.md)

