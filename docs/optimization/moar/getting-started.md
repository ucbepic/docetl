# Getting Started with MOAR

This guide walks you through running your first MOAR optimization step by step.

## Step 1: Create Your Pipeline

Start with a standard DocETL pipeline. You can define it in Python or YAML.

=== "Python"

    ```python
    from docetl.api import Pipeline, Dataset, MapOp, PipelineStep, PipelineOutput

    pipeline = Pipeline(
        name="medication_extraction",
        datasets={"transcripts": Dataset(type="file", path="data/transcripts.json")},
        operations=[
            MapOp(
                name="extract_medications",
                type="map",
                output={"schema": {"medication": "list[str]"}},
                prompt="Extract all medications mentioned in: {{ input.src }}",
            ),
        ],
        steps=[PipelineStep(name="extraction", input="transcripts", operations=["extract_medications"])],
        output=PipelineOutput(type="file", path="results.json"),
        default_model="gpt-4o-mini",
    )
    ```

=== "YAML"

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
        original_text = result.get("src", "").lower()
        extracted_items = result.get("medication", [])
        
        for item in extracted_items:
            if item.lower() in original_text:
                correct_count += 1
    
    return {
        "medication_extraction_score": correct_count,
        "total_extracted": total_count,
        "precision": correct_count / total_count if total_count > 0 else 0.0,
    }
```

!!! warning "Important Requirements"
    - The function must be decorated with `@register_eval`
    - It must take exactly two arguments: `dataset_file_path` and `results_file_path`
    - It must return a dictionary with numeric metrics
    - The `metric_key` must match one of the keys in this dictionary
    - Only one function per file can be decorated with `@register_eval`

For more details on evaluation functions, see the [Evaluation Functions guide](evaluation.md).

## Step 3: Run Optimization

=== "Python API (recommended)"

    `pipeline.optimize()` is the single entry point. Only `eval_fn` and `metric_key` are required -- everything else has smart defaults.

    ```python
    result = pipeline.optimize(
        eval_fn="evaluate_medications.py",
        metric_key="medication_extraction_score",
    )
    ```

    You can also pass a callable directly:

    ```python
    result = pipeline.optimize(
        eval_fn=lambda path: {"score": compute_score(path)},
        metric_key="score",
    )
    ```

    All optional parameters:

    ```python
    result = pipeline.optimize(
        eval_fn="evaluate_medications.py",
        metric_key="medication_extraction_score",
        models=None,              # auto-detect from API keys
        agent_model=None,         # auto-select best available
        max_iterations=20,        # search budget
        save_dir=None,            # defaults to temp dir
        exploration_weight=1.414, # UCB exploration constant
    )
    ```

    !!! tip "Auto-Detection"
        Models are auto-detected from your environment API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `AZURE_API_KEY`). The agent model is auto-selected as the best available model. No need to specify these unless you want to override the defaults.

=== "CLI"

    Add an `optimizer_config` to your YAML. Only `evaluation_file` and `metric_key` are required:

    ```yaml
    optimizer_config:
      evaluation_file: evaluate_medications.py
      metric_key: medication_extraction_score
    ```

    Then run:

    ```bash
    docetl build pipeline.yaml
    ```

    MOAR is the default optimizer -- no flag needed.

## Step 4: Review Results

=== "Python API"

    `optimize()` returns a `MOARResult` object. Results are runnable pipelines, not just data points:

    ```python
    # Get the best pipeline by accuracy
    best = result.best()       # Returns an OptimizedPipeline
    print(f"Accuracy: {best.accuracy}, Cost: ${best.cost:.4f}")
    best.run()                 # Execute the optimized pipeline

    # Get the cheapest pipeline on the frontier
    cheap = result.cheapest()  # Returns an OptimizedPipeline
    cheap.run()

    # View all frontier solutions
    for plan in result.frontier:
        print(f"Accuracy: {plan.accuracy}, Cost: ${plan.cost:.4f}, On frontier: {plan.on_frontier}")

    # Get a DataFrame of all explored plans
    df = result.to_df()
    print(df)
    ```

    Each `OptimizedPipeline` has `.pipeline` (DSLRunner), `.cost`, `.accuracy`, `.yaml_path`, `.on_frontier`, and `.run()`.

=== "CLI"

    After optimization completes, check your `save_dir` for:

    - **`experiment_summary.json`** — High-level summary of the run
    - **`pareto_frontier.json`** — List of optimal solutions
    - **`evaluation_metrics.json`** — Detailed evaluation results
    - **`pipeline_*.yaml`** — Optimized pipeline configurations

For details on interpreting results, see [Understanding Results](results.md).

## Next Steps

- Learn about [configuration options](configuration.md)
- See [complete examples](examples.md)
- Read [troubleshooting tips](troubleshooting.md)
