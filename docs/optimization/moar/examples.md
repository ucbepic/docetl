# MOAR Examples

Complete working examples for MOAR optimization.

## Medication Extraction Example

This example extracts medications from medical transcripts and evaluates extraction accuracy.

!!! note "Metric Key"
    The `metric_key` in the `optimizer_config` section specifies which key from your evaluation function's return dictionary will be used as the accuracy metric. In this example, `metric_key: medication_extraction_score` means MOAR will optimize using the `medication_extraction_score` value returned by the evaluation function.

### Python API

```python
import json
from docetl.api import Pipeline, Dataset, MapOp, PipelineStep, PipelineOutput

pipeline = Pipeline(
    name="medication_extraction",
    datasets={"transcripts": Dataset(type="file", path="workloads/medical/raw.json")},
    operations=[
        MapOp(
            name="extract_medications",
            type="map",
            output={"schema": {"medication": "list[str]"}},
            prompt=(
                "Analyze the following transcript of a conversation between a doctor and a patient:\n"
                "{{ input.src }}\n"
                "Extract and list all medications mentioned in the transcript.\n"
                "If no medications are mentioned, return an empty list."
            ),
        ),
    ],
    steps=[PipelineStep(name="medication_extraction", input="transcripts", operations=["extract_medications"])],
    output=PipelineOutput(type="file", path="workloads/medical/extracted_medications_results.json"),
    default_model="gpt-4o-mini",
)

# Define evaluation function
def evaluate_medications(results_path):
    with open(results_path) as f:
        output = json.load(f)
    correct = sum(
        1 for r in output
        for med in r.get("medication", [])
        if str(med).lower().strip() in r.get("src", "").lower()
    )
    total = sum(len(r.get("medication", [])) for r in output)
    return {
        "medication_extraction_score": correct,
        "total_extracted": total,
        "precision": correct / total if total > 0 else 0.0,
    }

# Optimize — only eval_fn and metric_key are required
result = pipeline.optimize(
    eval_fn=evaluate_medications,
    metric_key="medication_extraction_score",
)

# Run the best pipeline
best = result.best()
print(f"Accuracy: {best.accuracy}, Cost: ${best.cost:.4f}")
best.run()

# Or explore the full frontier
for plan in result.frontier:
    print(f"Cost: ${plan.cost:.4f}, Accuracy: {plan.accuracy}")
```

### pipeline.yaml

```yaml
datasets:
  transcripts:
    path: workloads/medical/raw.json
    type: file

default_model: gpt-4o-mini
bypass_cache: true

optimizer_config:
  dataset_path: workloads/medical/raw_sample.json  # Use sample for faster optimization
  save_dir: workloads/medical/moar_results
  available_models:  # LiteLLM model names - ensure API keys are set in your environment
    - gpt-5.1-nano
    - gpt-5.1-mini
    - gpt-5.1
    - gpt-4o
    - gpt-4o-mini
  evaluation_file: workloads/medical/evaluate_medications.py
  metric_key: medication_extraction_score
  max_iterations: 40
  rewrite_agent_model: gpt-5.1

system_prompt:
  dataset_description: a collection of transcripts of doctor visits
  persona: a medical practitioner analyzing patient symptoms and reactions to medications

operations:
  - name: extract_medications
    type: map
    output:
      schema:
        medication: list[str]
    prompt: |
      Analyze the following transcript of a conversation between a doctor and a patient:
      {{ input.src }}
      Extract and list all medications mentioned in the transcript.
      If no medications are mentioned, return an empty list.

pipeline:
  steps:
    - name: medication_extraction
      input: transcripts
      operations:
        - extract_medications
  output:
    type: file
    path: workloads/medical/extracted_medications_results.json
```

### evaluate_medications.py (for CLI)

When using the CLI, create a file with a `@register_eval` decorated function:

```python
import json
from typing import Any, Dict
from docetl.utils_evaluation import register_eval

@register_eval
def evaluate_results(dataset_file_path: str, results_file_path: str) -> Dict[str, Any]:
    with open(results_file_path, 'r') as f:
        output = json.load(f)
    
    total_correct = 0
    total_extracted = 0
    
    for result in output:
        original_transcript = result.get("src", "").lower()
        for medication in result.get("medication", []):
            total_extracted += 1
            if str(medication).lower().strip() in original_transcript:
                total_correct += 1
    
    precision = total_correct / total_extracted if total_extracted > 0 else 0.0
    
    return {
        "medication_extraction_score": total_correct,
        "total_extracted": total_extracted,
        "precision": precision,
    }
```

!!! tip "Python API vs CLI"
    In the Python API, you pass the evaluation function directly — no file or `@register_eval` needed. The file-based approach is only required for CLI usage.

### Running the Optimization via CLI

```bash
docetl build workloads/medical/pipeline_medication_extraction.yaml
```

!!! tip "Using Sample Datasets"
    Notice that `dataset_path` points to `raw_sample.json` for optimization, while the main pipeline uses `raw.json`. This prevents optimizing on your test set.

## Key Points

!!! info "Evaluation Function"
    - Python API: pass a callable directly — no file or decorator needed
    - CLI: use a `@register_eval` decorated function in a `.py` file
    - Returns multiple metrics, with `medication_extraction_score` as the primary one

!!! info "Configuration"
    - Only `evaluation_file` and `metric_key` are required in the YAML `optimizer_config`
    - `available_models` is optional -- auto-detected from API keys if omitted
    - Uses a sample dataset for optimization (`dataset_path`)
    - Sets `max_iterations` to 40 for a good balance of exploration and time

