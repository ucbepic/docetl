# MOAR Examples

Complete working examples for MOAR optimization.

## Medication Extraction Example

This example extracts medications from medical transcripts and evaluates extraction accuracy.

!!! note "Metric Key"
    The `metric_key` in the `optimizer_config` section specifies which key from your evaluation function's return dictionary will be used as the accuracy metric. In this example, `metric_key: medication_extraction_score` means MOAR will optimize using the `medication_extraction_score` value returned by the evaluation function.

### pipeline.yaml

```yaml
datasets:
  transcripts:
    path: workloads/medical/raw.json
    type: file

default_model: gpt-4o-mini
bypass_cache: true

optimizer_config:
  type: moar
  dataset_path: workloads/medical/raw_sample.json  # Use sample for faster optimization
  save_dir: workloads/medical/moar_results
  available_models:  # LiteLLM model names - ensure API keys are set in your environment
    - gpt-4.1-nano
    - gpt-4.1-mini
    - gpt-4.1
    - gpt-4o
    - gpt-4o-mini
  evaluation_file: workloads/medical/evaluate_medications.py
  metric_key: medication_extraction_score
  max_iterations: 40
  model: gpt-4.1

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

### evaluate_medications.py

```python
import json
from typing import Any, Dict
from docetl.utils_evaluation import register_eval

@register_eval
def evaluate_results(dataset_file_path: str, results_file_path: str) -> Dict[str, Any]:
    """
    Evaluate medication extraction results.
    
    Checks if each extracted medication appears verbatim in the original transcript.
    In this example, the dataset has a 'src' attribute with the original input text.
    """
    # Load pipeline output
    with open(results_file_path, 'r') as f:
        output = json.load(f)
    
    total_correct_medications = 0
    total_extracted_medications = 0
    
    # Evaluate each result
    for result in output:
        # In this example, the dataset has a 'src' attribute with the original transcript
        original_transcript = result.get("src", "").lower()
        extracted_medications = result.get("medication", [])
        
        # Check each extracted medication
        for medication in extracted_medications:
            total_extracted_medications += 1
            medication_lower = str(medication).lower().strip()
            
            # Check if medication appears in transcript
            if medication_lower in original_transcript:
                total_correct_medications += 1
    
    # Calculate metrics
    precision = total_correct_medications / total_extracted_medications if total_extracted_medications > 0 else 0.0
    
    return {
        "medication_extraction_score": total_correct_medications,  # This is used as the accuracy metric
        "total_correct_medications": total_correct_medications,
        "total_extracted_medications": total_extracted_medications,
        "precision": precision,
    }
```

### Running the Optimization

```bash
docetl build workloads/medical/pipeline_medication_extraction.yaml --optimizer moar
```

!!! tip "Using Sample Datasets"
    Notice that `dataset_path` points to `raw_sample.json` for optimization, while the main pipeline uses `raw.json`. This prevents optimizing on your test set.

## Key Points

!!! info "Evaluation Function"
    - In this example, uses the `src` attribute from output items (no need to load dataset separately)
    - Checks if extracted medications appear verbatim in the transcript
    - Returns multiple metrics, with `medication_extraction_score` as the primary one

!!! info "Configuration"
    - Uses a sample dataset for optimization (`dataset_path`)
    - Includes multiple models in `available_models` to explore trade-offs
    - Sets `max_iterations` to 40 for a good balance of exploration and time

