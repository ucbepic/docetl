# MOAR Configuration Reference

Complete reference for all MOAR configuration options.

## Required Fields

All fields in `optimizer_config` are required (no defaults):

| Field | Type | Description |
|-------|------|-------------|
| `type` | `str` | Must be `"moar"` |
| `save_dir` | `str` | Directory where MOAR results will be saved |
| `available_models` | `list[str]` | List of LiteLLM model names to explore (e.g., `["gpt-4o-mini", "gpt-4o"]`). Make sure your API keys are set in your environment for these models. |
| `evaluation_file` | `str` | Path to Python file containing `@register_eval` decorated function |
| `metric_key` | `str` | Key in evaluation results dictionary to use as accuracy metric |
| `max_iterations` | `int` | Maximum number of MOARSearch iterations to run |
| `rewrite_agent_model` | `str` | LLM model to use for directive instantiation during search |

!!! warning "All Fields Required"
    MOAR will error if any required field is missing. There are no defaults.

## Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset_path` | `str` | Inferred from `datasets` | Path to dataset file to use for optimization. Use a sample/hold-out dataset to avoid optimizing on your test set. |
| `exploration_weight` | `float` | `1.414` | UCB exploration constant (higher = more exploration) |
| `build_first_layer` | `bool` | `False` | Whether to build initial model-specific nodes |
| `ground_truth_path` | `str` | `None` | Path to ground truth file (for evaluation) |

## Dataset Path

### Automatic Inference

If `dataset_path` is not specified, MOAR will automatically infer it from the `datasets` section of your YAML:

```yaml
datasets:
  transcripts:
    path: data/full_dataset.json  # This will be used if dataset_path not specified
    type: file

optimizer_config:
  # dataset_path not specified - will use data/full_dataset.json
  # ... other config ...
```

### Using Sample/Hold-Out Datasets

!!! tip "Best Practice"
    Use a sample or hold-out dataset for optimization to avoid optimizing on your test set.

```yaml
optimizer_config:
  dataset_path: data/sample_dataset.json  # Use sample/hold-out for optimization
  # ... other config ...

datasets:
  transcripts:
    path: data/full_dataset.json  # Full dataset for final pipeline
```

The optimizer will use the sample dataset, but your final pipeline uses the full dataset. This ensures you don't overfit to your test set during optimization.

## Model Configuration

### Available Models

!!! info "LiteLLM Model Names"
    Use LiteLLM model names (e.g., `gpt-4o-mini`, `gpt-4o`, `gpt-5.1`). Make sure your API keys are set in your environment.

```yaml
available_models:  # LiteLLM model names - ensure API keys are set
  - gpt-5.1-nano      # Cheapest, lower accuracy
  - gpt-5.1-mini      # Low cost, decent accuracy
  - gpt-5.1           # Balanced
  - gpt-4o             # Higher cost, better accuracy
```

### Model for Directive Instantiation

The `rewrite_agent_model` field specifies which LLM to use for generating optimization directives during the search process. This doesn't affect the models tested in `available_models`.

!!! tip "Cost Consideration"
    Use a cheaper model (like `gpt-4o-mini`) for directive instantiation to reduce search costs.

## Iteration Count

The `max_iterations` parameter controls how many pipeline configurations MOAR explores:

- **10-20 iterations**: Quick exploration, good for testing
- **40 iterations**: Recommended for most use cases
- **100+ iterations**: For complex pipelines or when you need the absolute best results

!!! note "Time vs Quality"
    More iterations give better results but take longer and cost more.

## Complete Example

```yaml
optimizer_config:
  type: moar
  save_dir: results/moar_optimization
  available_models:
    - gpt-4o-mini
    - gpt-4o
    - gpt-5.1-mini
    - gpt-5.1
  evaluation_file: evaluate_medications.py
  metric_key: medication_extraction_score
  max_iterations: 40
  rewrite_agent_model: gpt-5.1
  dataset_path: data/sample.json  # Optional
  exploration_weight: 1.414  # Optional
```

