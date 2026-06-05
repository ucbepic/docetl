# MOAR Configuration Reference

Complete reference for all MOAR configuration options, covering both the Python API and YAML configuration.

## Python API Parameters

When calling `pipeline.optimize()`, MOAR is the default method:

```python
result = pipeline.optimize(
    eval_fn="evaluate.py",        # Required
    metric_key="score",           # Required
    models=None,                  # Optional — auto-detected from API keys
    agent_model=None,             # Optional — auto-selected best available
    max_iterations=20,            # Optional
    save_dir=None,                # Optional — defaults to temp dir
    exploration_weight=1.414,     # Optional
    method="moar",                # Optional — default, use "v1" for legacy
)
```

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `eval_fn` | `str \| Callable` | Path to Python file containing `@register_eval` decorated function, or a callable |
| `metric_key` | `str` | Key in evaluation results dictionary to use as accuracy metric |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `models` | `list[str] \| None` | `None` (auto-detect) | LiteLLM model names to explore. Auto-detected from environment API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `AZURE_API_KEY`). |
| `agent_model` | `str \| None` | `None` (auto-select) | LLM model for directive instantiation during search. Auto-selects best available. |
| `max_iterations` | `int` | `20` | Maximum number of MOARSearch iterations to run |
| `save_dir` | `str \| None` | `None` (temp dir) | Directory where MOAR results will be saved |
| `exploration_weight` | `float` | `1.414` | UCB exploration constant (higher = more exploration) |
| `method` | `str` | `"moar"` | Optimization method. Use `"v1"` for the legacy V1 optimizer. |

## YAML Configuration

In YAML, add an `optimizer_config` section. Only `evaluation_file` and `metric_key` are required:

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `evaluation_file` | `str` | Path to Python file containing `@register_eval` decorated function |
| `metric_key` | `str` | Key in evaluation results dictionary to use as accuracy metric |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `available_models` | `list[str]` | Auto-detected | LiteLLM model names to explore. Auto-detected from API keys if omitted. |
| `rewrite_agent_model` | `str` | Auto-selected | LLM model for directive instantiation during search |
| `model` | `str` | Auto-selected | Alias for `rewrite_agent_model` |
| `max_iterations` | `int` | `20` | Maximum number of MOARSearch iterations |
| `save_dir` | `str` | Temp dir | Directory where MOAR results will be saved |
| `type` | `str` | `"moar"` | No longer required. Use `"v1"` for legacy optimizer. |
| `dataset_path` | `str` | Inferred from `datasets` | Path to dataset file for optimization. Use a sample/hold-out dataset. |
| `exploration_weight` | `float` | `1.414` | UCB exploration constant (higher = more exploration) |
| `ground_truth_path` | `str` | `None` | Path to ground truth file (for evaluation) |

### Minimal YAML Example

```yaml
optimizer_config:
  evaluation_file: evaluate_medications.py
  metric_key: medication_extraction_score
```

### Full YAML Example

```yaml
optimizer_config:
  evaluation_file: evaluate_medications.py
  metric_key: medication_extraction_score
  available_models:
    - gpt-4o-mini
    - gpt-4o
    - gpt-5.1-mini
    - gpt-5.1
  max_iterations: 40
  rewrite_agent_model: gpt-5.1
  save_dir: results/moar_optimization
  dataset_path: data/sample.json
  exploration_weight: 1.414
```

## Model Auto-Detection

When `models` (Python) or `available_models` (YAML) is omitted, MOAR auto-detects available models from your environment API keys:

| Environment Variable | Models Detected |
|---------------------|----------------|
| `OPENAI_API_KEY` | OpenAI models (gpt-4o-mini, gpt-4o, etc.) |
| `ANTHROPIC_API_KEY` | Anthropic models (claude-sonnet, claude-opus, etc.) |
| `GEMINI_API_KEY` | Google Gemini models |
| `AZURE_API_KEY` | Azure OpenAI models |

!!! tip "Explicit Model Lists"
    You can always override auto-detection by providing an explicit model list. This is useful when you want to restrict the search to specific models.

## Dataset Path

### Automatic Inference

If `dataset_path` is not specified, MOAR will automatically infer it from the `datasets` section of your YAML:

```yaml
datasets:
  transcripts:
    path: data/full_dataset.json  # This will be used if dataset_path not specified
    type: file

optimizer_config:
  # dataset_path not specified — will use data/full_dataset.json
  evaluation_file: evaluate.py
  metric_key: score
```

### Using Sample/Hold-Out Datasets

!!! tip "Best Practice"
    Use a sample or hold-out dataset for optimization to avoid optimizing on your test set.

```yaml
optimizer_config:
  dataset_path: data/sample_dataset.json  # Use sample/hold-out for optimization
  evaluation_file: evaluate.py
  metric_key: score

datasets:
  transcripts:
    path: data/full_dataset.json  # Full dataset for final pipeline
```

## Iteration Count

The `max_iterations` parameter controls how many pipeline configurations MOAR explores:

- **10-20 iterations**: Quick exploration, good for testing (default is 20)
- **40 iterations**: Recommended for most production use cases
- **100+ iterations**: For complex pipelines or when you need the absolute best results

!!! note "Time vs Quality"
    More iterations give better results but take longer and cost more.
