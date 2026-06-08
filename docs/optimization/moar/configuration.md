# MOAR Configuration Reference

Complete reference for all MOAR configuration options, covering both the Python API and YAML configuration.

## Python API Parameters

Call `frame.optimize()` to run MOAR optimization. Only `eval_fn` and `metric_key` are required:

```python
def my_eval(results_path):
    import json
    with open(results_path) as f:
        results = json.load(f)
    return {"score": sum(1 for r in results if r.get("correct"))}

optimized = frame.optimize(
    eval_fn=my_eval,                 # Required — a callable
    metric_key="score",              # Required — key in eval_fn's return dict
    models=None,                     # Auto-detected from API keys
    agent_model=None,                # Auto-selected best available
    max_iterations=20,               # Search budget
    save_dir=None,                   # Defaults to temp dir
    exploration_weight=1.414,        # UCB exploration constant
    dataset_path=None,               # Sample dataset for optimization
    max_threads=None,                # Max concurrent LLM calls per pipeline run
    max_concurrent_agents=3,         # Parallel MCTS search agents
)
```

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `eval_fn` | `Callable` | A function that scores pipeline output. 1-arg: `(results_path) -> dict`. 2-arg: `(dataset_path, results_path) -> dict` (dataset path is curried automatically). Also accepts a file path string for CLI compatibility. |
| `metric_key` | `str` | Key in evaluation results dictionary to use as accuracy metric |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `models` | `list[str] \| None` | `None` (auto-detect) | LiteLLM model names to explore. Auto-detected from environment API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `AZURE_API_KEY`). |
| `agent_model` | `str \| None` | `None` (auto-select) | LLM model for directive instantiation during search. Auto-selects best available. Also settable via `docetl.agent_model`. |
| `max_iterations` | `int` | `20` | Number of MCTS search iterations. Higher = more exploration, better results, but slower and costlier. |
| `save_dir` | `str \| None` | `None` (temp dir) | Directory where MOAR results (optimized pipelines, metrics, frontier) will be saved. |
| `exploration_weight` | `float` | `1.414` | UCB exploration constant. Higher values favor exploring new configurations; lower values exploit known-good ones. |
| `dataset_path` | `str \| None` | `None` | Path to a sample/hold-out dataset for optimization. Avoids optimizing on your full dataset. If not set, uses the pipeline's dataset. |
| `max_threads` | `int \| None` | `None` | Max concurrent LLM calls for each pipeline execution during search. Falls back to `docetl.max_threads`, then `cpu_count * 4`. |
| `max_concurrent_agents` | `int` | `3` | Number of parallel MCTS search agents. Each agent independently explores the search tree. Higher values speed up search but use more resources. |

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
| `max_threads` | `int` | `None` | Max concurrent LLM calls per pipeline run |
| `max_concurrent_agents` | `int` | `3` | Number of parallel MCTS search agents |
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
  max_threads: 8
  max_concurrent_agents: 3
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
