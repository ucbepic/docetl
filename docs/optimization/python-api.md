# Optimizing Pipelines with the Python API

Use `.optimize()` to find cost-accuracy trade-offs for your pipeline. MOAR explores different configurations (models, validation steps, operation rewrites) and returns a frontier of optimized pipelines.

## Quick Example

```python
import docetl

docetl.default_model = "gpt-4o-mini"

frame = (
    docetl.read_json("medical_transcripts.json")
    .map(
        prompt="Analyze the transcript: {{ input.src }}\nList all medications mentioned.",
        output={"schema": {"medication": "list[str]"}},
    )
)

# Define your evaluation function
@docetl.register_eval
def evaluate(results):
    correct = sum(
        1 for r in results
        for med in r.get("medication", [])
        if med.lower() in r.get("src", "").lower()
    )
    return {"medication_extraction_score": correct}

# Optimize — models auto-detected from API keys
optimized = frame.optimize(
    eval_fn=evaluate,
    metric_key="medication_extraction_score",
)

# Run the optimized pipeline
df = optimized.collect()
print(f"Cost: ${optimized.total_cost:.4f}")

# Inspect the Pareto frontier
print(optimized.search_results.to_df())
```

## Evaluation Function

Pass any callable that takes the results list and returns a dict of metrics:

```python
@docetl.register_eval
def evaluate(results):
    correct = sum(
        1 for r in results
        for med in r.get("medication", [])
        if med.lower() in r.get("src", "").lower()
    )
    return {"medication_extraction_score": correct}

optimized = frame.optimize(eval_fn=evaluate, metric_key="medication_extraction_score")
```

!!! tip "File paths for CLI"
    The CLI uses file-based evaluation via `@register_eval`. See the [Evaluation Functions guide](moar/evaluation.md) for that workflow.

## Configuration Options

All parameters beyond `eval_fn` and `metric_key` are optional:

```python
optimized = frame.optimize(
    eval_fn=evaluate,                    # Your evaluation function
    metric_key="score",                  # Key in eval_fn's return dict to optimize
    models=["gpt-4o", "gpt-4o-mini"],   # Override auto-detection
    agent_model="gpt-4o",               # Override auto-selection (or set docetl.agent_model)
    max_iterations=40,                   # Search budget (default: 20)
    save_dir="./moar_results",           # Where to save results (default: temp dir)
    exploration_weight=1.414,            # UCB exploration constant
    dataset_path="data/sample.json",     # Sample dataset for optimization (default: full dataset)
    max_threads=8,                       # Max concurrent LLM calls per pipeline run
    max_concurrent_agents=3,             # Parallel MCTS search agents (default: 3)
)
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `eval_fn` | Callable that scores pipeline output. Takes a results file path and returns a dict of metrics. | **Required** |
| `metric_key` | Which key from `eval_fn`'s return dict to use as the optimization metric. | **Required** |
| `models` | List of LiteLLM model names to explore. | Auto-detected from API keys |
| `agent_model` | Model for the MOAR rewrite agent. | Auto-selected best available (or `docetl.agent_model`) |
| `max_iterations` | Number of MCTS search iterations. Higher = more exploration. | `20` |
| `save_dir` | Directory to save optimized pipelines and results. | Temp directory |
| `exploration_weight` | UCB exploration constant. Higher values explore more; lower values exploit. | `1.414` |
| `dataset_path` | Path to a sample dataset for optimization (avoids optimizing on your full/test set). | Uses the pipeline's dataset |
| `max_threads` | Max concurrent LLM calls for each pipeline execution during search. | `docetl.max_threads` or `cpu_count * 4` |
| `max_concurrent_agents` | Number of parallel MCTS search agents. Each agent explores a different part of the search tree. | `3` |

See the [Configuration Reference](moar/configuration.md) for details.

## Working with Results

```python
optimized = frame.optimize(eval_fn=evaluate, metric_key="score")

# The optimized frame is ready to run
df = optimized.collect()

# Access the full MOAR search results
results = optimized.search_results

# Best accuracy on the frontier
best = results.best()
print(f"Best accuracy: {best.accuracy}, cost: ${best.cost:.4f}")

# Cheapest option on the frontier
cheap = results.cheapest()
print(f"Cheapest cost: ${cheap.cost:.4f}, accuracy: {cheap.accuracy:.4f}")

# Browse the full frontier
for plan in results.frontier:
    print(f"Cost: ${plan.cost:.4f}, Accuracy: {plan.accuracy:.4f}")

# Analyze as a DataFrame
print(results.to_df())
```

See [Understanding Results](moar/results.md) for more details.
