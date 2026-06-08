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
    metric_key="score",
    models=["gpt-4o", "gpt-4o-mini"],   # Override auto-detection
    agent_model="gpt-4o",               # Override auto-selection (or set docetl.agent_model)
    max_iterations=40,                   # Default: 20
    save_dir="./moar_results",           # Default: temp dir
    exploration_weight=1.414,            # UCB constant
)
```

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
