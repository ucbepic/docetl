# Optimizing Pipelines with the Python API

Use `pipeline.optimize()` to find cost-accuracy trade-offs for your pipeline. MOAR explores different configurations (models, validation steps, operation rewrites) and returns a frontier of optimized pipelines.

## Quick Example

```python
from docetl.api import Pipeline, Dataset, MapOp, PipelineStep, PipelineOutput

import json
from docetl.api import Pipeline, Dataset, MapOp, PipelineStep, PipelineOutput

pipeline = Pipeline(
    name="medication_extraction",
    datasets={"transcripts": Dataset(type="file", path="medical_transcripts.json")},
    operations=[
        MapOp(
            name="extract_medications",
            type="map",
            output={"schema": {"medication": "list[str]"}},
            prompt="Analyze the transcript: {{ input.src }}\nList all medications mentioned.",
        ),
    ],
    steps=[PipelineStep(name="extraction", input="transcripts", operations=["extract_medications"])],
    output=PipelineOutput(type="file", path="medication_summaries.json"),
    default_model="gpt-4o-mini",
)

# Define your evaluation function
def evaluate(results_path):
    with open(results_path) as f:
        output = json.load(f)
    correct = sum(
        1 for r in output
        for med in r.get("medication", [])
        if med.lower() in r.get("src", "").lower()
    )
    return {"medication_extraction_score": correct}

# Optimize — models auto-detected from API keys
result = pipeline.optimize(
    eval_fn=evaluate,
    metric_key="medication_extraction_score",
)

# Run the best pipeline
best = result.best()
print(f"Best accuracy: {best.accuracy}, cost: ${best.cost:.4f}")
best.run()

# Or inspect all options as a DataFrame
df = result.to_df()
print(df)
```

## Evaluation Function

Pass any callable that reads the results file and returns a dict of metrics:

```python
def evaluate(results_path):
    with open(results_path) as f:
        output = json.load(f)

    correct = sum(
        1 for r in output
        for med in r.get("medication", [])
        if med.lower() in r.get("src", "").lower()
    )
    return {"medication_extraction_score": correct}

result = pipeline.optimize(eval_fn=evaluate, metric_key="medication_extraction_score")
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

!!! tip "File paths for CLI"
    The CLI still uses file-based evaluation via `@register_eval`. See the [Evaluation Functions guide](moar/evaluation.md) for that workflow.

## Configuration Options

All parameters beyond `eval_fn` and `metric_key` are optional:

```python
result = pipeline.optimize(
    eval_fn=evaluate,                    # Your evaluation function
    metric_key="score",
    models=["gpt-4o", "gpt-4o-mini"],   # Override auto-detection
    agent_model="gpt-4o",                # Override auto-selection
    max_iterations=40,                   # Default: 20
    save_dir="./moar_results",           # Default: temp dir
    exploration_weight=1.414,            # UCB constant
    method="moar",                       # Default; use "v1" for legacy
)
```

!!! tip "Legacy V1 Optimizer"
    To use the legacy V1 optimizer instead of MOAR, pass `method="v1"`:
    ```python
    optimized_pipeline = pipeline.optimize(method="v1")
    ```

See the [Configuration Reference](moar/configuration.md) for details.

## Working with Results

```python
result = pipeline.optimize(eval_fn=evaluate, metric_key="score")

# Best accuracy on the frontier
best = result.best()
best.run()

# Cheapest option on the frontier
cheap = result.cheapest()
cheap.run()

# Browse the full frontier
for plan in result.frontier:
    print(f"Cost: ${plan.cost:.4f}, Accuracy: {plan.accuracy:.4f}")

# Analyze as a DataFrame
df = result.to_df()
```

See [Understanding Results](moar/results.md) for more details.
