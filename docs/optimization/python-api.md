# Optimizing Pipelines with the Python API

Use `pipeline.optimize()` to find cost-accuracy trade-offs for your pipeline. MOAR explores different configurations (models, validation steps, operation rewrites) and returns a frontier of optimized pipelines.

## Quick Example

```python
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

# Optimize — models auto-detected from API keys
result = pipeline.optimize(
    eval_fn="evaluate_medications.py",
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

Create a Python file with a function decorated with `@register_eval`:

```python
# evaluate_medications.py
import json
from docetl.utils_evaluation import register_eval

@register_eval
def evaluate_results(dataset_file_path: str, results_file_path: str) -> dict:
    with open(results_file_path, 'r') as f:
        output = json.load(f)
    with open(dataset_file_path, 'r') as f:
        dataset = json.load(f)

    correct = sum(
        1 for r in output
        for med in r.get("medication", [])
        if med.lower() in r.get("src", "").lower()
    )
    return {"medication_extraction_score": correct}
```

Or pass a callable directly:

```python
result = pipeline.optimize(
    eval_fn=lambda path: {"score": my_scoring_function(path)},
    metric_key="score",
)
```

## Configuration Options

All parameters beyond `eval_fn` and `metric_key` are optional:

```python
result = pipeline.optimize(
    eval_fn="evaluate.py",
    metric_key="score",
    models=["gpt-4o", "gpt-4o-mini"],   # Override auto-detection
    agent_model="gpt-4o",                # Override auto-selection
    max_iterations=40,                   # Default: 20
    save_dir="./moar_results",           # Default: temp dir
    exploration_weight=1.414,            # UCB constant
)
```

See the [Configuration Reference](moar/configuration.md) for details.

## Working with Results

```python
result = pipeline.optimize(eval_fn="evaluate.py", metric_key="score")

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
