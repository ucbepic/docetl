# Understanding MOAR Results

What MOAR outputs and how to interpret the results.

## Python API Results

When using the Python API, `pipeline.optimize()` returns a `MOARResult` object with methods to access optimized pipelines.

### MOARResult

```python
result = pipeline.optimize(eval_fn=evaluate, metric_key="score")

result.best()      # OptimizedPipeline with highest accuracy on the frontier
result.cheapest()  # OptimizedPipeline with lowest cost on the frontier
result.frontier    # list[OptimizedPipeline] — all Pareto-optimal solutions
result.to_df()     # pandas DataFrame of all explored plans
```

| Method / Property | Return Type | Description |
|-------------------|-------------|-------------|
| `best()` | `OptimizedPipeline` | The frontier solution with the highest accuracy |
| `cheapest()` | `OptimizedPipeline` | The frontier solution with the lowest cost |
| `frontier` | `list[OptimizedPipeline]` | All Pareto-optimal solutions, sorted by cost |
| `to_df()` | `pandas.DataFrame` | DataFrame of all explored plans with cost, accuracy, and metadata |

### OptimizedPipeline

Each result on the frontier is an `OptimizedPipeline` that you can inspect and run directly:

```python
best = result.best()

# Inspect
print(best.cost)        # Estimated cost per run
print(best.accuracy)    # Evaluation metric score
print(best.yaml_path)   # Path to the optimized YAML file
print(best.on_frontier) # True if on the Pareto frontier

# Run
best.run()              # Execute the optimized pipeline

# Access the underlying DSLRunner
best.pipeline           # DSLRunner instance
```

| Property / Method | Type | Description |
|-------------------|------|-------------|
| `pipeline` | `DSLRunner` | The underlying pipeline runner |
| `cost` | `float` | Estimated cost per run |
| `accuracy` | `float` | Evaluation metric score |
| `yaml_path` | `str` | Path to the optimized YAML configuration |
| `on_frontier` | `bool` | Whether this plan is on the Pareto frontier |
| `run()` | `float` | Execute the pipeline; returns execution cost |

### Working with Results

```python
# Choose based on your priorities
result = pipeline.optimize(eval_fn=evaluate, metric_key="score")

# Highest accuracy
best = result.best()
print(f"Best accuracy: {best.accuracy}, cost: ${best.cost:.4f}")
best.run()

# Lowest cost
cheap = result.cheapest()
print(f"Cheapest cost: ${cheap.cost:.4f}, accuracy: {cheap.accuracy}")

# Explore the full frontier
for plan in result.frontier:
    print(f"Cost: ${plan.cost:.4f}, Accuracy: {plan.accuracy}")

# Analyze all explored configurations as a DataFrame
df = result.to_df()
print(df[["cost", "accuracy", "on_frontier"]].sort_values("accuracy", ascending=False))
```

## CLI Output Files

After running `docetl build pipeline.yaml`, you'll find several files in your `save_dir`:

- **`experiment_summary.json`** — High-level summary
- **`pareto_frontier.json`** — Optimal solutions
- **`evaluation_metrics.json`** — Detailed evaluation results
- **`pipeline_*.yaml`** — Optimized pipeline configurations

### experiment_summary.json

High-level summary of the optimization run:

```json
{
  "optimizer": "moar",
  "input_pipeline": "pipeline.yaml",
  "rewrite_agent_model": "gpt-5.1",
  "max_iterations": 40,
  "save_dir": "results/moar_optimization",
  "dataset": "transcripts",
  "start_time": "2024-01-15T10:30:00",
  "end_time": "2024-01-15T11:15:00",
  "duration_seconds": 2700,
  "num_best_nodes": 5,
  "total_nodes_explored": 120,
  "total_search_cost": 15.50
}
```

!!! info "Key Metrics"
    - `num_best_nodes`: Number of solutions on the Pareto frontier
    - `total_nodes_explored`: Total configurations tested
    - `total_search_cost`: Total cost of the optimization search

### pareto_frontier.json

List of Pareto-optimal solutions (the cost-accuracy frontier):

```json
[
  {
    "node_id": 5,
    "yaml_path": "results/moar_optimization/pipeline_5.yaml",
    "cost": 0.05,
    "accuracy": 0.92
  },
  {
    "node_id": 12,
    "yaml_path": "results/moar_optimization/pipeline_12.yaml",
    "cost": 0.08,
    "accuracy": 0.95
  }
]
```

!!! tip "Choosing a Solution"
    Review the Pareto frontier to find solutions that match your priorities:
    
    - **Low cost priority**: Choose solutions with lower cost
    - **High accuracy priority**: Choose solutions with higher accuracy
    - **Balanced**: Choose solutions in the middle

Each solution includes a `yaml_path` pointing to the optimized pipeline configuration.

### evaluation_metrics.json

Detailed evaluation results for all explored configurations. This file contains comprehensive metrics for every pipeline configuration tested during optimization.

### Pipeline Configurations

Each solution on the Pareto frontier has a corresponding YAML file (e.g., `pipeline_5.yaml`) containing the optimized pipeline configuration. You can:

1. Review the changes MOAR made
2. Test the pipeline on your full dataset
3. Use it in production

## Next Steps

After reviewing the results:

1. **Choose a solution** — Use `result.best()` / `result.cheapest()` in Python, or review `pareto_frontier.json` from the CLI
2. **Run the chosen pipeline** — Call `.run()` on the `OptimizedPipeline`, or run the YAML with `docetl run`
3. **Integrate into production** — Use the optimized configuration

!!! success "Success"
    You now have multiple optimized pipeline options to choose from, each representing a different point on the cost-accuracy trade-off curve.
