# Understanding MOAR Results

What MOAR outputs and how to interpret the results.

## Output Files

After running MOAR optimization, you'll find several files in your `save_dir`:

- **`experiment_summary.json`** - High-level summary
- **`pareto_frontier.json`** - Optimal solutions
- **`evaluation_metrics.json`** - Detailed evaluation results
- **`pipeline_*.yaml`** - Optimized pipeline configurations

## experiment_summary.json

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

## pareto_frontier.json

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

## evaluation_metrics.json

Detailed evaluation results for all explored configurations. This file contains comprehensive metrics for every pipeline configuration tested during optimization.

## Pipeline Configurations

Each solution on the Pareto frontier has a corresponding YAML file (e.g., `pipeline_5.yaml`) containing the optimized pipeline configuration. You can:

1. Review the changes MOAR made
2. Test the pipeline on your full dataset
3. Use it in production

## Next Steps

After reviewing the results:

1. **Review the Pareto frontier** - See available options
2. **Choose a solution** - Based on your accuracy/cost priorities
3. **Test the chosen pipeline** - Run it on your full dataset
4. **Integrate into production** - Use the optimized configuration

!!! success "Success"
    You now have multiple optimized pipeline options to choose from, each representing a different point on the cost-accuracy trade-off curve.

