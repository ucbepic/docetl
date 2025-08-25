# Test Frontier Evaluation

This script runs the Pareto frontier plans on test datasets to evaluate generalization performance.

## How it works

1. Reads pareto frontier JSON files from Modal volume (e.g., `pareto_frontier_cuad.json`)
2. For each frontier point, finds the corresponding YAML pipeline file
3. Modifies the YAML to:
   - Use test data instead of train data (changes `/train/` to `/test/` in dataset path)
   - Save outputs to a `test_plans/` subdirectory
4. Runs each pipeline and evaluates accuracy on test data
5. Saves all results to `test_logs.json` in the `{dataset}_original` folder

## Usage

### Run for a specific dataset and method
```bash
modal run experiments/reasoning/run_test_frontier.py --dataset cuad --method mcts
```

### Run all methods for a specific dataset (default)
```bash
modal run experiments/reasoning/run_test_frontier.py --dataset cuad
```

### Run all methods for all datasets
```bash
modal run experiments/reasoning/run_test_frontier.py --dataset all
```

### Generate plot only (without running evaluations)
```bash
modal run experiments/reasoning/run_test_frontier.py --dataset cuad --plot-only
```

Note: Modal will automatically convert function parameters to command-line arguments.

## Available Options

**Datasets:**
- cuad
- blackvault
- game_reviews
- sustainability
- biodex
- medec
- facility
- all (runs all datasets)

**Methods:**
- simple_baseline
- baseline
- mcts
- all (runs all methods, default)

## Output Structure

Test results are integrated directly into the existing pareto frontier files:
- `/mnt/docetl-ro-experiments/outputs/{dataset}_{method}/pareto_frontier_{dataset}.json` - Updated with test results
- `/mnt/docetl-ro-experiments/outputs/{dataset}_{method}/test_plans/{method}/` - Directory containing test pipeline outputs for each method
- `/mnt/docetl-ro-experiments/outputs/{dataset}_original/test_frontier_plot.png` - Scatter plot showing test cost vs accuracy for all methods

## Results Format

The pareto frontier JSON files are updated with test results for each frontier point:
```json
{
  "frontier_points": [
    {
      "file": "cuad_modal_12.json",
      "iteration": 12,
      "cost": 0.23,
      "accuracy": 0.87,
      "accuracy_metric": "avg_f1",
      "test_cost": 0.25,
      "test_accuracy": 0.85,
      "test_accuracy_metric": "avg_f1"
    },
    {
      "file": "cuad_modal_15.json",
      "iteration": 15,
      "cost": 0.30,
      "accuracy": 0.91,
      "accuracy_metric": "avg_f1",
      "test_cost": 0.32,
      "test_accuracy": 0.89,
      "test_accuracy_metric": "avg_f1"
    }
  ],
  "test_evaluation": {
    "timestamp": "2024-01-20T10:30:00",
    "status": "completed"
  }
}
```

## Notes

- The script uses the same evaluation functions as the training pipeline
- Ground truth files are automatically selected based on the dataset
- All pipelines run with `bypass_cache=True` to ensure fresh execution
- Test data files must exist in `experiments/reasoning/data/test/`