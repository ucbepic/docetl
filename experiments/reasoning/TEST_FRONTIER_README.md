# Test Frontier Evaluation

This script runs the Pareto frontier plans on test datasets to evaluate generalization performance and generates comparison matrices.

## How it works

1. Reads pareto frontier JSON files from Modal volume (e.g., `pareto_frontier_cuad.json`)
2. For each frontier point, finds the corresponding YAML pipeline file
3. Modifies the YAML to:
   - Use test data instead of train data (changes `/train/` to `/test/` in dataset path)
   - Save outputs to a `test_plans/` subdirectory
4. Runs each pipeline and evaluates accuracy on test data
5. Saves all results to `test_frontier_summary.json` in the `{dataset}_original_final` folder
6. Generates comparison matrices and plots

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

### Generate cost savings matrices only (without running evaluations)
```bash
modal run experiments/reasoning/run_test_frontier.py --dataset cuad --matrix-only
```

### Run top 2 accuracy tradeoff analysis for MCTS across all datasets
```bash
modal run experiments/reasoning/run_test_frontier.py --tradeoff
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
- mcts
- all (runs all methods, default)

**Flags:**
- `--plot-only`: Only generate the test frontier plot without running evaluations
- `--matrix-only`: Only generate cost savings matrices without running evaluations
- `--tradeoff`: Run top 2 accuracy tradeoff analysis for MCTS across all datasets

## Output Structure

Test results are saved in the following structure:

### Test Results
- `/mnt/docetl-ro-experiments/outputs/{dataset}_original_final/test_frontier_summary.json` - Summary of all test results
- `/mnt/docetl-ro-experiments/outputs/{dataset}_{method}_final/pareto_frontier_{dataset}.json` - Updated with test results for each frontier point
- `/mnt/docetl-ro-experiments/outputs/{dataset}_{method}_final/test_plans/{method}/` - Directory containing test pipeline outputs for each method

### Plots
- `/mnt/docetl-ro-experiments/outputs/{dataset}_original_final/test_frontier_plot.pdf` - Scatter plot showing test cost vs accuracy for all methods

### Matrices
- `/mnt/docetl-ro-experiments/outputs/{dataset}_original_final/best_cost_savings_matrix.json` - Best cost savings matrix (JSON format)
- `/mnt/docetl-ro-experiments/outputs/{dataset}_original_final/avg_cost_savings_matrix.json` - Average cost savings matrix (JSON format)

## Results Format

### Test Frontier Summary

The `test_frontier_summary.json` file contains results for all methods:

```json
{
  "dataset": "cuad",
  "timestamp": "2024-01-20T10:30:00",
  "methods_processed": ["original", "simple_baseline", "mcts"],
  "successful_methods": ["original", "simple_baseline", "mcts"],
  "failed_methods": [],
  "results": {
    "original": {
      "success": true,
      "cost": 0.25,
      "accuracy": 0.85,
      "accuracy_metric": "avg_f1"
    },
    "mcts": {
      "success": true,
      "results": [
        {
          "file": "cuad_modal_12",
          "cost": 0.25,
          "accuracy": 0.85,
          "accuracy_metric": "avg_f1"
        }
      ]
    }
  }
}
```

### Pareto Frontier Files

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
    }
  ],
  "test_evaluation": {
    "timestamp": "2024-01-20T10:30:00",
    "status": "completed"
  }
}
```

## Cost Savings Matrices

The script generates two types of cost savings matrices:

### Best Cost Savings Matrix

Shows how much cost method A (row header) saves for achieving or surpassing the highest accuracy of method B (column header). Each cell contains:
- **absolute**: Absolute cost savings in dollars
- **ratio**: Cost ratio (what fraction of method B's cost method A uses)

### Average Cost Savings Matrix

Shows the average cost savings when method A (row header) achieves the same or higher accuracy as each plan in method B (column header). Each cell contains:
- **absolute**: Average absolute cost savings
- **ratio**: Average cost ratio

'n/a' indicates method A cannot achieve method B's accuracy; '-' indicates method B does not achieve the original accuracy. Both matrices are printed to console and saved as JSON files.

## Notes

- The script uses the same evaluation functions as the training pipeline
- Ground truth files are automatically selected based on the dataset
- All pipelines run with `bypass_cache=True` to ensure fresh execution
- Test data files must exist in `experiments/reasoning/data/test/`