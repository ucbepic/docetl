# DocETL Reasoning Optimizer Experiments

This folder contains experiments for the DocETL reasoning optimizer, which uses AI agents and Monte Carlo Tree Search (MCTS) to automatically optimize document processing pipelines.

## Running Experiments

### CUAD Experiments

### Prerequisites

Pip install the following:

```bash
pip install pymoo
```

Set up environment variables:
```bash
export AZURE_API_KEY="your_key_here"
export AZURE_API_BASE="your_endpoint_here" 
export AZURE_API_VERSION="your_version_here"
```

### CUAD Baseline

Run the baseline agent experiment:

```bash
python experiments/reasoning/run_baseline.py \
  --yaml_path experiments/reasoning/pipelines/cuad.yaml \
  --experiment_name cuad_baseline \
  --iterations 5 \
  --model gpt-4.1
```

### CUAD MCTS

Run the MCTS optimization experiment:

```bash
python experiments/reasoning/run_mcts.py \
  --yaml_path experiments/reasoning/pipelines/cuad.yaml \
  --dataset_path experiments/reasoning/data/CUAD_input_data.json \
  --experiment_name cuad_mcts \
  --max_iterations 5 \
  --model gpt-4.1
```

### BlackVault Experiments

#### BlackVault Baseline

Run the baseline agent experiment:

```bash
python experiments/reasoning/run_baseline.py \
  --yaml_path experiments/reasoning/pipelines/blackvault.yaml \
  --experiment_name blackvault_baseline \
  --iterations 5 \
  --model gpt-4.1 \
  --dataset blackvault
```

#### BlackVault MCTS

Run the MCTS optimization experiment:

```bash
python experiments/reasoning/run_mcts.py \
  --yaml_path experiments/reasoning/pipelines/blackvault.yaml \
  --dataset_path experiments/reasoning/data/blackvault_articles_pdfs.json \
  --experiment_name blackvault_mcts \
  --max_iterations 10 \
  --model gpt-4.1 \
  --dataset blackvault
```

### Game Reviews Experiments

#### Game Reviews Baseline

Run the baseline agent experiment:

```bash
python experiments/reasoning/run_baseline.py \
  --yaml_path experiments/reasoning/pipelines/game_reviews.yaml \
  --experiment_name game_reviews_baseline \
  --iterations 5 \
  --model gpt-4.1 \
  --dataset game_reviews
```

#### Game Reviews MCTS

Run the MCTS optimization experiment:

```bash
python experiments/reasoning/run_mcts.py \
  --yaml_path experiments/reasoning/pipelines/game_reviews.yaml \
  --dataset_path experiments/reasoning/data/train/game_reviews.json \
  --experiment_name game_reviews_mcts \
  --max_iterations 10 \
  --model gpt-4.1 \
  --dataset game_reviews
```

### MEDEC Experiments

#### MEDEC Baseline

Run the baseline agent experiment:

```bash
python experiments/reasoning/run_baseline.py \
  --yaml_path experiments/reasoning/pipelines/medec.yaml \
  --experiment_name medec_baseline \
  --iterations 5 \
  --model gpt-4.1 \
  --dataset medec
```

#### MEDEC MCTS

Run the MCTS optimization experiment:

```bash
python experiments/reasoning/run_mcts.py \
  --yaml_path experiments/reasoning/pipelines/medec.yaml \
  --dataset_path experiments/reasoning/data/medec_sample_50.json \
  --experiment_name medec_mcts \
  --max_iterations 10 \
  --model gpt-4.1 \
  --dataset medec
```

### Running on Modal (shared volume)

- Baseline (MEDEC):

```bash
modal run experiments/reasoning/run_baseline.py \
  --yaml-path=experiments/reasoning/pipelines/medec.yaml \
  --experiment-name=medec_baseline \
  --dataset=medec \
  --iterations=3
```

- MCTS (MEDEC):

```bash
modal run experiments/reasoning/run_mcts.py \
  --yaml-path=experiments/reasoning/pipelines/medec.yaml \
  --dataset-path=experiments/reasoning/data/train/medec.json \
  --experiment-name=medec_mcts \
  --dataset=medec
  --max-iterations=30
```

Outputs are written to `/mnt/docetl-ro-experiments/outputs/{experiment_name}` in the shared Modal volume.

#### Run multiple datasets at once with Modal using run_all.py

Use `experiments/reasoning/run_all.py` to spawn baseline and MCTS runs together per dataset.

1) Edit the inline `CONFIG` in `experiments/reasoning/run_all.py` to list your datasets and settings. Example:

```python
CONFIG = {
    "experiments": [
        {
            "dataset": "medec",
            "baseline": {"iterations": 3, "model": "gpt-4o-mini"},
            "mcts": {"max_iterations": 30, "exploration_weight": 1.414, "model": "gpt-4o-mini"}
        },
        {
            "dataset": "biodex",
            "baseline": {"iterations": 5},
            "mcts": {"max_iterations": 50}
        }
    ]
}
```

- Defaults for pipeline YAMLs and dataset inputs are provided internally for these dataset keys: `cuad`, `blackvault`, `game_reviews`, `sustainability`, `biodex`, `medec`.
- You can override paths per experiment with optional keys: `yaml_path`, `dataset_path`, `data_dir`, `output_dir`, `ground_truth`.

2) Run all experiments on Modal (uses the same shared volume as above):

```bash
modal run --detach experiments/reasoning/run_all.py
```

This will spawn the requested baseline and MCTS jobs concurrently per dataset and wait for completion. Outputs are saved under `/mnt/docetl-ro-experiments/outputs/{dataset}_{baseline|mcts}`.

### Sustainability Experiments

#### Sustainability Baseline

Run the baseline agent experiment:

```bash
python experiments/reasoning/run_baseline.py \
  --yaml_path experiments/reasoning/pipelines/sustainability.yaml \
  --experiment_name sustainability_baseline \
  --iterations 5 \
  --model gpt-4.1 \
  --dataset sustainability
```

#### Sustainability MCTS

Run the MCTS optimization experiment:

```bash
python experiments/reasoning/run_mcts.py \
  --yaml_path experiments/reasoning/pipelines/sustainability.yaml \
  --dataset_path experiments/reasoning/data/company_reports_gt.json \
  --experiment_name sustainability_mcts \
  --max_iterations 10 \
  --model gpt-4.1 \
  --dataset sustainability
```

### BioDEX Experiments

BioDEX is a biomedical dataset for extracting adverse drug reactions from scientific papers. The task involves identifying and matching reactions to standardized medical terms.

#### BioDEX Baseline

Run the baseline agent experiment:

```bash
python experiments/reasoning/run_baseline.py \
  --yaml_path experiments/reasoning/pipelines/biodex.yaml \
  --experiment_name biodex_baseline \
  --iterations 5 \
  --model gpt-4.1 \
  --dataset biodex
```

#### BioDEX MCTS

Run the MCTS optimization experiment:

```bash
python experiments/reasoning/run_mcts.py \
  --yaml_path experiments/reasoning/pipelines/biodex.yaml \
  --dataset_path experiments/reasoning/data/train/biodex.json \
  --experiment_name biodex_mcts \
  --max_iterations 10 \
  --model gpt-4.1 \
  --dataset biodex
```

## Simple Baseline Agent

The simple baseline agent is an extremely lightweight alternative to the full reasoning optimizer. It uses basic tool calling to generate and test pipeline configurations with minimal complexity.

### What is the Simple Baseline?

The simple baseline agent:

1. **Loads operator documentation** - Reads available DocETL operators and their usage
2. **Analyzes sample data** - Examines the structure of your dataset 
3. **Uses tool calling** - Leverages LLM tool calling to generate pipeline configurations
4. **Tests iteratively** - Runs a few iterations (baseline + 3 agent iterations) to find improvements
5. **Returns best pipeline** - Outputs the best performing configuration found

This is much simpler than the full MCTS approach, making it ideal for:
- Quick experimentation
- Understanding baseline performance
- Simple optimization tasks
- Testing new datasets

### Running Simple Baseline

#### Local execution:
```bash
python experiments/reasoning/run_simple_baseline.py \
  --dataset medec \
  --model gpt-4o-mini \
  --experiment_name medec_simple_baseline
```

#### Modal execution (recommended):
```bash
modal run experiments/reasoning/run_simple_baseline.py --dataset=medec
```

Other supported datasets: `cuad`, `blackvault`, `reviews`, `sustainability`, `biodex`, `medec`

### Simple Baseline Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | Required | Dataset name (cuad, reviews, blackvault, etc.) |
| `--model` | `o3` | LLM model to use for optimization |
| `--experiment_name` | `simple_baseline_{dataset}` | Unique experiment identifier |
| `--output_dir` | `outputs/simple_baseline` | Output directory |
| `--ground_truth` | `None` | Path to ground truth file for evaluation |

### Simple Baseline Output

Results are saved to `outputs/simple_baseline_{dataset}/` containing:
- `operators.json` - Final optimized operators
- `results.json` - Experiment results and metrics
- `evaluation_metrics.json` - Performance evaluation results
- `cost_vs_{metric}.png` - Plot showing cost vs performance across iterations (0, 1, 2, 3, 4)
- `final_pipeline.yaml` - Complete pipeline configuration
- Pipeline outputs for each iteration

The plot shows 5 data points representing:
- **0**: Original baseline pipeline
- **1-3**: Agent optimization iterations  
- **4**: Final optimized pipeline

## Output

Results are saved to `outputs/{experiment_name}/` containing:
- `experiment_summary.json` - Experiment metadata and results
- For CUAD: `cost_vs_f1.png` - Plot of cost vs F1 score
- For BlackVault: `cost_vs_avg_locations.png` - Plot of cost vs average distinct locations
- For Game Reviews: Performance metrics for review analysis and selection
- For MEDEC: `cost_vs_combined_score.png` - Plot showing medical error detection performance
- For Sustainability: Performance metrics for economic activity classification and sustainability analysis
- For BioDEX: `cost_vs_rp_at_10.png` - Plot of cost vs Rank Precision @ 10 (primary optimization metric)

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--yaml_path` | Required | Path to input pipeline YAML |
| `--dataset_path` | Required | Path to dataset file for sample input data |
| `--experiment_name` | Required | Unique experiment identifier |
| `--iterations` | `1` | Number of optimization rounds (baseline) |
| `--max_iterations` | `100` | MCTS search iterations |
| `--model` | `gpt-4o-mini` | LLM model to use |