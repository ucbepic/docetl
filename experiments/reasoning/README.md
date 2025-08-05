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
  --max_iterations 10 \
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
  --dataset_path experiments/reasoning/data/reviews.json \
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
  --dataset_path experiments/reasoning/data/company_reports_sample.json \
  --experiment_name sustainability_mcts \
  --max_iterations 10 \
  --model gpt-4.1 \
  --dataset sustainability
```

## Output

Results are saved to `outputs/{experiment_name}/` containing:
- `experiment_summary.json` - Experiment metadata and results
- For CUAD: `cost_vs_f1.png` - Plot of cost vs F1 score
- For BlackVault: `cost_vs_avg_locations.png` - Plot of cost vs average distinct locations
- For Game Reviews: Performance metrics for review analysis and selection
- For MEDEC: `cost_vs_combined_score.png` - Plot showing medical error detection performance
- For Sustainability: Performance metrics for economic activity classification and sustainability analysis

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--yaml_path` | Required | Path to input pipeline YAML |
| `--dataset_path` | Required | Path to dataset file for sample input data |
| `--experiment_name` | Required | Unique experiment identifier |
| `--iterations` | `1` | Number of optimization rounds (baseline) |
| `--max_iterations` | `100` | MCTS search iterations |
| `--model` | `gpt-4o-mini` | LLM model to use |