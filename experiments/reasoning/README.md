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
  --experiment_name blackvault_mcts \
  --max_iterations 10 \
  --model gpt-4.1 \
  --dataset blackvault
```

## Output

Results are saved to `outputs/{experiment_name}/` containing:
- `experiment_summary.json` - Experiment metadata and results
- For CUAD: `cost_vs_f1.png` - Plot of cost vs F1 score
- For BlackVault: `cost_vs_avg_locations.png` - Plot of cost vs average distinct locations

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--yaml_path` | Required | Path to input pipeline YAML |
| `--experiment_name` | Required | Unique experiment identifier |
| `--iterations` | `1` | Number of optimization rounds (baseline) |
| `--max_iterations` | `100` | MCTS search iterations |
| `--model` | `gpt-4o-mini` | LLM model to use |