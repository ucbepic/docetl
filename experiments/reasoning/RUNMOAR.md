# Running MOARSearch and Simple Agent Experiments

This guide explains how to run the MOAR optimizer `run_moar.py` and the simple agent optimizer `run_simple_agent.py`.

## Table of Contents

- [Prerequisites](#prerequisites)
- [MOAR (`run_moar.py`)](#moarsearch-run_moarpy)
- [Simple Agent (`run_simple_agent.py`)](#simple-agent-run_simple_agentpy)
- [Supported Datasets](#supported-datasets)
- [Available Models](#available-models)
- [Output Files](#output-files)

## Prerequisites

### Python Dependencies

Install the required Python packages:

```bash
pip install -r experiments/reasoning/requirements.txt
```

### Environment Variables

#### Required Environment Variables

Set up your Azure API credentials (required for LLM calls):

```bash
export AZURE_API_KEY="your_key_here"
export AZURE_API_BASE="your_endpoint_here" 
export AZURE_API_VERSION="your_version_here"
```

#### Optional Environment Variables

- **`EXPERIMENT_OUTPUT_DIR`**: Directory to save experiment outputs. If not set, defaults are used:
  - `run_moar.py`: `./outputs` 
  - `run_simple_agent.py`: `./outputs/simple_agent`
  

- **`EXPERIMENT_DATA_DIR`**: Directory containing input data files. Can also be set via `--data_dir` parameter in `run_moar.py`.
  

## MOARSearch (`run_moar.py`)

MOARSearch is a multi-objective optimization algorithm that uses graph search to find Pareto-optimal pipelines balancing cost and accuracy.

### Local Execution

#### Basic Usage

```bash
python experiments/reasoning/run_moar.py \
  --yaml_path experiments/reasoning/pipelines/cuad.yaml \
  --dataset_path experiments/reasoning/data/train/cuad.json \
  --experiment_name cuad_moar \
  --dataset cuad
```

### Modal Execution (Recommended)

Modal execution is recommended for longer-running experiments as it provides better resource management and persistence.

```bash
modal run experiments/reasoning/run_moar.py \
  --yaml-path=experiments/reasoning/pipelines/medec.yaml \
  --dataset-path=experiments/reasoning/data/train/medec.json \
  --experiment-name=medec_moar \
  --dataset=medec \
  --max-iterations=30
```

Outputs are written to `/mnt/docetl-ro-experiments/outputs/{experiment_name}` in the shared Modal volume.

### Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--yaml_path` | ✅ Yes | - | Path to the input YAML pipeline file |
| `--dataset_path` | ✅ Yes | - | Path to the dataset file for sample input data |
| `--experiment_name` | ✅ Yes | - | Unique experiment identifier |
| `--dataset` | No | `cuad` | Dataset name for evaluation (cuad, blackvault, medec, etc.) |
| `--max_iterations` | No | `100` | Maximum MOARSearch iterations |
| `--exploration_weight` | No | `1.414` | UCB exploration parameter c (controls exploration vs exploitation) |
| `--model` | No | `gpt-5` | LLM model to use for directive instantiation |
| `--available_models` | No | All models | Space-separated list of models for first layer testing. Example: `gpt-5 gpt-5-mini gpt-4o` |
| `--data_dir` | No | - | Directory containing input data files |
| `--output_dir` | No | `EXPERIMENT_OUTPUT_DIR` env var | Directory to save experiment outputs |
| `--ground_truth` | No | - | Path to ground-truth file (if not default) |

## Simple Agent (`run_simple_agent.py`)

The Simple Agent is a baseline optimizer that uses basic tool calling to generate and test pipeline configurations iteratively. 

### Local Execution

#### Basic Usage

```bash
python experiments/reasoning/run_simple_agent.py \
  --dataset medec \
  --model gpt-4o-mini
```

### Modal Execution (Recommended)

```bash
modal run experiments/reasoning/run_simple_agent.py \
  --dataset=medec \
  --model=gpt-4o-mini \
  --experiment-name=medec_simple
```

### Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--dataset` | ✅ Yes | - | Dataset name (cuad, blackvault, medec, biodex, sustainability, game_reviews, facility) |
| `--model` | No | `gpt-5` | LLM model to use for optimization |
| `--experiment_name` | No | `simple_agent_{dataset}` | Unique experiment identifier |
| `--output_dir` | No | `outputs/simple_agent` | Output directory |
| `--ground_truth` | No | - | Path to ground truth file for evaluation |
| `--available_models` | No | All models | Space-separated list of available models. Example: `gpt-5 gpt-5-mini gpt-4o` |

## Supported Datasets

Both `run_moar.py` and `run_simple_agent.py` support the following datasets:

- `cuad` - Legal clause extraction
- `blackvault` - UFO sighting analysis
- `medec` - Medical entity classification
- `biodex` - Biochemical reaction prediction
- `sustainability` - Sustainability analysis
- `game_reviews` - Game review sentiment analysis
- `facility` - Facility support message classification (Simple Agent only)

## Available Models

Both scripts support the `--available_models` parameter to specify which models should be tested. If not provided, the following default model list is used:

- `gpt-5`
- `gpt-5-mini`
- `gpt-5-nano`
- `gpt-4.1`
- `gpt-4.1-mini`
- `gpt-4.1-nano`
- `gpt-4o`
- `gpt-4o-mini`
- `gemini-2.5-pro`
- `gemini-2.5-flash`
- `gemini-2.5-flash-lite`


## Output Files

### MOARSearch Output

Results are saved to `outputs/{experiment_name}/` containing:

- `experiment_summary.json` - Experiment metadata and results
- `pareto_frontier.json` - All Pareto-optimal solutions with accuracy and cost
- `evaluation_metrics.json` - Detailed evaluation results
- `moar_tree_log.txt` - Search tree structure and visit counts
- `cost_vs_{metric}.png` - Plot showing cost vs performance (dataset-specific)
- Pipeline YAML files for each explored configuration

### Simple Agent Output

Results are saved to `outputs/simple_agent/{experiment_name}/` containing:

- `final_pipeline.yaml` - Final optimized pipeline configuration
- `iteration_{n}_output.json` - Pipeline outputs for each iteration
- `evaluation_metrics.json` - Performance evaluation results
- `cost_vs_{metric}.png` - Plot showing cost vs performance across iterations
