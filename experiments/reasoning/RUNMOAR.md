# Running MOARSearch and Simple Agent Experiments

This guide explains how to run the MOAR optimizer `run_moar.py` and the simple agent optimizer `run_simple_agent.py`.

## Table of Contents

- [Prerequisites](#prerequisites)
- [MOAR](#moarsearch-run_moarpy)
- [Simple Agent](#simple-agent-run_simple_agentpy)
- [Supported Datasets](#supported-datasets)
  - [Custom Datasets with User-Authored Accuracy Functions](#custom-datasets-with-user-authored-accuracy-functions)
- [Available Models](#available-models)
- [Output Files](#output-files)

## Prerequisites

### Dataset and Pipeline Files

Download the 6 workload experiment datasets and initial pipelines from the [Google Drive Link](https://drive.google.com/drive/folders/1pNFqYCguAZL3iGYHd-jDoialrtf73fbR?usp=drive_link). Extract the `data/` folder and `pipelines/` folder to `experiments/reasoning/` directory.

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
  

- **`EXPERIMENT_DATA_DIR`**: Directory containing input data files. Defaults to `./data/` if not set. Can also be set via `--data_dir` parameter in `run_moar.py`.
  

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
| `--yaml_path` | ✅ Yes | - | Path to the user-authored input YAML pipeline file |
| `--dataset_path` | ✅ Yes | - | Path to the dataset file for sample input data |
| `--experiment_name` | ✅ Yes | - | Unique experiment identifier |
| `--dataset` | Conditional | `cuad` | Dataset name for evaluation. **Required** if `--accuracy_function` is not provided. Must be one of: cuad, blackvault, medec, biodex, sustainability, game_reviews. If using `--accuracy_function`, can be any string (used for naming). |
| `--max_iterations` | No | `100` | Maximum MOARSearch iterations |
| `--exploration_weight` | No | `1.414` | UCB exploration parameter c (controls exploration vs exploitation) |
| `--model` | No | `gpt-5` | LLM model to use for directive instantiation |
| `--available_models` | No | All 11 default models (gpt-5, gpt-5-mini, gpt-5-nano, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, gpt-4o, gpt-4o-mini, gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite) | Space-separated list of models for first layer testing. Example: `gpt-5 gpt-5-mini gpt-4o` |
| `--data_dir` | No | `EXPERIMENT_DATA_DIR` env var or `./data/` | Directory containing input data files |
| `--output_dir` | No | `EXPERIMENT_OUTPUT_DIR` env var or `./outputs` | Directory to save experiment outputs |
| `--ground_truth` | No | - | Path to ground-truth file |
| `--accuracy_function` | No | - | Path to Python file containing custom `evaluate_results` function (for user datasets) |
| `--accuracy_metric_key` | No | - | Key to extract from evaluation results dict for accuracy metric (required with `--accuracy_function`) |

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
| `--dataset` | Conditional | - | Dataset name. **Required** if `--accuracy_function` is not provided. Must be one of: cuad, blackvault, medec, biodex, sustainability, game_reviews. If using `--accuracy_function`, can be any string (used for naming). |
| `--model` | No | `gpt-5` | LLM model to use for optimization |
| `--experiment_name` | No | `simple_agent_{dataset}` | Unique experiment identifier |
| `--output_dir` | No | `outputs/simple_agent` | Output directory |
| `--ground_truth` | No | - | Path to ground truth file for evaluation |
| `--available_models` | No | All 11 default models (gpt-5, gpt-5-mini, gpt-5-nano, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, gpt-4o, gpt-4o-mini, gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite) | Space-separated list of available models. Example: `gpt-5 gpt-5-mini gpt-4o` |
| `--accuracy_function` | No | - | Path to Python file containing custom `evaluate_results` function (for user datasets) |
| `--accuracy_metric_key` | No | - | Key to extract from evaluation results dict for accuracy metric (required with `--accuracy_function`) |

## Supported Datasets

Both `run_moar.py` and `run_simple_agent.py` support the following datasets:

- `cuad` - Legal clause extraction
- `blackvault` - UFO sighting analysis
- `medec` - Medical entity classification
- `biodex` - Biochemical reaction prediction
- `sustainability` - Sustainability analysis
- `game_reviews` - Game review sentiment analysis

### Custom Datasets with User-Authored Accuracy Functions

For datasets not listed above, you can provide your own accuracy evaluation function using the `--accuracy_function` and `--accuracy_metric_key` parameters.

#### Creating a Custom Accuracy Function

Create a Python file (e.g., `my_evaluate.py`) with an `evaluate_results` function:

```python
# my_evaluate.py
import json

def evaluate_results(method_name, results_file_path):
    """
    Evaluate pipeline results and return metrics.
    
    Args:
        method_name: Name of the method being evaluated
        results_file_path: Path to the JSON file containing pipeline results
        
    Returns:
        dict: Dictionary containing evaluation metrics. Must include the metric
              specified by --accuracy_metric_key.
    """
    # Load results
    with open(results_file_path, 'r') as f:
        results = json.load(f)
    
    # Your evaluation logic here
    # Calculate metrics based on your dataset's requirements
    
    metrics = {
        "my_accuracy_metric": 0.95,  # Primary accuracy metric (specify key with --accuracy_metric_key)
        # Add any other metrics you want to track
    }
    
    return metrics
```

#### Using Custom Accuracy Functions

**MOARSearch:**
```bash
python experiments/reasoning/run_moar.py \
  --yaml_path my_pipeline.yaml \
  --dataset_path my_data.json \
  --experiment_name my_custom_experiment \
  --dataset my_custom_dataset \
  --accuracy_function my_evaluate.py \
  --accuracy_metric_key my_accuracy_metric
```

**Simple Agent:**
```bash
python experiments/reasoning/run_simple_agent.py \
  --dataset my_custom_dataset \
  --accuracy_function my_evaluate.py \
  --accuracy_metric_key my_accuracy_metric
```

**Note:** When using custom accuracy functions, the `--dataset` parameter can be any string (it's used for naming/organization). The actual evaluation logic comes from your custom function.

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
