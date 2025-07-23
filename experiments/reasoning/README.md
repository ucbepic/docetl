# DocETL Reasoning Optimizer Experiments

This folder contains experiments for the DocETL reasoning optimizer, which uses AI agents and Monte Carlo Tree Search (MCTS) to automatically optimize document processing pipelines by suggesting and applying rewrite directives.

## 📁 Folder Structure

```
experiments/reasoning/
├── README.md              # This file
├── run_baseline.py        # Baseline agent experiment
├── run_mcts.py           # MCTS optimization experiment
├── run_tests.py          # Directive testing (links to test runner)
├── data/                 # Input data for experiments
├── outputs/              # Experiment results
│   ├── baseline_exp1/    # Baseline experiment outputs
│   ├── mcts_exp1/        # MCTS experiment outputs
│   └── ...
└── __init__.py
```

## 🎯 Experiments Overview

### 1. **Baseline Agent** (`run_baseline.py`)
- Uses AI agent to suggest single directive optimizations
- Iterative refinement with conversation history
- Fast, single-shot optimization suggestions

### 2. **MCTS Optimization** (`run_mcts.py`) 
- Uses Monte Carlo Tree Search to explore directive space
- Multi-objective optimization (accuracy vs cost)
- Finds Pareto frontier of optimal configurations
- More thorough but slower exploration

### 3. **Directive Testing** (`run_tests.py`)
- Tests all directives for correct instantiation
- Validates directive logic with LLM judges  
- Ensures system reliability
- Links to the main test runner in `tests/reasoning_optimizer/`

## 🚀 Quick Start

### Prerequisites

1. **Environment Setup**:
   ```bash
   # Set up environment variables
   export AZURE_API_KEY="your_key_here"
   export AZURE_API_BASE="your_endpoint_here" 
   export AZURE_API_VERSION="your_version_here"
   
   # Optional: Set experiment data directory (defaults to ./data/)
   export EXPERIMENT_DATA_DIR="/path/to/your/experiment/data"
   
   # Optional: Set experiment output directory (defaults to ./outputs/)
   export EXPERIMENT_OUTPUT_DIR="/path/to/your/experiment/outputs"
   ```

2. **Data Preparation**:
   - Place your pipeline YAML file anywhere accessible
   - Ensure required data files are in your experiment data directory:
     - `CUAD_random_sample.json` - Sample input data
     - `CUAD_input_data.json` - Full input data (optional)

### Running Experiments

**Note: All commands should be run from the project root directory.**

#### 1. Baseline Agent Experiment

```bash
# Basic run
python experiments/reasoning/run_baseline.py --yaml_path ./my_pipeline.yaml --experiment_name baseline_test

# With custom data and output directories
python experiments/reasoning/run_baseline.py \
  --yaml_path ./pipelines/contract_analysis.yaml \
  --data_dir ./experiment_data \
  --output_dir ./results \
  --experiment_name contract_baseline_v1

# Using environment variables (no need for --data_dir and --output_dir)
export EXPERIMENT_DATA_DIR="./experiment_data"
export EXPERIMENT_OUTPUT_DIR="./results"
python experiments/reasoning/run_baseline.py \
  --yaml_path ./pipelines/contract_analysis.yaml \
  --experiment_name contract_baseline_v2

# Multiple iterations with different model
python experiments/reasoning/run_baseline.py \
  --yaml_path ./my_pipeline.yaml \
  --iterations 3 \
  --model gpt-4o \
  --experiment_name multi_iteration_test
```

#### 2. MCTS Optimization Experiment

```bash
# Basic MCTS run
python experiments/reasoning/run_mcts.py --yaml_path ./my_pipeline.yaml --experiment_name mcts_test

# Deep exploration with custom parameters
python experiments/reasoning/run_mcts.py \
  --yaml_path ./pipelines/complex_analysis.yaml \
  --data_dir ./experiment_data \
  --output_dir ./results \
  --max_iterations 200 \
  --exploration_weight 2.0 \
  --experiment_name mcts_deep_exploration

# Using environment variables for cleaner commands
export EXPERIMENT_DATA_DIR="./experiment_data"
export EXPERIMENT_OUTPUT_DIR="./results"
python experiments/reasoning/run_mcts.py \
  --yaml_path ./pipelines/complex_analysis.yaml \
  --max_iterations 200 \
  --exploration_weight 2.0 \
  --experiment_name mcts_deep_exploration

# Quick exploration 
python experiments/reasoning/run_mcts.py \
  --yaml_path ./my_pipeline.yaml \
  --max_iterations 50 \
  --experiment_name mcts_quick
```

#### 3. Testing Directives

```bash
# Test all directives
python experiments/reasoning/run_tests.py

# Test specific directive  
python experiments/reasoning/run_tests.py --directive chaining

# Test with different model
python experiments/reasoning/run_tests.py --model gpt-4o

# Or run directly from tests folder
python tests/reasoning_optimizer/test_runner.py
```

## 📊 Understanding Outputs

### Baseline Experiment Outputs

After running a baseline experiment, you'll find in `outputs/{experiment_name}/`:

- **`experiment_summary.json`** - Overall experiment metadata and results
- **`message_history.json`** - Complete conversation history with the AI agent
- **`iteration_N_output.yaml`** - Optimized pipeline from iteration N
- **`original_output.json`** - Original pipeline output (if provided)

```json
{
  "experiment_name": "baseline_test",
  "success_rate": 1.0,
  "results": [
    {
      "iteration": 1,
      "output_file": "./outputs/baseline_test/iteration_1_output.yaml",
      "success": true
    }
  ]
}
```

### MCTS Experiment Outputs

After running an MCTS experiment, you'll find in `outputs/{experiment_name}/`:

- **`experiment_summary.json`** - Experiment metadata and statistics
- **`best_config_N.yaml`** - Top N optimized pipeline configurations  
- **`pareto_frontier.json`** - Pareto-optimal solutions (accuracy vs cost)

```json
{
  "experiment_name": "mcts_test",
  "duration_seconds": 125.3,
  "num_best_nodes": 3,
  "best_configurations": [
    {
      "config_id": 1,
      "config_file": "./outputs/mcts_test/best_config_1.yaml",
      "value": 0.87,
      "visits": 12
    }
  ]
}
```

## ⚙️ Configuration Options

### Baseline Agent Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--yaml_path` | Required | Path to input pipeline YAML |
| `--data_dir` | `./data/` | Directory containing input data (sets EXPERIMENT_DATA_DIR) |
| `--output_dir` | `./outputs` | Where to save results (overrides EXPERIMENT_OUTPUT_DIR) |
| `--model` | `gpt-4o-mini` | LLM model to use |
| `--iterations` | `1` | Number of optimization rounds |
| `--max_tpm` | `5000000` | Tokens per minute limit |
| `--experiment_name` | Required | Unique experiment identifier |

### MCTS Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--yaml_path` | Required | Path to input pipeline YAML |
| `--data_dir` | `./data/` | Directory containing input data (sets EXPERIMENT_DATA_DIR) |
| `--output_dir` | `./outputs` | Where to save results (overrides EXPERIMENT_OUTPUT_DIR) |
| `--model` | `gpt-4o-mini` | LLM model for directive instantiation |
| `--max_iterations` | `100` | MCTS search iterations |
| `--exploration_weight` | `1.414` | UCB exploration parameter (c) |
| `--experiment_name` | Required | Unique experiment identifier |

## 🧪 Available Directives

The system can optimize pipelines using these directives:

1. **Chaining**: Decompose complex operations into sequential steps
2. **Gleaning**: Add validation loops to improve output quality  
3. **Change Model**: Switch to more appropriate LLM models

Each directive has embedded test cases that validate proper instantiation.

## 📈 Comparing Experiments

### Baseline vs MCTS

- **Baseline**: Fast, single suggestions, good for quick optimization
- **MCTS**: Thorough exploration, finds multiple optimal solutions, better for complex pipelines

### Choosing Parameters

- **High `exploration_weight`** (>2.0): More exploration, finds diverse solutions
- **Low `exploration_weight`** (<1.0): More exploitation, focuses on promising areas
- **More `iterations`**: Better solutions but longer runtime

## 🐛 Troubleshooting

### Common Issues

1. **Missing Data Files**:
   ```
   Error: CUAD_random_sample.json not found
   ```
   **Solution**: Ensure data files are in your `EXPERIMENT_DATA_DIR` or default `./data/` folder

2. **Import Errors**:
   ```
   ImportError: No module named 'docetl'
   ```
   **Solution**: Run from project root or ensure proper Python path setup

3. **API Key Issues**:
   ```
   Error: Azure API key not found
   ```
   **Solution**: Set environment variables for Azure OpenAI access

4. **Rate Limiting**:
   ```
   Rate limit exceeded
   ```
   **Solution**: Reduce `max_tpm` parameter or use slower model

### Debug Mode

Add debug prints by setting:
```bash
export DOCETL_DEBUG=1
```

## 📚 Further Reading

- **Core Modules**: See `docetl/reasoning_optimizer/` for the underlying implementation
- **Directives**: Check `docetl/reasoning_optimizer/directives/` for directive definitions
- **MCTS Implementation**: Explore `MCTS/` folder for tree search details

## 🤝 Contributing

When adding new experiments:

1. Follow the same pattern as existing scripts
2. Import from `docetl.reasoning_optimizer` modules
3. Save structured outputs to enable comparison
4. Update this README with new experiment documentation