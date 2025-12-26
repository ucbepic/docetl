# MOAR Optimizer

The MOAR (Multi-Objective Agentic Rewrites) optimizer explores different ways to optimize your pipeline, finding solutions that balance accuracy and cost.

## What is MOAR?

When optimizing pipelines, you trade off cost and accuracy. MOAR explores many different pipeline configurations (like changing models, adding validation steps, combining operations, etc.) and evaluates each one to find the best trade-offs. It returns a frontier of plans that balance cost and accuracy, giving you multiple optimized options to choose from based on your budget and accuracy requirements.

## Quick Navigation

- **[Getting Started](moar/getting-started.md)** - Step-by-step guide to run your first MOAR optimization
- **[Configuration](moar/configuration.md)** - Complete reference for all configuration options
- **[Evaluation Functions](moar/evaluation.md)** - How to write and use evaluation functions
- **[Understanding Results](moar/results.md)** - What MOAR outputs and how to interpret it
- **[Examples](moar/examples.md)** - Complete working examples
- **[Troubleshooting](moar/troubleshooting.md)** - Common issues and solutions

## When to Use MOAR

!!! success "Good for"
    - Finding cost-accuracy trade-offs across different models
    - When you want multiple optimization options to choose from
    - Custom evaluation metrics specific to your use case
    - Exploring different pipeline configurations automatically

## Basic Workflow

1. **Create your pipeline YAML** - Define your DocETL pipeline
2. **Write an evaluation function** - Create a Python function to measure accuracy
3. **Configure MOAR** - Set up `optimizer_config` in your YAML
4. **Run optimization** - Execute `docetl build pipeline.yaml --optimizer moar`
5. **Review results** - Choose from the cost-accuracy frontier

Ready to get started? Head to the [Getting Started guide](moar/getting-started.md).
