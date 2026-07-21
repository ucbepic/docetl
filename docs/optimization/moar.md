# MOAR Optimizer

The MOAR (Multi-Objective Agentic Rewrites) optimizer explores different ways to optimize your pipeline, finding solutions that balance accuracy and cost.

## What is MOAR?

When optimizing pipelines, you trade off cost and accuracy. MOAR explores many different pipeline configurations (like changing models, adding validation steps, combining operations, etc.) and evaluates each one to find the best trade-offs. It returns a frontier of plans that balance cost and accuracy, giving you multiple optimized options to choose from based on your budget and accuracy requirements.

!!! info "MOAR paper (VLDB 2026)"
    Lindsey Linxi Wei, Shreya Shankar, Sepanta Zeighami, Yeounoh Chung,
    Fatma Ozcan, and Aditya G. Parameswaran.
    “[Multi-Objective Agentic Rewrites for Unstructured Data Processing](https://arxiv.org/abs/2512.02289).”
    *Proceedings of the VLDB Endowment*, 2026.

## Quick Navigation

- **[Getting Started](moar/getting-started.md)** - Step-by-step guide to run your first MOAR optimization
- **[Configuration](moar/configuration.md)** - Complete reference for all configuration options
- **[Evaluation Functions](moar/evaluation.md)** - How to write and use evaluation functions
- **[Understanding Results](moar/results.md)** - What MOAR outputs and how to interpret it
- **[Examples](moar/examples.md)** - Complete working examples
- **[Troubleshooting](moar/troubleshooting.md)** - Common issues and solutions
- **[Extending MOAR](../developer-reference/moar-extensibility.md)** - How to implement and test a rewrite directive

## When to Use MOAR

!!! success "Good for"
    - Finding cost-accuracy trade-offs across different models
    - When you want multiple optimization options to choose from
    - Custom evaluation metrics specific to your use case
    - Exploring different pipeline configurations automatically

## Basic Workflow

1. **Create your pipeline** — Define your DocETL pipeline in Python or YAML
2. **Write an evaluation function** — Create a Python function to measure accuracy
3. **Run optimization** — Call `frame.optimize()` with your eval function:

    ```python
    optimized = frame.optimize(eval_fn=evaluate, metric_key="score")
    ```

    Or via CLI: `docetl build pipeline.yaml`

4. **Review results** — Run the optimized pipeline, or browse the cost-accuracy frontier:

    ```python
    rows = optimized.collect()           # Execute the optimized pipeline

    results = optimized.search_results
    best = results.best()              # Highest accuracy
    cheap = results.cheapest()         # Lowest cost
    print(results.to_df())             # All explored plans
    ```

## Extending and contributing to MOAR

MOAR separates rewrite directives from the search policy. A directive describes
one pipeline transformation, while the search decides where to try it and the
evaluation function measures the result. Developers can therefore add a new
optimization strategy without changing the MCTS implementation. See
[Extending MOAR with rewrite directives](../developer-reference/moar-extensibility.md)
for the implementation, registration, testing, and observability workflow.

Useful contribution areas include:

- **Novel rewrite directives.** Add a transformation with explicit
  applicability rules, a narrow instantiate schema, deterministic validation,
  and benchmarks on more than one workload.
- **Better directive testing.** Improve deterministic schema and pipeline tests,
  add replayable rewrite-agent responses, and measure selection rate, valid-plan
  rate, accuracy change, cost change, and variance across repeated runs.
- **A lightweight search policy.** Add a one-pass, greedy, best-first, or small
  beam search that reuses the directive registry, evaluator, candidate
  validation, and result format without running the full MCTS loop. Treat the
  policy as a comparable search backend rather than a separate optimizer, and
  make model-call and execution budgets explicit.
- **Search observability.** Record structured action traces that connect a
  selected directive to its generated plan, validation result, evaluation, and
  parent pipeline.

The lightweight policy is especially useful for small pipelines and development
smoke tests. MCTS remains useful when interactions among several rewrites and a
larger cost-accuracy frontier justify the additional search calls.

Ready to get started? Head to the [Getting Started guide](moar/getting-started.md).
