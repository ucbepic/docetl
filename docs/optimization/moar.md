# MOAR optimizer

MOAR searches for pipeline changes that improve accuracy or reduce cost. It
evaluates each candidate with your evaluation function and returns several
plans. Each returned plan represents a different balance between cost and
accuracy.

!!! info "MOAR paper, VLDB 2026"
    Lindsey Linxi Wei, Shreya Shankar, Sepanta Zeighami, Yeounoh Chung,
    Fatma Ozcan, and Aditya G. Parameswaran.
    ["Multi-Objective Agentic Rewrites for Unstructured Data Processing"](https://arxiv.org/abs/2512.02289).
    *Proceedings of the VLDB Endowment*, 2026.

## Guides

- **[Getting started](moar/getting-started.md).** Run your first MOAR search.
- **[Configuration](moar/configuration.md).** Set models, budgets, and other options.
- **[Evaluation functions](moar/evaluation.md).** Write the function that scores each candidate.
- **[Understanding results](moar/results.md).** Read the plans that MOAR returns.
- **[Examples](moar/examples.md).** See complete examples.
- **[Troubleshooting](moar/troubleshooting.md).** Fix common errors.
- **[Adding a rewrite directive](../developer-reference/moar-extensibility.md).** Add and test a new kind of pipeline change.

## When to use MOAR

!!! success "MOAR is useful when"
    - You want to compare cost and accuracy across models.
    - You want several optimized plans with different costs.
    - You have an evaluation function for your pipeline.
    - You want MOAR to test several kinds of pipeline changes.

## Basic workflow

1. **Create a pipeline.** Define the pipeline in Python or YAML.
2. **Write an evaluation function.** Create a Python function that measures accuracy.
3. **Run MOAR.** Call `frame.optimize()` with the evaluation function.

    ```python
    optimized = frame.optimize(eval_fn=evaluate, metric_key="score")
    ```

    You can also run MOAR from the command line.

    ```bash
    docetl build pipeline.yaml
    ```

4. **Review the results.** Run one optimized pipeline or compare all plans that
   MOAR returned.

    ```python
    rows = optimized.collect()         # Run the optimized pipeline

    results = optimized.search_results
    best = results.best()              # Highest accuracy
    cheap = results.cheapest()         # Lowest cost
    print(results.to_df())             # All plans that MOAR tested
    ```

## Extending and contributing to MOAR

In MOAR, rewrite directives are separate from the search method. When you add a
directive, you define one way to change a pipeline. The search method chooses
where to try the directive, and your evaluation function scores the changed
pipeline. You can add a directive without changing the code for Monte Carlo
tree search (MCTS).

See [Adding a MOAR rewrite directive](../developer-reference/moar-extensibility.md)
for instructions on implementation, registration, testing, and logs.

Possible contributions include the following work:

- **New rewrite directives.** Add a new pipeline change with clear rules for
  when MOAR should use it. Define the agent output and reject invalid output.
  Test the directive on more than one workload.
- **Tests for directives.** Add tests for schemas and complete pipelines. Store
  model responses so tests can replay them without a model call. Benchmarks can
  record how often MOAR selects the directive and builds a valid pipeline. They
  can also record changes in cost and accuracy.
- **Search methods that use fewer model calls.** For example, add a search
  method that tries each chosen directive once and keeps only the best
  candidates. Reuse the existing directive list, evaluator, validation code,
  and result format. Set clear limits for model calls and pipeline runs.
- **Logs for each search.** Save the selected directive and the pipeline that
  MOAR built. Also save the validation result and the candidate's score.

A search method with fewer model calls can help with small pipelines and quick
development tests. MCTS is more useful when you want to combine several
rewrites or inspect more balances between cost and accuracy.

See the [Getting started guide](moar/getting-started.md) to run an optimization.
