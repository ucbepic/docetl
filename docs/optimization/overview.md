# Optimization

LLM pipelines have two competing objectives: **accuracy** (is the output
right?) and **cost** (how much do the model calls cost?). DocETL has two
optimizers for improving them:

- **[MOAR](moar.md)** optimizes accuracy and cost together, for the whole
  pipeline. It tries rewritten variants of your pipeline on a sample of your
  data, scores them with an evaluation function you write, and returns the
  best ones. You run it once, before your real runs.
- **[Model cascades](cascades.md)** reduce the cost of a single `filter`,
  `resolve`, or `equijoin`: a cheap model answers the easy items and the
  expensive model only sees the hard ones, with a statistical guarantee on
  quality. They run as part of normal execution — there is no separate step.

**Recommendation:** use MOAR when you can write an evaluation function and
can wait — it runs the pipeline many times on data samples, so a search takes
tens of minutes. Use a cascade when one expensive `filter`, `resolve`, or
`equijoin` dominates your cost; it adds no extra runs. You can use both in
the same pipeline. We plan to merge the two into one optimizer.

## Running MOAR

=== "Python"

    ```python
    optimized = frame.optimize(eval_fn=my_eval_function, metric_key="score")
    df = optimized.collect()                  # run the optimized pipeline
    best = optimized.search_results.best()    # inspect the frontier
    ```

=== "YAML / CLI"

    ```bash
    docetl build pipeline.yaml
    ```

MOAR's rewrites include decomposing operations (for example, splitting a map
over long documents into split → map-per-chunk → reduce), adding gleaning or
resolve steps, and changing models. The [MOAR Optimizer Guide](moar.md)
covers how the search works, writing evaluation functions, and reading the
results.

!!! warning "V1 Optimizer Deprecated"
    The V1 optimizer is deprecated; use MOAR. Existing V1-optimized pipelines
    continue to work (via `method="v1"` on the deprecated `docetl.api.Pipeline`
    class), but new optimizations should use MOAR.
