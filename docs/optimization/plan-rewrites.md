# Plan Rewrites

DocETL lifts every pipeline into a typed logical plan before running it and
applies a small set of **equivalence-preserving rewrites** — reorderings
that cut LLM cost without changing your output. Unlike [MOAR](moar.md),
which searches for accuracy/cost trade-offs offline, plan rewrites are free
wins applied automatically at the start of every run.

## What the rules do

**Selection pushdown** moves a filter below an LLM map that doesn't produce
anything the filter reads, so the map only processes rows the filter keeps:

```
map(1000 rows) → filter(keeps 200)        # before: 1000 LLM calls
filter(keeps 200) → map(200 rows)         # after:   200 LLM calls
```

This fires only when DocETL can *prove* the swap is safe: the filter's
predicate must not reference any field the map writes (including
`drop_keys`), the map must be one-output-per-row with no reordering, and
neither op's output may be shared by another step. The analysis is
fail-closed — if a prompt or code predicate reads fields in a way DocETL
can't statically enumerate (`{{ input.items() }}`, dynamic keys, multi-arg
callables), the rewrite simply doesn't fire.

**Limit pushdown** moves a positional head (`sample` with `method: first`)
below one-to-one ops, so upstream LLM calls run only on the rows that
survive the head. One caveat: if an LLM call exhausts its retries, the row
is silently dropped today, and a head that was pushed below such an op can
then select a slightly different row set. This only matters in runs that
are already losing rows to that failure mode; disable the rule (below) if
exact head semantics matter more to you than the cost saving.

## Seeing what fired

```python
frame.explain(optimized=True)
# -- applied selection_pushdown: pushed keep (filter) below summarize (map), ...
# keep [filter · selection · llm]
#   summarize [map · 1:1 · llm] +{summary: string}
#     scan(docs)
```

`frame.plan()` returns the typed plan for programmatic inspection. At run
time, every applied rewrite is logged as `Plan rewrite — ...`.

## Turning rewrites off (or down)

Rewrites are **on by default**. Control them per pipeline or globally:

=== "YAML"

    ```yaml
    plan_rewrites: false                      # off entirely
    # or select rules by name:
    plan_rewrites: ["selection_pushdown"]     # no limit pushdown
    ```

=== "Python"

    ```python
    import docetl
    docetl.plan_rewrites = False              # all in-process runs
    ```

Misspelled rule names raise an error rather than silently disabling
anything. The CLI runs in its own process, so for `docetl run` use the YAML
key.

!!! note "Checkpoints"

    Checkpoint hashes are computed over the *rewritten* pipeline. The first
    run after a rewrite starts firing (or after toggling `plan_rewrites`)
    re-executes instead of reusing old checkpoints — a one-time cost, never
    a correctness issue.

## MOAR integration

MOAR candidates are validated statically before any execution cost is paid
(malformed directive output is rejected immediately), and each saved
candidate is canonicalized with the same rewrite rules — honoring your
`plan_rewrites` setting. Set the `DOCETL_MOAR_PLAN_SUMMARY` environment
variable to include the typed plan in the rewrite agent's context.
