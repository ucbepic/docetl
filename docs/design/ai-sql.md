# AI SQL for DocETL

Status: v1 implemented on branch `ai-sql` (`docetl/aisql/`, opt-in extra
`docetl[aisql]`); SELECT / WHERE / GROUP BY / JOIN / `ai_resolve` compile
and run, tested LLM-free. Remaining: OR/NOT around AI predicates,
multiple `ai_agg` per query, the user-facing entry surface, and live-LLM
end-to-end tests (see Milestones / Open questions).
Foundation: the plan IR in `docetl/plan/` (PR #497) is the compile target.

## Goal

SQL where some functions are LLM-backed. AI functions compile to DocETL
operators (not engine scalar UDFs); plain relational work goes to DuckDB.

```sql
SELECT ai_summarize(transcript) AS summary
FROM calls
WHERE duration > 300
```

`duration > 300` → DuckDB (file read + pushdown). `ai_summarize` → `map`.

## Decisions

- AI functions lower to operators, not engine UDFs — so cascades,
  gleaning, fold/merge, and MOAR still apply.
- Hybrid execution, DocETL IR conducts, DuckDB is a delegated subroutine
  for relational fragments. Not engine-as-conductor (DocETL ops as
  DataFusion nodes): that's a Rust rewrite that hands plan optimization to
  an engine that can't see inside LLM ops.
- DuckDB executes relational fragments; sqlglot parses and splits.
  DataFusion only wins under engine-as-conductor — see end.

## SQL → operators

| SQL | operator |
|---|---|
| `SELECT ai_*(col, prompt) AS x` | `map` / `parallel_map` |
| `WHERE ai_filter(col, prompt)` | `filter` (cascade applies) |
| `GROUP BY k` + `ai_agg(col, prompt)` | `reduce` (`reduce_key = k`) |
| `JOIN ON ai_match(a, b, prompt)` | `equijoin` |
| `ORDER BY ai_score(col, prompt) LIMIT k` | `rank` / `topk` |
| dedup (open question) | `resolve` |

Surface: named functions (`ai_filter`, `ai_classify`, `ai_extract`,
`ai_summarize`, `ai_agg`, `ai_match`, `ai_score`) plus a generic
`ai(col, prompt, output := '...')`. Output types come from the operator's
`output.schema`. Pure-relational subtrees pass to DuckDB verbatim.

## Splitting

Tag each plan node relational vs LLM (the `is_llm` trait), reshape into
alternating DuckDB/DocETL stages, order by the IR's existing pushdown
rules. The non-trivial cases:

- `AND`: `σ_{A∧B} = σ_B(σ_A)`. Relational first in DuckDB, LLM on the
  survivors.
- `OR`: `σ_{A∨B} = σ_A ∪ σ_B(σ_¬A)`. LLM runs on everything `A` didn't
  keep — warn when that's most of the table.
- Mixed leaf `ai_score(text) > 0.8`: `ai_score` → `map` (produces a
  column), `> 0.8` → DuckDB filter on it.

## Handoff

Zero-copy Arrow both directions; DuckDB at the leaves and again above AI
output if needed.

```python
rows = duckdb.sql(
    "SELECT id, transcript FROM 'calls.parquet' WHERE duration > 300"
).arrow()

summarized = (
    docetl.from_arrow(rows)
          .map(prompt="Summarize: {{ input.transcript }}",
               output={"schema": {"summary": "string"}})
          .to_arrow()
)

duckdb.register("summarized", summarized)
final = duckdb.sql(
    "SELECT summary, COUNT(*) FROM summarized GROUP BY summary"
).arrow()
```

## Components

Reused: plan IR, pushdown rules, runner, operators.
New: sqlglot frontend (parse + classify + split); DuckDB delegate (IR
subtree → SQL → Arrow, register back); Frame `from_arrow`/`to_arrow`
adapters; orchestrator that walks the partitioned plan.

## Open problems

- Predicate splitting: `AND`/`OR`/`NOT` + mixed leaves. Fall back to
  whole-predicate-in-DocETL when a clean split isn't obvious.
- Crossing cost: minimize boundary crossings; let DuckDB own the scan.
- Types: cast AI output back to relational via `output.schema` (plan
  schema propagation already computes these).
- `resolve` has no SQL keyword — table function `ai_resolve(...)` or
  `GROUP BY ai_cluster(...)`? Decide before milestone 5.
- SQL strings vs a builder API — UX choice, same machinery.

## Milestones

1. ✅ Frame `from_arrow` / `to_arrow`.
2. ✅ DuckDB delegate (`DuckDBEngine`) — pure-relational queries end to end.
3. ✅ Straight-line frontend — `SELECT ai_*(...) FROM src [WHERE relational]`.
4. ✅ Splitter — `AND` of relational / `ai_filter` / `ai_score(...) <cmp> k`
   (cost-correct staging). Deferred: `OR`/`NOT` around an AI predicate.
5. ✅ Aggregates + joins + resolve — `GROUP BY` + `ai_agg` → `reduce`,
   `JOIN ON ai_match` → `equijoin`, `ai_resolve(...)` → `resolve`.
   Deferred: multiple `ai_agg` per query, `WHERE`/extra-AI with a join.
6. ✅ Plan-IR bridge (`semantic_pipelines`) — compiled ops lift into the
   IR and the rewrite engine runs over them. End-to-end MOAR *tuning* of
   a whole query goes through `run_sql` + an eval function.

Tested LLM-free throughout (compile structure asserted; execution via
relational + code ops). What is *not* yet covered: live-LLM end-to-end
runs of the AI operators, and a top-level user entry point (`run_sql`
returns Arrow today — see open questions).

## DuckDB vs DataFusion

Since DocETL conducts, the engine only runs relational SQL and returns
Arrow; DuckDB is the mature fit. DataFusion's edge is Rust-native
extension nodes/rules, useful only if a query engine conducts and DocETL
ops become engine nodes. Revisit then.
