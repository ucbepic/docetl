# AI SQL

Query your data with SQL where some functions are LLM-powered. AI
functions compile to DocETL's operators (so gleaning, cascades,
fold/merge reduce, blocking, and MOAR all still apply), while the plain
relational parts — `WHERE price > 10`, joins, `GROUP BY`, `ORDER BY`,
`LIMIT` — run on an embedded DuckDB engine.

```python
from docetl.aisql import run_sql

table = run_sql("""
    SELECT ai_summarize(transcript) AS summary
    FROM 'calls.json'
    WHERE duration > 300
""")
print(table.to_pylist())
```

`duration > 300` runs in DuckDB (reading the file with pushdown, so the
LLM never sees short calls); `ai_summarize` runs as a DocETL `map`.
`run_sql` returns a PyArrow table — call `.to_pylist()` for rows or
`.to_pandas()` for a DataFrame.

Install the optional dependencies:

```bash
pip install "docetl[aisql]"   # duckdb, sqlglot, pyarrow
```

Set your model the usual way (`docetl.default_model = "gpt-4o-mini"`, or a
`default_model` in your environment/config).

## The AI functions

| Function | Where it goes | Compiles to |
|---|---|---|
| `ai_summarize(col)` | SELECT | `map` |
| `ai_classify(col, 'instruction')` | SELECT | `map` |
| `ai_extract(col, 'instruction')` | SELECT | `map` |
| `ai(col, 'prompt')` | SELECT | `map` (generic) |
| `ai_filter(col, 'question')` | WHERE | `filter` |
| `ai_score(col, 'criteria') <cmp> k` | WHERE | `map` + relational filter |
| `ai_agg(col, 'instruction')` | SELECT with GROUP BY | `reduce` |
| `ai_match(l.col, r.col, 'question')` | JOIN … ON | `equijoin` |
| `ai_resolve(table, on := col, prompt := 'q')` | FROM | `resolve` |

SELECT-position AI functions need an alias (`... AS name`).

## Examples

**Filter, then transform — cheaply.** A relational predicate shrinks the
set in DuckDB before the LLM runs:

```sql
SELECT ai_summarize(body) AS summary
FROM 'tickets.json'
WHERE created_at > '2026-01-01' AND ai_filter(body, 'Is this a complaint?')
ORDER BY summary
LIMIT 20
```

The relational `created_at` filter runs first in DuckDB; `ai_filter` runs
on the survivors; `ai_summarize` runs on what's left; `ORDER BY`/`LIMIT`
apply to the final result.

**Score and threshold:**

```sql
SELECT id FROM 'reviews.json'
WHERE ai_score(text, 'Rate the positivity from 0 to 1') > 0.7
```

`ai_score` becomes a `map` that produces a number; the `> 0.7` runs in
DuckDB on that column.

**Aggregate a group:**

```sql
SELECT category, ai_agg(review, 'Summarize the common themes') AS themes
FROM 'reviews.json'
GROUP BY category
```

**Join on a fuzzy match:**

```sql
SELECT p.name, l.title
FROM 'products.json' p
JOIN 'listings.json' l ON ai_match(p.name, l.title, 'Same product?')
```

**Resolve duplicates:**

```sql
SELECT name FROM ai_resolve('customers.json', on := name,
                            prompt := 'Do these refer to the same person?')
```

## How it runs

A query is split into stages: DuckDB executes the relational parts
(scans with pushdown, plain predicates, joins, group-by, order/limit),
and DocETL runs the AI operators in between, with rows handed across the
boundary as Arrow. Relational filters are pushed ahead of LLM work so the
model processes as few rows as the query allows. See
[the design doc](design/ai-sql.md) for the architecture.

Because AI functions are real DocETL operators, the optimizer applies:
`semantic_pipelines(compile_sql(query))` (in `docetl.aisql`) exposes the
LLM stages as standard pipeline configs for MOAR.

## Not yet supported

These raise a clear error rather than misbehaving:

- `OR` / `NOT` around an AI predicate (only `AND` of predicates splits).
- An AI function on both sides of a comparison.
- AI functions inside `GROUP BY`, `ORDER BY`, or a join `ON` beyond a
  single `ai_match`.
- Multiple `ai_agg` in one query; a `WHERE` alongside an AI join.

Plain `ORDER BY` / `LIMIT` over columns and AI-output aliases are
supported.
