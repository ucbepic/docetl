# AI SQL

Write SQL queries where some functions call an LLM. Regular SQL
(`WHERE`, `JOIN`, `GROUP BY`, `ORDER BY`, `LIMIT`) runs in DuckDB.
AI functions (`ai_summarize`, `ai_filter`, etc.) run as DocETL
operations, so you get gleaning, cascades, and optimization for free.

```python
from docetl.aisql import run_sql

table, cost = run_sql("""
    SELECT ai_summarize(transcript) AS summary
    FROM 'calls.parquet'
    WHERE duration > 300
""")
print(table.to_pylist())
```

`duration > 300` runs in DuckDB first. Only rows that pass get sent to
the LLM for summarization. `run_sql` returns a PyArrow table and the
total LLM cost. Call `.to_pylist()` for rows or `.to_pandas()` for a
DataFrame.

The `FROM` clause accepts any format DuckDB can read: parquet, CSV,
JSON, and others.

Install the optional dependencies:

```bash
pip install "docetl[aisql]"   # adds duckdb and sqlglot
```

Set your model the usual way (`docetl.default_model = "gpt-4o-mini"`, or a
`default_model` in your environment/config).

## The AI functions

| Function | Used in | Compiles to |
|---|---|---|
| `ai_summarize(col)` | SELECT | `map` |
| `ai_classify(col, 'instruction')` | SELECT | `map` |
| `ai_extract(col, 'instruction')` | SELECT | `extract` |
| `ai(col, 'prompt')` | SELECT | `map` (generic) |
| `ai_filter(col, 'question')` | WHERE | `filter` |
| `ai_score(col, 'criteria') <cmp> k` | WHERE | `map` + relational filter |
| `ai_agg(col, 'instruction')` | SELECT with GROUP BY | `reduce` |
| `ai_match(l.col, r.col, 'question')` | JOIN ... ON | `equijoin` |
| `ai_resolve(table, on := col, prompt := 'q')` | FROM | `resolve` |

AI functions in SELECT need an alias (`... AS name`).

### Output schemas

By default, each AI function produces one string column named after the
alias. So `ai_summarize(text) AS summary` outputs a column called
`summary` with type `string`.

To get multiple columns or non-string types, pass a schema as a second
argument:

```sql
SELECT ai(review, 'Rate the sentiment and explain why',
          'score:number, reason:string') AS analysis
FROM 'reviews.csv'
```

This produces two columns: `score` (number) and `reason` (string).

Supported types: `string`/`varchar`/`text`, `number`/`float`/`double`,
`integer`/`int`, `boolean`/`bool`, `list`/`array`. A bare key with no
type defaults to `string`.

## Examples

**Filter and transform:**

```sql
SELECT ai_summarize(body) AS summary
FROM 'tickets.parquet'
WHERE created_at > '2026-01-01' AND ai_filter(body, 'Is this a complaint?')
ORDER BY summary
LIMIT 20
```

You can mix regular filters and AI filters in the same `WHERE` clause.
DocETL automatically runs the regular filters first in DuckDB to reduce
the number of rows before calling the LLM. In this example,
`created_at > '2026-01-01'` runs first, then `ai_filter` runs on the
remaining rows, then `ai_summarize` runs on what passes, then
`ORDER BY`/`LIMIT` apply at the end.

**Score and threshold:**

```sql
SELECT id FROM 'reviews.csv'
WHERE ai_score(text, 'Rate the positivity from 0 to 1') > 0.7
```

`ai_score` asks the LLM to produce a number. The `> 0.7` comparison
then runs in DuckDB on that number.

**Aggregate a group:**

```sql
SELECT category, ai_agg(review, 'Summarize the common themes') AS themes
FROM 'reviews.parquet'
GROUP BY category
```

**Join on a fuzzy match:**

```sql
SELECT p.name, l.title
FROM 'products.csv' p
JOIN 'listings.csv' l ON ai_match(p.name, l.title, 'Same product?')
```

**Resolve duplicates:**

```sql
SELECT name FROM ai_resolve('customers.parquet', on := name,
                            prompt := 'Do these refer to the same person?')
```

## Agent mode

When an AI agent is writing queries, it can make mistakes: wrong column
names, no `WHERE` clause (sending too many rows to the LLM), text that's
too long, or picking the wrong data source when multiple sources match.

Pass `agent_mode=True` to catch these mistakes automatically. Some checks
run before the LLM call (like missing columns), and some run after (like
checking if there were other data sources the LLM should have looked at).

```python
table, cost = run_sql(query, agent_mode=True, max_ai_rows=200)
```

The errors have clear messages so the agent can fix the query and retry.
See [Agent Mode](agent-mode.md) for the full list of checks with
examples.

## How OR works

`OR` between AI and relational predicates is supported:

```sql
SELECT id FROM 'tickets.csv'
WHERE ai_filter(body, 'Is this a complaint?') OR priority = 1
```

The compiler runs the AI filter on all rows to produce a boolean
column, then evaluates the full `OR` expression in DuckDB. This means
every row goes through the LLM (the relational side can't narrow the
input when it's `OR`). Use `AND` when you want the relational filter
to run first and reduce LLM calls.
