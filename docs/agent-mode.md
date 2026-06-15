# Agent Mode

Agent mode adds checks to DocETL operations. These checks raise errors
with clear messages so an AI agent (or a person) can fix the problem.

Some checks run **before** the LLM is called. Others run **after** the
LLM returns.

```python
import docetl

docetl.agent_mode = True
```

| Surface | How to enable |
|---------|---------------|
| **Python API** | `docetl.agent_mode = True` (global) or `agent_mode=True` on one operation |
| **YAML** | `agent_mode: true` at the top level or on one operation |
| **AI SQL** | `run_sql(query, agent_mode=True)` |

To skip checks on a single AI SQL query, add `/* nocheck */` anywhere in
the SQL string. For YAML or Python, remove `agent_mode` or set it to
`false` on that operation.

---

## Before the LLM runs

### EmptyInputError

There are zero rows. The LLM has nothing to work on.

**Happens when:** The input list is empty.

```python
run_sql("""
    SELECT ai_summarize(text) AS summary
    FROM 'tickets.json'
    WHERE created_at > '2099-01-01'
""", agent_mode=True)
# => EmptyInputError: Your relational predicates matched 0 rows
```

```python
docetl.agent_mode = True
docetl.from_list([]).map(
    prompt="Summarize: {{ input.text }}",
    output={"schema": {"summary": "string"}},
).collect()
# => EmptyInputError: Your relational predicates matched 0 rows
```

---

### TooManyRowsError

There are too many rows. The default limit is 100. This usually means
a `WHERE` clause is missing.

**Happens when:** The row count is above `max_ai_rows` for `map` or
`filter` operations. `reduce` and `resolve` are not checked (they
are expected to get many rows).

```python
run_sql("""
    SELECT ai_extract(body, 'key facts') AS facts
    FROM 'corpus.parquet'
""", agent_mode=True, max_ai_rows=50)
# => TooManyRowsError: Query sends 10,000 rows to the LLM
#    (threshold: 50). Add WHERE predicates to narrow the input.
```

---

### MissingColumnError

The prompt uses a column name that does not exist in the data.

**Happens when:** `{{ input.X }}` in the prompt does not match any
column in the input rows.

```python
run_sql("""
    SELECT ai_extract(body, 'key facts') AS facts
    FROM 'data.parquet'
""", agent_mode=True)
# => MissingColumnError: Prompt references columns that don't exist:
#    ['body']. Available columns: ['id', 'text', 'category'].
```

```yaml
operations:
  - name: extract_facts
    type: map
    agent_mode: true
    prompt: "Extract key facts from: {{ input.body }}"
    output:
      schema:
        facts: "list[string]"
```

---

### ChunkOverflowError

Some text values are too long for the LLM. The error says which columns
are too long, how many rows are over the limit, and the longest value.

**Happens when:** A text column in the prompt has at least one value
longer than `max_chunk_tokens` (default: 100,000).

```python
run_sql("""
    SELECT ai_summarize(full_text) AS summary
    FROM 'books.parquet'
""", agent_mode=True)
# => ChunkOverflowError: Text too long for AI functions
#    (limit: ~100,000 tokens). 5 rows matched.
#      Column 'full_text': longest value ~450,000 tokens,
#      3/5 rows exceed the limit.
#    Split text into chunks before AI functions.
```

---

### HighCardinalityError

The `GROUP BY` or `reduce_key` has almost all unique values. The LLM
would run once per row with one-row groups. This is slow and usually
wrong.

**Happens when:** For `reduce` or `resolve`, the number of unique key
values is 80% or more of the total row count.

```python
run_sql("""
    SELECT user_id, ai_agg(message, 'Summarize') AS summary
    FROM 'chat_logs.parquet'
    GROUP BY user_id
""", agent_mode=True)
# => HighCardinalityError: GROUP BY (user_id) has 9,500 distinct values
#    across 10,000 rows.
```

```python
docetl.agent_mode = True
docetl.read_json("chat_logs.json").reduce(
    reduce_key="user_id",
    prompt="Summarize: {% for m in inputs %}{{ m.message }}{% endfor %}",
    output={"schema": {"summary": "string"}},
).collect()
# => HighCardinalityError (same message)
```

---

## After the LLM runs

### AmbiguousSourceError

The search term appears in rows from multiple files or sections, but the
LLM only used some of them. The other files or sections might have had
better data.

For example: a prompt asks about "Judiciary" spending. The input has rows
from three different tables. The LLM only read one table and returned an
answer. But "Judiciary" also appears in the other two tables with
different numbers. This error tells the agent to look at those other
tables before deciding.

**Happens when:** After a `map` or `filter` runs, the check:

1. Finds quoted terms in the prompt (like `'Judiciary'`)
2. Looks at which input rows contain those terms
3. Groups those rows by their source file or section
4. Checks if the LLM produced output from all groups or just some

If there are multiple groups but the LLM only used some, the error fires.

**Needs a source column:** The input rows need a column that says where
each row came from. The check looks for columns named `source`,
`source_file`, `filename`, `file`, `path`, or `_source` (for files),
and `section`, `section_header`, `table_title`, `heading`, or `_section`
(for sections within a file). If none of these columns exist, this check
does nothing.

```python
# "Judiciary" is in rows from three tables, but the LLM only used one
# => AmbiguousSourceError: 'judiciary' appears in 3 distinct regions
#    but only 1 were used.
#    Regions used: ['Table 3.2: Outlays by Function'].
#    Also found in: ['Table FFO-3: Outlays by Agency',
#    'Table 4.1: Budget Authority'].
```

```python
data = docetl.from_list([
    {"source": "report_2024.txt", "text": "Judiciary spending: $100M"},
    {"source": "report_2023.txt", "text": "Judiciary spending: $95M"},
    {"source": "report_2024.txt", "text": "Defense spending: $800M"},
])
# If the LLM only reads report_2024.txt but "Judiciary" is also in
# report_2023.txt, AmbiguousSourceError fires.
```

---

## Turning it on for one operation

You can set `agent_mode` on just one operation instead of the whole
pipeline:

```python
results = (
    docetl.read_json("data.json")
    .map(
        prompt="Extract: {{ input.text }}",
        output={"schema": {"entity": "string"}},
        agent_mode=True,  # only this operation gets checks
    )
    .reduce(
        reduce_key="entity",
        prompt="Summarize: {% for item in inputs %}{{ item.text }}{% endfor %}",
        output={"schema": {"summary": "string"}},
    )
    .collect()
)
```

```yaml
operations:
  - name: extract
    type: map
    agent_mode: true
    prompt: "Extract: {{ input.text }}"
    output:
      schema:
        entity: string

  - name: summarize
    type: reduce
    reduce_key: entity
    prompt: "Summarize: {% for item in inputs %}{{ item.text }}{% endfor %}"
    output:
      schema:
        summary: string
```

Or for the whole pipeline:

```yaml
agent_mode: true
default_model: gpt-4o-mini

operations:
  - name: extract
    type: map
    # ...
```

---

## Catching errors

All errors are subclasses of `ValueError`. You can catch them one by one
or all at once:

```python
from docetl.checks import (
    EmptyInputError,
    TooManyRowsError,
    MissingColumnError,
    ChunkOverflowError,
    HighCardinalityError,
    AmbiguousSourceError,
)

try:
    results = run_sql(query, agent_mode=True)
except EmptyInputError:
    # add fewer filters, or check the data path
except TooManyRowsError:
    # add more WHERE conditions
except MissingColumnError as e:
    # fix the column name (e.available has the real names)
except ChunkOverflowError:
    # split text into smaller pieces first
except HighCardinalityError:
    # group by a less unique column
except AmbiguousSourceError as e:
    # check the other sources in e.all_sources
```
