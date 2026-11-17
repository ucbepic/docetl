# Quail SQL Operation

The Quail SQL operation runs one SQL query, written by you, over the input rows on [Quail Server](../execution/quail.md). Use it when the pipeline's models are `quail/*` models and no single DocETL operation expresses the logic: for example, an `AI_EXTRACT` that runs only where a cheap `AI.IF` says the answer exists.

## Example

```yaml
- name: governing_law
  type: quail_sql
  model: quail/qwen3.8-27b-fp8
  sql: |
    SELECT i._docetl_id,
           CASE WHEN AI.IF(PROMPT('{0} Does the contract say which law governs it?', i.document))
                THEN get_field(AI_EXTRACT(i.document, named_struct('a', 'Which sentence states the governing law?')), 'a')
           END AS governing_law
    FROM input i
  output:
    schema:
      governing_law: string
```

## How it works

DocETL uploads the rows as a table named `input`, with each field as text and a `_docetl_id` column. The query must select `_docetl_id` and one column per output schema key. DocETL maps each returned row back to its input row and adds the output columns; a row the query leaves out is dropped.

The query may use any of Quail's AI functions in any SQL expression:

| Function | Returns |
| --- | --- |
| `AI.IF(PROMPT('text {0}', i.field))` | a boolean |
| `AI.SCORE(PROMPT(...))` | a number from 0 to 1 |
| `AI.CLASSIFY(PROMPT(...), ['a', 'b'])` | one of the labels |
| `get_field(AI_EXTRACT(i.field, named_struct('k', 'question')), 'k')` | text copied from the field, or NULL when the field has no answer |
| `AI.MAP(i.field, 'instruction', max_tokens)` | text the model writes |

`PROMPT` text and `AI.MAP` instructions may not contain braces, other than the `{0}`, `{1}`, ... placeholders. Everything else is DataFusion SQL in the BigQuery dialect: `CASE WHEN`, CTEs, window functions, `GROUP BY`, and string functions. AI functions over the same column share one read of the document.

## Parameters

| Parameter | Description | Default |
| --- | --- | --- |
| `model` | The `quail/<model>` to run on; `quail_models` names its server | Required |
| `sql` | The query over the table `input` | Required |
| `output.schema` | The columns the query returns; lists and objects come back as JSON text | Required |
| `input_fields` | The fields to upload, and to plan the query with at startup | The fields the query names |

With `input_fields` set, DocETL asks Quail Server to plan the query against an empty table at startup, so a query Quail cannot plan fails before any GPU time.
