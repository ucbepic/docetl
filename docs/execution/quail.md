# Quail Engine

DocETL can run filter and map operations on [Quail](https://github.com/fsdatalab/quail) Server, a SQL engine with AI functions that runs local models, such as Qwen3 4B, on your GPUs. The Quail engine compiles each run of consecutive filter and map operations into one Quail SQL query. Quail then reads each document once for all AI functions in the query, and runs a later function only on the rows an earlier filter keeps.

## Setup

Start `quail-server` on a GPU machine, as the [Quail Server guide](https://fsdatalab.github.io/quail/docs/user-guide/server) describes:

```bash
quail-server --model qwen3-4b-fp8 --device h100-sxm --listen 0.0.0.0:50051
```

Then point the pipeline at the server:

```yaml
engine:
  type: quail
  endpoint: grpc://quail-host:50051
  fallback: litellm   # or strict
```

DocETL talks to the server over Arrow Flight with `pyarrow`, so DocETL needs no Quail package. Set `QUAIL_SERVER_TOKEN` when the server uses a token.

## What runs on Quail

| DocETL | Quail SQL |
| --- | --- |
| `filter` with one `bool` output | `WHERE AI.IF(PROMPT(...))` |
| `map`, one AI function per output item | `bool`: `AI.IF`; `enum[...]`: `AI.CLASSIFY`; `int`, `float`: `AI.MAP` with `TRY_CAST`; `string`: `AI.MAP`; `list[...]`, objects: `AI.MAP` with a JSON instruction |
| everything else | LiteLLM |

A filter and a map become one query:

```sql
WITH c0 AS (SELECT * FROM `docs`),
c1 AS (SELECT ... FROM c0 WHERE AI.IF(PROMPT('Does the review discuss the ending? {0}', c0.`body`))),
c2 AS (SELECT ..., AI.CLASSIFY(PROMPT('Label the review: {0}', c1.`body`), ['praise', 'complaint']) AS `kind` FROM c1)
SELECT ... FROM c2
```

A later operation can read an earlier operation's output, such as `{{ input.summary }}`, and stays in the same query.

## Prompts

Quail builds every model prompt as a shared preamble, the document, the question, and an answer cue, so the question must stay the same across rows. A Quail operation therefore accepts only plain field references, such as `{{ input.body }}`, with fixed text around them. Any other Jinja, such as filters, loops, conditionals, or macros, has no Quail form; braces in the fixed text are refused too.

## Fallback

An operation runs on LiteLLM when it has no Quail form: another operation type, gleaning, validation, agents, tools, PDFs, `batch_prompt`, calibration, a retriever, a cascade, a filter `limit`, or a prompt outside the Quail form. With `fallback: litellm` (the default), DocETL logs one line and runs the operation on LiteLLM. With `fallback: strict`, the pipeline fails before it runs. An operation can set `engine: litellm` to opt out of a pipeline-level Quail engine.

## Notes

- The Quail engine ignores an operation's `model`: Quail Server runs the model it started with.
- `max_tokens` for `AI.MAP` comes from the operation's `litellm_completion_kwargs.max_tokens`, or Quail's default of 256.
- The query id is a hash of the SQL and the uploaded data. Quail Server runs one id once and keeps its rows, so a rerun reads the saved rows instead of using GPU time.
- Quail operations report a cost of $0.00, because the GPU runs outside DocETL.
