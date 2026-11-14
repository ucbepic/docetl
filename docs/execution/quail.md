# Quail Engine

DocETL can run filter, map, and reduce operations on [Quail](https://github.com/fsdatalab/quail) Server, a SQL engine with AI functions that runs local models, such as Qwen3 4B, on your GPUs. The Quail engine compiles each run of consecutive filter, map, and reduce operations into one Quail SQL query. Quail then reads each document once for all AI functions in the query, and runs a later function only on the rows an earlier filter keeps.

## Setup

Install the engine's one dependency, `pyarrow`:

```bash
pip install "docetl[quail]"
```

Start `quail-server` on a GPU machine, as the [Quail Server guide](https://fsdatalab.github.io/quail/docs/user-guide/server) describes:

```bash
quail-server --model qwen3-4b-fp8 --device h100-sxm --listen 0.0.0.0:50051
```

Then point the pipeline at the server:

```yaml
engine:
  type: quail
  endpoint: grpc://quail-host:50051   # grpc+tls://host:443 for a TLS endpoint
  fallback: litellm                   # or strict
  max_group_tokens: 24000             # reduce groups larger than this run on LiteLLM
```

DocETL talks to the server over Arrow Flight with `pyarrow`, so DocETL needs no Quail package. Set `QUAIL_SERVER_TOKEN` when the server uses a token, and `QUAIL_TLS_ROOT_CERTS` (or `SSL_CERT_FILE`) for a TLS endpoint behind a proxy that signs its own certificates.

## What runs on Quail

| DocETL | Quail SQL |
| --- | --- |
| `filter` with one `bool` output | `WHERE AI.IF(PROMPT(...))` |
| `map`, one AI function per output item | `bool`: `AI.IF`; `enum[...]`: `AI.CLASSIFY`; `int`, `float`: `AI.MAP` with `TRY_CAST`; `string`: `AI.MAP`; `list[...]`, objects: `AI.MAP` with a JSON instruction |
| `reduce` | `GROUP BY reduce_key`, with the same AI functions over `STRING_AGG` of the group's items, in row order |
| `equijoin`, `resolve`, everything else | LiteLLM |

A filter and a map become one query:

```sql
WITH c0 AS (SELECT * FROM `docs`),
c1 AS (SELECT ... FROM c0 WHERE AI.IF(PROMPT('Does the review discuss the ending? {0}', c0.`body`))),
c2 AS (SELECT ..., AI.CLASSIFY(PROMPT('Label the review: {0}', c1.`body`), ['praise', 'complaint']) AS `kind` FROM c1)
SELECT ... FROM c2
```

A later operation can read an earlier operation's output, such as `{{ input.summary }}` or `{{ item.summary }}` in a reduce, and stays in the same query.

## Prompts

Quail builds every model prompt as a shared preamble, the document, the question, and an answer cue, so the question must stay the same across rows. A filter or map prompt in Quail form has only plain field references, such as `{{ input.body }}`, with fixed text around them:

```text
Jinja:  Read the review: {{ input.body }}. Does it discuss the ending?
Quail:  AI.IF(PROMPT('Read the review: {0}. Does it discuss the ending?', body))
```

A reduce prompt in Quail form has fixed text around one loop over plain item fields:

```text
Summarize the reviews.{% for r in inputs %}
- {{ r.body }}{% endfor %}
```

Any other Jinja, such as filters, loops, conditionals, or macros, has no Quail form; braces in the fixed text have none either. DocETL never rewrites a prompt at run time, so the model reads the prompt you wrote.

### Rewrite helper

`docetl quail-rewrite` asks an LLM to rewrite each prompt that has no Quail form, checks the result with the engine's own parser, and lets you accept, edit, or skip each rewrite:

```bash
docetl quail-rewrite pipeline.yaml --model gpt-4.1 -o pipeline.quail.yaml
```

## Fallback

An operation runs on LiteLLM when it has no Quail form: another operation type, gleaning, validation, agents, tools, PDFs, `batch_prompt`, calibration, a retriever, a cascade, a filter `limit`, a reduce with `fold_prompt`, `merge_prompt`, `value_sampling`, or `pass_through`, or a prompt outside the Quail form. With `fallback: litellm` (the default), DocETL logs one line and runs the operation on LiteLLM. With `fallback: strict`, the pipeline fails before it runs. An operation can set `engine: litellm` to opt out of a pipeline-level Quail engine.

A reduce group larger than `max_group_tokens` (about 4 characters per token) does not fit in the model's context. DocETL checks group sizes before the query when the reduce reads uploaded fields; with `fallback: litellm` it runs that run of operations on LiteLLM, where reduce can fold and merge, and with `fallback: strict` it fails.

## Caching

- **Quail Server's saved rows.** The query id is a hash of the SQL and the uploaded data. Quail Server runs one id once and keeps its rows in its `--data` folder, so a rerun reads the saved rows instead of using GPU time. `bypass_cache: true`, on the pipeline or on any operation in a run, sends a fresh id, so the server runs the query again.
- **Intermediate checkpoints.** With `intermediate_dir` set, a fused run saves its rows under its last operation's name. The runner's operation hashes chain through a step, so the checkpoint is reused only while every operation in the run, and every operation before it, is unchanged.
- **LLM call cache.** Quail runs make no LiteLLM calls, so DocETL's LLM cache does not hold them, and `docetl clear-cache` does not clear Quail Server's saved rows; delete the server's `--data` folder for that.

## Notes

- The Quail engine ignores an operation's `model`: Quail Server runs the model it started with.
- `max_tokens` for `AI.MAP` comes from the operation's `litellm_completion_kwargs.max_tokens`, or Quail's default of 256.
- At startup, DocETL asks the server to plan each compiled query (`explain`) against an empty table, so a query Quail cannot plan fails before any GPU time.
- Quail operations report a cost of $0.00, because the GPU runs outside DocETL.
- `tests/quail_e2e/modal_e2e.py` runs the engine end to end against Qwen3 4B on a Modal H100.
