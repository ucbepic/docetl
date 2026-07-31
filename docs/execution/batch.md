# Deferred Batch Execution

DocETL normally sends each LLM request immediately. For large, non-interactive
map and filter operations, you can instead materialize all independent requests
as OpenAI-compatible JSONL and run them with either a hosted provider Batch API
or a short-lived local vLLM job.

This is different from `batch_prompt`:

| Feature | What is batched | LLM requests |
| --- | --- | --- |
| `batch_prompt` | Several input rows inside one prompt | One request per group |
| `execution.mode: batch` | Independent request envelopes | One request per row, submitted as one job |

The second form preserves row-level prompts and output parsing while moving the
execution schedule out of the per-row thread pool.

## Hosted provider batches through LiteLLM

DocETL uses LiteLLM's Files and Batches abstraction for the hosted job
lifecycle. LiteLLM currently documents Batch support for OpenAI, Azure OpenAI,
Vertex AI, Bedrock, and hosted vLLM. This is a smaller set than the providers
supported by ordinary LiteLLM completions.

=== "YAML"

    ```yaml
    - name: classify_documents
      type: map
      model: openai/gpt-4o-mini
      prompt: Classify this document: {{ input.text }}
      output:
        schema:
          category: string
      execution:
        mode: batch
        backend: litellm
        provider: openai # optional when it can be inferred from model
        completion_window: 24h
        poll_interval_seconds: 30
    ```

=== "Python"

    ```python
    results = (
        docetl.read_json("documents.json")
        .map(
            prompt="Classify this document: {{ input.text }}",
            output={"schema": {"category": "string"}},
            model="openai/gpt-4o-mini",
            execution={
                "mode": "batch",
                "backend": "litellm",
                "provider": "openai",
                "completion_window": "24h",
                "poll_interval_seconds": 30,
            },
        )
        .collect()
    )
    ```

DocETL infers the provider adapter from a normal LiteLLM model string, so for
supported providers changing `openai/gpt-4o-mini` to a model such as
`vertex_ai/gemini-...` is enough. Set `provider` explicitly when using a model
alias or custom routing. If you use a LiteLLM model alias for multi-account
routing, set `routing_model` in the execution block as well.

OpenAI advertises 50% lower token prices and a completion window of up to 24
hours for its Batch API. Other providers have their own pricing and lifecycle
rules. DocETL therefore applies the OpenAI 0.5 cost multiplier only when
`provider: openai`; set `cost_multiplier` explicitly for other providers if you
want DocETL's estimated cost to reflect their batch pricing.

Provider batches are split at 50,000 requests or 200 MB by default, matching
OpenAI's per-batch limits. Set `max_batch_requests` or `max_batch_bytes` to a
smaller provider-specific limit when needed.

## Offline batches with vLLM

The vLLM backend writes the same JSONL requests and invokes `vllm run-batch` as
a subprocess. This is useful for an ephemeral GPU worker: start the worker,
load the model once, process the materialized dataset, write the results, and
shut the worker down.

```yaml
- name: classify_documents
  type: map
  model: Qwen/Qwen3-8B
  prompt: Classify this document: {{ input.text }}
  output:
    schema:
      category: string
  execution:
    mode: batch
    backend: vllm
    model: Qwen/Qwen3-8B
    request_defaults:
      max_tokens: 512
      min_tokens: 1
    engine_args:
      tensor_parallel_size: 2
      gpu_memory_utilization: 0.9
      max_model_len: 4096
      max_num_seqs: 256
      enforce_eager: false
```

`request_defaults` become fields in each request body. `engine_args` become
vLLM CLI flags by converting underscores to hyphens; for example,
`tensor_parallel_size` becomes `--tensor-parallel-size`. Use vLLM's own option
names (`max_model_len`, not `max_model_length`). A false boolean is omitted,
while a true boolean becomes a flag.

An always-on vLLM or SGLang OpenAI-compatible server can already be used for
ordinary DocETL calls through LiteLLM. That provides continuous batching inside
the server, but it is not a durable offline job and does not start or stop GPU
capacity. Use the vLLM backend above when the job lifecycle is the objective.

## Durability and ordering

DocETL writes batch artifacts under `.docetl/batches/<job-hash>/` by default:

- `input.jsonl` contains the materialized logical requests.
- `manifest.json` records a hosted provider batch ID.
- `output.jsonl` and `errors.jsonl` contain downloaded results.

If the process stops after submission, rerunning the same operation resumes the
provider job from the manifest instead of creating a duplicate. Completed
outputs are reused locally. Provider output order is not trusted; DocETL joins
responses to inputs by `custom_id` and restores input order.

Use `work_dir` to put these artifacts on persistent storage when the process or
node filesystem is ephemeral.

## Current compatibility

The first batch execution slice supports independent `map` calls and the normal
non-cascade `filter` path. It supports DocETL's tool-call and structured-output
modes, caching, retrieval context, observability, and `skip_on_error`.

Features that create dependent follow-up calls are rejected rather than
silently changing semantics: agents, gleaning, validation retries, calibration,
and cascades. `batch_prompt` and PDF inputs are also not currently combined with
deferred execution. A filter `limit` is rejected because DocETL cannot know
which rows pass until the full batch has run.
