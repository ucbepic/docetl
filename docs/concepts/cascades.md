# Model Cascades with BARGAIN

Several DocETL operators issue an LLM call whose output is a single
**binary** value — keep/drop or match/no-match:

| Operator | Items | Binary output |
|---|---|---|
| `filter` | records | boolean `keep` |
| `resolve` | candidate pairs (after blocking) | `is_match` |
| `equijoin` | candidate pairs across two datasets | `is_match` |

For these, a cheap "proxy" model is usually right, and only the hard cases
need an expensive "oracle". A **model cascade** runs the proxy on everything,
learns a confidence threshold on a small oracle-labeled sample, trusts the
proxy above the threshold, and escalates the rest to the oracle — while
preserving a **statistical guarantee** that holds with probability `1 - delta`
for finite samples.

This is the approach of **BARGAIN** ([paper](https://arxiv.org/abs/2509.02896),
[code](https://github.com/ucbepic/BARGAIN)):

> Sepanta Zeighami, Shreya Shankar, Aditya Parameswaran.
> "Cut Costs, Not Accuracy: LLM-Powered Data Processing with Guarantees."
> *SIGMOD 2026.*

DocETL depends on the BARGAIN library directly for the statistical core
(threshold learning and guarantee certification) and wraps it with thin
adapters for each operator's proxy/oracle calls.

The result: far fewer expensive calls, with a quality guarantee you choose.

## Enabling a cascade

Add an opt-in `cascade:` block to a supported operator. The operator's existing
`model` (or `comparison_model`) is the **oracle**; you supply a cheaper
`proxy_model`.

```yaml
- name: is_relevant
  type: filter
  model: gpt-4o                  # oracle (unchanged)
  output:
    schema:
      keep: "bool"
  cascade:
    proxy_model: gpt-4o-mini     # the cheap model
    guarantee: recall            # default for filter
    target: 0.95                 # keep >= 95% of truly-relevant docs
    delta: 0.05                  # guarantee holds w.p. 1 - delta
    label_budget: 300            # max oracle calls spent learning the threshold
```

## Parameters

| Parameter | Type | Description | Default |
|---|---|---|---|
| `proxy_model` | string | The cheap model to use for the proxy pass (required). A chat model or an embedding model — see below | — |
| `guarantee` | string | Statistical guarantee to enforce (see [Guarantees](#guarantees)) | operator-specific (see below) |
| `target` | float | Target value for the guarantee metric, in `(0, 1)` (required) | — |
| `delta` | float | Failure probability; the guarantee holds with probability `1 - delta` | `0.05` |
| `label_budget` | int | Maximum oracle calls spent learning the confidence threshold | `400` |

The raw logprob threshold is deliberately **not** exposed — it is learned from
the data, and exposing it would defeat the guarantee. You stay in metric-space.

### Embedding models as the proxy

`proxy_model` can also name an embedding model (e.g.
`text-embedding-3-small`) — detected automatically from litellm's model
registry. Instead of one cheap LLM call per item, the cascade embeds every
item (batched), oracle-labels a training slice out of `label_budget`, fits a
small logistic-regression head on those embeddings, and uses the head's
probabilities as the proxy scores. The threshold search then runs as usual on
the *remaining* budget, with disjoint labels so the statistical bounds stay
valid. Training rows keep their oracle answers in the output, so no label is
wasted.

Embeddings cost orders of magnitude less than LLM calls and batch into a
handful of requests, so this is the cheapest proxy for high-volume,
semantically separable predicates (topical filters, near-duplicate checks).
It is weaker on reasoning-shaped predicates — there the head's scores won't
separate, the threshold search can't certify much, and items simply escalate
to the oracle (the same graceful failure as a weak LLM proxy). If the
training slice comes back all one class, the head can't be fit and everything
escalates.

Because part of the budget is spent fitting the head, give embedding proxies
roughly **2× the `label_budget`** you'd give an LLM proxy (≥ 100 recommended
for precision/recall guarantees).

## Guarantees

Pick the guarantee that matches the operator's intent:

| Guarantee | What it means | Best for | BARGAIN procedure |
|---|---|---|---|
| `accuracy` | Output matches the oracle on ≥ `target` fraction of items | Any binary operator | BARGAIN_A |
| `precision` | Of items returned positive, ≥ `target` are truly positive | `resolve` / `equijoin` (don't over-merge) | BARGAIN_P |
| `recall` | Of truly-positive items, ≥ `target` are returned | `filter` (don't drop relevant docs) | BARGAIN_R |

**Default guarantee per operator** (applied when `guarantee` is omitted):

| Operator | Default guarantee |
|---|---|
| `filter` | `recall` |
| `resolve` | `precision` |
| `equijoin` | `precision` |

### Small-sample behavior

If the oracle sample is too small to certify the `target` at the chosen `delta`,
the engine errs toward the guarantee — escalating more items to the oracle (or
keeping more items) — rather than silently violating it. That costs more oracle
calls, so give `label_budget` enough room for a meaningful sample on small
datasets.

## Complete example — run it end to end

A cascade is just a `cascade:` block on an operator in your normal pipeline YAML;
you run it the same way you run any DocETL pipeline. Here is a full, minimal
pipeline that filters documents for relevance with a cascade:

```yaml
# pipeline.yaml
datasets:
  documents:
    type: file
    path: documents.json          # a JSON list of objects, each with a "text" field

default_model: gpt-4o             # the oracle (high-quality) model

operations:
  - name: is_relevant
    type: filter
    prompt: |
      Is the following document about climate policy? Answer true or false.
      {{ input.text }}
    output:
      schema:
        keep: "bool"
    cascade:
      proxy_model: gpt-4o-mini    # the cheap model
      guarantee: recall           # don't drop relevant docs (filter's default)
      target: 0.95                # keep >= 95% of the truly-relevant docs
      delta: 0.05                 # guarantee holds with probability 0.95
      label_budget: 300           # at most 300 oracle calls to learn the threshold

pipeline:
  steps:
    - name: relevance
      input: documents
      operations:
        - is_relevant
  output:
    type: file
    path: relevant_documents.json
```

Run it from the command line:

```bash
docetl run pipeline.yaml
```

While the filter runs, the cascade prints a summary of what it did:

```text
Cascade filter 'is_relevant'
           proxy     gpt-4o-mini · 1000 scored · $0.0200
           oracle    gpt-4o · 137 sampled for calibration (budget 300) · $0.4000
           guarantee recall ≥ 95%  δ=0.05
           threshold 0.847 proxy confidence
           result    863 proxy-accepted + 137 calibration samples → 1000 items
           total cost $0.4200
```

How to read this: all 1000 documents were scored by the cheap proxy. The
oracle was called on 137 of them to learn a confidence threshold (bounded by
`label_budget: 300`). Items above the learned threshold (0.847) were decided
by the proxy alone — 863 of the 1000. So `137 (oracle) + 863 (proxy) = 1000`,
versus 1000 oracle calls without the cascade. (Quality is measured against the
oracle's answers, treated as ground truth.)

Re-running the identical pipeline reuses the cached result and makes no new
model calls — the summary replays the originally-recorded cost:

```text
Cascade filter 'is_relevant' (cached)
           proxy     gpt-4o-mini · 1000 scored · $0.0200
           oracle    gpt-4o · 137 sampled for calibration (budget 300) · $0.4000
           guarantee recall ≥ 95%  δ=0.05
           threshold 0.847 proxy confidence
           result    863 proxy-accepted + 137 calibration samples → 1000 items
           total cost $0.4200
```

### Python API

`cascade` is a named parameter on `filter`, `resolve`, and `equijoin` in the
[Frame API](../python/index.md):

```python
import docetl

kept = (
    docetl.read_json("reviews.json")
    .filter(
        prompt="Is this review about shipping problems? {{ input.text }}",
        output={"schema": {"keep": "bool"}},
        cascade={
            "proxy_model": "text-embedding-3-small",  # or a chat model
            "guarantee": "recall",
            "target": 0.9,
            "label_budget": 120,
        },
    )
    .collect()
)
```

YAML pipelines with cascades also run as usual via
`DSLRunner.from_yaml("pipeline.yaml").load_run_save()`.

## Per-operator examples

### filter

```yaml
- name: is_relevant
  type: filter
  model: gpt-4o
  prompt: "Is this document about climate policy? {{ input.text }}"
  output: { schema: { keep: "bool" } }
  cascade: { proxy_model: gpt-4o-mini, target: 0.95 }   # guarantee=recall
```

### resolve / equijoin

The cascade runs over the candidate pairs produced by **existing blocking** —
it replaces "compare every candidate pair with the oracle" with "proxy on all
pairs, oracle on a calibrated subset". Matched pairs feed the existing
clustering (resolve) / join (equijoin) unchanged.

```yaml
- name: dedupe
  type: resolve
  comparison_model: gpt-4o
  comparison_prompt: "Are these the same entity? {{ input1.name }} / {{ input2.name }}"
  blocking_threshold: 0.8
  cascade: { proxy_model: gpt-4o-mini, target: 0.9 }    # guarantee=precision
```

## Caching

The learned threshold and the cascade's labels are cached, keyed on the
operation's identity, its config, and a signature of the input items. An
identical re-run reuses the cached result and **does not re-pay** the
proxy/oracle calls. Set `bypass_cache: true` on the operation (or clear the
DocETL cache) to force recomputation.

## Cost & escalation reporting

When a cascade runs, the operation logs a summary like the one shown in the
[example above](#complete-example--run-it-end-to-end). The proxy line shows
how many items were scored and at what cost; the oracle line shows the
calibration/escalation count and cost; the result line shows the split between
proxy-decided and oracle-decided items.

The same numbers are available programmatically on the operation instance as
`op.cascade_stats` (`n_items`, `proxy_calls`, `oracle_calls`,
`escalation_rate`, `guarantee`, `target`, `delta`).

## Limitations (v1)

- Cascades only support **binary predictions** — `filter` (keep/drop),
  `resolve` / `equijoin` (match/no-match). Multiclass (`map` with enum) is not
  currently supported.
- The proxy uses a **single-token** decode with **token logprobs** to estimate
  confidence. It requires a provider/model that returns logprobs; if it
  doesn't, the proxy raises a clear error.
- `target` must be strictly between 0 and 1 — BARGAIN cannot guarantee 100%
  with finite samples.
- The proxy renders the operator's prompt with `{{ input }}` (or
  `{{ input1/input2 }}`, `{{ left/right }}`); retrieval-context and PDF inputs
  are not yet wired into the cascade path. Combining `cascade` with
  `pdf_url_key` or a `retriever` is **rejected at config validation** (rather
  than silently dropping that context) — remove the `cascade` block or those
  inputs.
