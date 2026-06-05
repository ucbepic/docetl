# Model Cascades

Several DocETL operators issue an LLM call whose output is a single
**categorical** value — a boolean or an enum:

| Operator | Items | Categorical output |
|---|---|---|
| `filter` | records | boolean `keep` |
| `map` (enum output) | records | one enum value |
| `resolve` | candidate pairs (after blocking) | `is_match` |
| `equijoin` | candidate pairs across two datasets | `is_match` |

For these, a cheap "proxy" model is usually right, and only the hard cases
need an expensive "oracle". A **model cascade** runs the proxy on everything,
learns a confidence threshold on a small oracle-labeled sample, trusts the
proxy above the threshold, and escalates the rest to the oracle — while
preserving a **statistical guarantee** that holds with probability `1 - delta`
for finite samples. This is the approach of
[BARGAIN](https://github.com/ucbepic/BARGAIN); the statistical core is
reimplemented in DocETL so it can run over any operator's proxy/oracle calls.

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

While the filter runs, the cascade prints a one-line summary of what it did:

```text
Cascade filter 'is_relevant': 1000 items | proxy 1000 + oracle 137
(escalation 14%; 863 served by proxy) | guarantee=recall target=0.95
delta=0.05 | cost=$0.42
```

Read that as: all 1000 documents were classified by the cheap proxy; the engine
spent 137 oracle calls learning/verifying the threshold and escalating the
uncertain cases (a 14% escalation rate), while 863 documents were decided by the
proxy alone — at a recall guarantee of 0.95. Without the cascade, the same step
would have made 1000 oracle calls.

Re-running the identical pipeline reuses the cached result (no new model calls):

```text
Cascade filter 'is_relevant' (cached): 1000 items | proxy 1000 + oracle 137
(escalation 14%; 863 served by proxy) | guarantee=recall target=0.95
delta=0.05 | cost=$0.42
```

### Python API

The same config works through the Python API:

```python
from docetl import DSLRunner

runner = DSLRunner.from_yaml("pipeline.yaml")
runner.load_run_save()
```

### Config knobs

| Knob | Meaning | Default |
|---|---|---|
| `proxy_model` | the cheap model (required) | — |
| `guarantee` | `accuracy` \| `precision` \| `recall` | operator-specific (see below) |
| `target` | threshold for that metric, in `(0, 1]` (required) | — |
| `delta` | failure probability; guarantee holds w.p. `1 - delta` | `0.05` |
| `label_budget` | max oracle calls spent *learning* the threshold | `400` |

The raw logprob threshold is deliberately **not** exposed — it is learned from
the data, and exposing it would defeat the guarantee. You stay in metric-space.

## The three guarantees

Pick the guarantee that matches the operator's intent:

- **accuracy** — output matches the oracle on at least `target` fraction of
  items. Natural for **map (enum)**, where precision/recall per class is
  ill-defined.
- **precision** — of the items returned positive, at least `target` are truly
  positive. Natural for **resolve / equijoin** (don't over-merge / over-join).
- **recall** — of the truly-positive items, at least `target` are returned.
  Natural for **filter** (don't drop relevant docs).

If you omit `guarantee`, each operator applies its natural default:
**filter → recall**, **map → accuracy**, **resolve / equijoin → precision**.

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

### map (single enum)

`map` cascades are limited to a **single `enum[...]` output field** (v1). Every
row is kept; the enum is filled by the proxy where confident, by the oracle
where escalated.

```yaml
- name: categorize
  type: map
  model: gpt-4o
  prompt: "Classify the ticket: {{ input.body }}"
  output: { schema: { category: "enum[bug, feature, question]" } }
  cascade: { proxy_model: gpt-4o-mini, target: 0.9 }    # guarantee=accuracy
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

When a cascade runs, the operation logs a one-line summary, e.g.:

```
Cascade filter 'is_relevant': 1000 items | proxy 1000 + oracle 137
(escalation 14%; 863 served by proxy) | guarantee=recall target=0.95
delta=0.05 | cost=$0.42
```

The same numbers are available programmatically on the operation instance as
`op.cascade_stats` (`n_items`, `proxy_calls`, `oracle_calls`,
`escalation_rate`, `guarantee`, `target`, `delta`).

## Limitations (v1)

- The proxy uses a **single-token** decode, so a cascade supports at most **9
  labels** (booleans and small enums). Larger enums fall back to all-oracle —
  remove the `cascade` block.
- The proxy requires a provider/model that returns **token logprobs**; if it
  doesn't, the proxy raises a clear error.
- `map` cascades require a **single `enum[...]`** output; multi-field or
  free-text map outputs are not guaranteeable and should not set `cascade`.
- The proxy renders the operator's prompt with `{{ input }}` (or
  `{{ input1/input2 }}`, `{{ left/right }}`); retrieval-context and PDF inputs
  are not yet wired into the cascade path. Combining `cascade` with
  `pdf_url_key` or a `retriever` is **rejected at config validation** (rather
  than silently dropping that context) — remove the `cascade` block or those
  inputs.
