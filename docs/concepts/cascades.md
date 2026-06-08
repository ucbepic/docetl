# Model Cascades

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
for finite samples. This is the approach of
[BARGAIN](https://github.com/ucbepic/BARGAIN); DocETL depends on the BARGAIN
library directly for the statistical core (threshold learning and guarantee
certification) and wraps it with thin adapters for each operator's
proxy/oracle calls.

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

How to read this: all 1000 documents were classified by the cheap proxy. The
oracle was called on 137 of them — that single count covers everything the
expensive model touched: the sample it labeled to learn the threshold **and**
any escalated cases (the same pool, deduplicated). The other 863 documents were
decided by the proxy alone. So `137 (oracle) + 863 (proxy) = 1000`, versus 1000
oracle calls without the cascade. `label_budget: 300` is a ceiling on those
oracle calls — here only 137 were needed. (Quality is measured against the
oracle's answers, treated as ground truth.)

Re-running the identical pipeline reuses the cached result and makes no new
model calls — the line below replays the originally-recorded cost:

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
| `guarantee` | `accuracy` \| `precision` \| `recall` \| `precision+recall` | operator-specific (see below) |
| `target` | threshold for that metric, in `(0, 1)` (required) | — |
| `delta` | failure probability; guarantee holds w.p. `1 - delta` | `0.05` |
| `label_budget` | max oracle calls spent *learning* the threshold | `400` |

The raw logprob threshold is deliberately **not** exposed — it is learned from
the data, and exposing it would defeat the guarantee. You stay in metric-space.

## Guarantees

Pick the guarantee that matches the operator's intent:

- **accuracy** — output matches the oracle on at least `target` fraction of
  items. Works for any binary operator.
- **precision** — of the items returned positive, at least `target` are truly
  positive. Natural for **resolve / equijoin** (don't over-merge / over-join).
- **recall** — of the truly-positive items, at least `target` are returned.
  Natural for **filter** (don't drop relevant docs).
- **precision+recall** — both precision and recall hold simultaneously at the
  same `target`. Uses a union bound (δ/2 each) and oracle-verifies items in
  the gap between the precision and recall thresholds. Total oracle calls are
  capped at `label_budget`; if the gap exceeds the remaining budget, items are
  verified in order of proxy confidence (most likely positives first) and recall
  becomes best-effort. Increase `label_budget` for stronger recall. Useful for
  **resolve / equijoin** when you want both "don't over-merge" and "don't miss
  matches".

If you omit `guarantee`, each operator applies its natural default:
**filter → recall**, **resolve / equijoin → precision**.

For `precision+recall`, both passes share the same `target` value. If you need
different targets (e.g. precision ≥ 0.95 but recall ≥ 0.8), run two separate
operations — one with each guarantee.

If the oracle sample is too small to certify the `target` at the chosen `delta`,
the engine errs toward the guarantee — escalating more items to the oracle (or
keeping more items) — rather than silently violating it. That costs more oracle
calls, so give `label_budget` enough room for a meaningful sample on small
datasets.

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

To guarantee both precision **and** recall on the same operation, set
`guarantee: precision+recall`. The engine runs a precision pass and a recall
pass (splitting δ and the label budget equally), then oracle-verifies items in
the gap between the two thresholds so both guarantees transfer:

```yaml
- name: dedupe
  type: resolve
  comparison_model: gpt-4o
  comparison_prompt: "Are these the same entity? {{ input1.name }} / {{ input2.name }}"
  blocking_threshold: 0.8
  cascade:
    proxy_model: gpt-4o-mini
    guarantee: precision+recall   # both hold simultaneously
    target: 0.9                   # precision >= 0.9 AND recall >= 0.9
    delta: 0.05                   # P(either fails) <= 0.05 via union bound
    label_budget: 300             # split equally: 150 for each threshold search
```

The extra oracle cost for verifying the gap depends on the proxy quality — a
good proxy has a small gap (few extra calls); a weak proxy has a large gap
(many calls, but both guarantees still hold).

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
