# Model Cascades for DocETL operators

Status: complete — all 5 milestones landed (engine, proxy logprob path, filter
/ map(enum) / resolve / equijoin wiring, caching + cost reporting + user docs at
`docs/concepts/cascades.md`)
Owner: (you)
Branch: `claude/zen-maxwell-OPO34`

## Motivation

Several DocETL operators issue an LLM call whose output is a single
**categorical** value:

| Operator | Items | Categorical output |
|---|---|---|
| `filter` | records | boolean `keep` |
| `map` (enum output) | records | enum value |
| `resolve` | candidate pairs (post-blocking) | `is_match` |
| `equijoin` | candidate pairs across two datasets | `is_match` |

For these, a cheap "proxy" model is usually right, and only the hard cases
need an expensive "oracle". A *model cascade* runs the proxy on everything,
learns a confidence threshold on a small oracle-labeled sample, trusts the
proxy above the threshold, and escalates the rest — while preserving a
**statistical guarantee** that holds with probability `1 - delta` for finite
samples.

This is the approach of [BARGAIN](https://github.com/ucbepic/BARGAIN). We do
**not** depend on BARGAIN; its statistical core is small and is reimplemented
here (`docetl/operations/utils/cascade.py`) so it can operate over pluggable
proxy/oracle callables and be shared by all four operators.

## The single abstraction

Everything reduces to a **categorical prediction over a set of items**, so
there is one shared engine:

```python
class CategoricalCascade:
    def __init__(self, spec: CascadeSpec, proxy_predict, oracle_predict): ...
    def run(self, items: list) -> CascadeResult: ...
```

An operator supplies two thin adapters and nothing else:

- `proxy_predict(item) -> (label, confidence)` — renders the operator's
  prompt, does a **single-token decode with logprobs** on the cheap model,
  returns the label and its probability in `[0, 1]`.
- `oracle_predict(item) -> label` — the operator's existing full-quality call.

The operator differs only in (a) how it enumerates items, (b) prompt
rendering, (c) writing the label back. The statistics live entirely in the
engine.

## The three guarantees (the interface knobs)

`guarantee` is operator-dependent, so it is not hardcoded:

- **accuracy** — output matches the oracle on at least `target` fraction of
  items. Natural for **map (enum)** (multi-class; precision/recall per class
  is ill-defined). Mirrors `BARGAIN_A`.
- **precision** — of the items returned positive, at least `target` are truly
  positive. Natural for **resolve / equijoin** (don't over-merge). `BARGAIN_P`.
- **recall** — of the truly-positive items, at least `target` are returned.
  Natural for **filter** (don't drop relevant docs). `BARGAIN_R` (beta=0,
  the documented guarantee-preserving setting).

The user-facing config knobs are exactly:

| Knob | Meaning |
|---|---|
| `proxy_model` | the cheap model (oracle = operator's existing `model`) |
| `guarantee` | `accuracy` \| `precision` \| `recall` |
| `target` | threshold for that metric (0–1) |
| `delta` | failure probability; guarantee holds w.p. `1 - delta` (default 0.05) |
| `label_budget` | max oracle calls spent *learning* the threshold (P/R) |

We deliberately do **not** expose the raw logprob threshold — that is learned,
and exposing it would defeat the guarantee. The interface stays in
metric-space.

### Proposed config (per operator, opt-in)

```yaml
- name: is_relevant
  type: filter
  model: gpt-4o                      # oracle
  output: {schema: {keep: "bool"}}
  cascade:
    proxy_model: gpt-4o-mini
    guarantee: recall                # default for filter
    target: 0.95
    delta: 0.05
    label_budget: 300
```

Defaults if `guarantee` is omitted: filter→recall, map→accuracy,
resolve/equijoin→precision. `syntax_check` validates compatibility
(enum-map → accuracy only).

## Architecture / where it hooks in

Cascade is a **batch-level** rewrite (proxy-all → calibrate → escalate), not a
per-item one. Each operator gains a branch at `execute()` start when `cascade`
is set:

- **filter** (`filter.py`) — items = records; write-back via `_handle_result`.
  Cleanest vertical slice.
- **map** (`map.py`) — restricted to single-enum output in v1.
- **resolve** (`resolve.py:172`) / **equijoin** (`equijoin.py`) — items =
  candidate pairs *after existing blocking*; matched pairs feed the existing
  clustering / join.

### Confidence path (new LLM mechanism)

DocETL currently calls models in `OutputMode.TOOLS` / `STRUCTURED_OUTPUT`
(`api.py:724`), which don't surface usable logprobs. A dedicated path is
needed for proxies:

```python
APIWrapper.classify_with_logprob(model, messages, labels) -> (label, prob)
```

It renders the label set as a **single-token menu** (`1=bug 2=feature ...`) so
booleans and enums both decode to one token regardless of label text, calls
`litellm.completion(..., logprobs=True, top_logprobs=k)`, and maps the answer
token's probability back to the label. Only the proxy uses this; the oracle
keeps the operator's normal structured path, so output fidelity is unchanged.

### Calibration caching

Threshold learning runs once at `execute()` start; the learned threshold +
sample labels should be cached (DocETL cache, keyed on op name + config hash +
dataset signature) so re-runs don't re-pay the labeling cost.

## Milestones

1. **Core engine + statistical tests** (this commit) —
   `docetl/operations/utils/cascade.py`, model-free correctness tests.
2. `classify_with_logprob` in `api.py` + logprob tests.
3. Filter vertical slice end-to-end.
4. Generalize to map(enum), resolve, equijoin.
5. Caching, cost/escalation reporting, docs.

## Risks

- **Logprob availability** varies by provider — need a clear error + optional
  self-reported-confidence fallback.
- **map with mixed outputs** isn't cleanly guaranteeable — v1 limits to
  single-enum.
- **Calibration cost** — `label_budget` is a hard ceiling; tiny datasets may
  not afford a meaningful sample (warn + fall back to all-oracle).

## Attribution

The statistical procedures (betting confidence sequences, without-replacement
sampling, and the accuracy/precision/recall threshold search) are ports of the
methods in BARGAIN (UC Berkeley EPIC lab). Reimplemented here, not vendored.

### Dependency decision: vendor the statistical core

We deliberately do **not** take a runtime dependency on the `bargain` package:

- It is **not published on PyPI** (0.1.1, MIT) — the only way to depend on it
  is a git pin to a research repo with no release cadence, which is fragile
  for a core runtime feature.
- It ships `openai` + `pandas` as hard deps and its `Proxy`/`Oracle` model is
  OpenAI-logprob-specific; DocETL goes through litellm, so we need our own
  proxy/oracle adapters regardless.
- The reused math is ~120 lines of pure numpy and is mathematically frozen;
  MIT permits a verbatim, attributed port, and the coverage tests
  (`tests/test_cascade_core.py`) guard against regressions.

Trade-off accepted: we own correctness of the port and will not auto-receive
upstream fixes. The `proxy_predict`/`oracle_predict` callable seam keeps the
door open to an *optional* `bargain` backend later without changing operators.

