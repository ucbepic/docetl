---
name: docetl
description: Build and run LLM-powered data processing pipelines with DocETL. Use when users say "docetl", want to analyze unstructured data, process documents, extract information, or run ETL tasks on text. Helps with data collection, pipeline creation, execution, and optimization.
---

# DocETL Pipeline Development

DocETL is a system for creating LLM-powered data processing pipelines. This skill helps you build end-to-end pipelines: from data preparation to execution and optimization.

## Workflow Overview: Iterative Data Analysis

Work like a data analyst: **write → run → inspect → iterate**. Never write all scripts at once and run them all at once. Each phase should be completed and validated before moving to the next.

### Phase 1: Data Collection
1. Write data collection script
2. **Run it immediately** (with user permission)
3. **Inspect the dataset** - show the user:
   - Total document count
   - Keys/fields in each document
   - Sample documents (first 3-5)
   - Length distribution (avg chars, min/max)
   - Any other relevant statistics
4. Iterate if needed (e.g., collect more data, fix parsing issues)

### Phase 2: Pipeline Development
1. Read sample documents to understand format
2. Write pipeline YAML with `sample: 10-20` for testing
3. **Run the test pipeline**
4. **Inspect intermediate results** - show the user:
   - Extraction quality on samples
   - Domain/category distributions
   - Any validation failures
5. Iterate on prompts/schema based on results
6. Remove `sample` parameter and run full pipeline
7. **Show final results** - distributions, trends, key insights

### Phase 3: Visualization & Presentation
1. Write visualization script based on actual output structure
2. **Run and show the report** to the user
3. Iterate on charts/tables if needed

**Visualization Aesthetics:**
- **Clean and minimalist** - no clutter, generous whitespace
- **Warm and elegant color theme** - 1-2 accent colors max
- **Subtle borders** - not too rounded (border-radius: 8-10px max)
- **Sans-serif fonts** - system fonts like -apple-system, Segoe UI, Roboto
- **"Created by DocETL"** - add subtitle after the main title
- **Mix of charts and tables** - charts for distributions, tables for detailed summaries
- **Light background** - off-white (#f5f5f5) with white cards for content

**Report Structure:**
1. Title + "Created by DocETL" subtitle
2. Key stats cards (document count, categories, etc.)
3. Distribution charts (bar charts, pie charts)
4. Summary table with detailed analysis
5. Minimal footer

**Interactive Tables:**
- **All truncated content must be expandable** - never use static "..." truncation
- Long text: Show first ~250 chars with "(show more)" toggle
- Long lists: Show first 4-6 items with "(+N more)" toggle
- Use JavaScript to toggle visibility, not page reloads

**Source Document Links:**
- **Link aggregated results to source documents** - users should be able to drill down
- **Never use external links or file:// URLs** — all data must be self-contained in the HTML
- Embed source data as a `<script>` JSON blob in the page, then use JavaScript onclick handlers to show modals
- Modal should show: extracted fields + original source text
- Original text can be collapsed by default with "Show original" toggle
- **Never generate `<a href="...">` links to local files, APIs, or URLs that won't resolve** — use `onclick` handlers with inline data instead

**Key principle:** The user should see results at every step. Don't proceed to the next phase until the current phase produces good results.

## Step 1: Data Preparation

DocETL datasets must be **JSON arrays** or **CSV files**.

### JSON Format
```json
[
  {"id": 1, "text": "First document content...", "metadata": "value"},
  {"id": 2, "text": "Second document content...", "metadata": "value"}
]
```

### CSV Format
```csv
id,text,metadata
1,"First document content...","value"
2,"Second document content...","value"
```

### Data Collection Scripts

If user needs to collect data, write a Python script:

```python
import json

# Collect/transform data
documents = []
for source in sources:
    documents.append({
        "id": source.id,
        "text": source.content,  # DO NOT truncate text
        # Add relevant fields
    })

# Save as DocETL dataset
with open("dataset.json", "w") as f:
    json.dump(documents, f, indent=2)
```

**Important:** Never truncate document text in collection scripts. DocETL operations like `split` handle long documents properly. Truncation loses information.

### After Running Data Collection

**Always run the collection script and inspect results before proceeding.** Show the user:

```python
import json
data = json.load(open("dataset.json"))

print(f"Total documents: {len(data)}")
print(f"Keys: {list(data[0].keys())}")
print(f"Avg length: {sum(len(str(d)) for d in data) // len(data)} chars")

# Show sample
print("\nSample document:")
print(json.dumps(data[0], indent=2)[:500])
```

Only proceed to pipeline development once the data looks correct.

## Step 2: Read and Understand the Data

**CRITICAL**: Before writing any prompts, READ the actual input data to understand:
- The structure and format of documents
- The vocabulary and terminology used
- What information is present vs. absent
- Edge cases and variations

```python
import json
with open("dataset.json") as f:
    data = json.load(f)
# Examine several examples
for doc in data[:5]:
    print(doc)
```

This understanding is essential for writing specific, effective prompts.

## Step 3: Pipeline Structure

Create a YAML file with this structure:

```yaml
default_model: gpt-5-nano
ui: "web"  # Always set when agent authors the pipeline

system_prompt:
  dataset_description: <describe the data based on what you observed>
  persona: <role for the LLM to adopt>

datasets:
  input_data:
    type: file
    path: "dataset.json"  # or dataset.csv

operations:
  - name: <operation_name>
    type: <operation_type>
    prompt: |
      <Detailed, specific prompt based on the actual data>
    output:
      schema:
        <field_name>: <type>

pipeline:
  steps:
    - name: process
      input: input_data
      operations:
        - <operation_name>
  output:
    type: file
    path: "output.json"
    intermediate_dir: "intermediates"  # ALWAYS set this for debugging
```

### Key Configuration

- **default_model**: Use `gpt-5-nano` or `gpt-5-mini` for extraction/map operations
- **intermediate_dir**: Always set to log intermediate results
- **system_prompt**: Describe the data based on what you actually observed

### Model Selection by Operation Type

| Operation Type | Recommended Model | Rationale |
|---------------|-------------------|-----------|
| Map (extraction) | `gpt-5-nano` or `gpt-5-mini` | High volume, simple per-doc tasks |
| Filter | `gpt-5-nano` | Simple yes/no decisions |
| Reduce (summarization) | `gpt-4.1` or `gpt-5.1` | Complex synthesis across many docs |
| Resolve (deduplication) | `gpt-5-nano` or `gpt-5-mini` | Simple pairwise comparisons |

Use cheaper models for high-volume extraction, and more capable models for synthesis/summarization where quality matters most.

## Step 4: Writing Effective Prompts

**Prompts must be specific to the data, not generic.** After reading the input data:

### Bad (Generic) Prompt
```yaml
prompt: |
  Extract key information from this document.
  {{ input.text }}
```

### Good (Specific) Prompt
```yaml
prompt: |
  You are analyzing a medical transcript from a doctor-patient visit.

  The transcript follows this format:
  - Doctor statements are prefixed with "DR:"
  - Patient statements are prefixed with "PT:"
  - Timestamps appear in brackets like [00:05:23]

  From the following transcript, extract:
  1. All medications mentioned (brand names or generic)
  2. Dosages if specified
  3. Patient-reported side effects or concerns

  Transcript:
  {{ input.transcript }}

  Be thorough - patients often mention medication names informally.
  If a medication is unclear, include it with a note.
```

### Prompt Writing Guidelines

1. **Describe the data format** you observed
2. **Be specific about what to extract** - list exact fields
3. **Mention edge cases** you noticed in the data
4. **Provide examples** if the task is ambiguous
5. **Set expectations** for handling missing/unclear information

## Step 5: Choosing Operations

Many tasks only need a **single map operation**. Use good judgement:

| Task | Recommended Approach |
|------|---------------------|
| Extract info from each doc | Single `map` |
| Multiple extractions | Multiple `map` operations chained |
| Extract then summarize | `map` → `reduce` |
| Filter then process | `filter` → `map` |
| Split long docs | `split` → `map` → `reduce` |
| Deduplicate entities | `map` → `unnest` → `resolve` |

## Operation Reference

**Do not rely on hardcoded documentation below — always read the current docs from the repo.** Before writing any operation, read the relevant doc file:

- **Map**: `docs/operators/map.md`
- **Filter**: `docs/operators/filter.md`
- **Reduce**: `docs/operators/reduce.md`
- **Resolve**: `docs/operators/resolve.md`
- **Split**: `docs/operators/split.md`
- **Unnest**: `docs/operators/unnest.md`
- **Code operations**: `docs/operators/code.md`
- **Equijoin**: `docs/operators/equijoin.md`
- **Retrievers**: `docs/retrievers.md`
- **Pipelines**: `docs/concepts/pipelines.md`
- **Schemas**: `docs/concepts/schemas.md`
- **Optimization**: `docs/optimization/overview.md`
- **Interactive progress (web UI)**: `docs/execution/interactive-progress.md`
- **Running pipelines**: `docs/execution/running-pipelines.md`

Read the doc file for the operation type you're about to use. The docs contain the full parameter reference, examples, and edge cases.

### Quick Examples (for orientation only — check docs for current syntax)

#### Map Operation

```yaml
- name: extract_info
  type: map
  prompt: |
    Analyze this document:
    {{ input.text }}

    Extract the main topic and 3 key points.
  output:
    schema:
      topic: string
      key_points: list[string]
  model: gpt-5-nano  # optional, uses default_model if not set
  skip_on_error: true  # recommended for large-scale runs
  validate:  # optional
    - len(output["key_points"]) == 3
  num_retries_on_validate_failure: 2  # optional
```

**Key parameters:**
- `prompt`: Jinja2 template, use `{{ input.field }}` to reference fields
- `output.schema`: Define output structure
- `skip_on_error`: Set `true` to continue on LLM errors (recommended at scale)
- `validate`: Python expressions to validate output
- `sample`: Process only N documents (for testing)
- `limit`: Stop after producing N outputs

#### Filter

One boolean output field. Read `docs/operators/filter.md` for full syntax.

#### Reduce

Aggregates docs by key. **Always include `fold_prompt` and `fold_batch_size`** — the fold prompt must produce a clean standalone output (not "updated X" or "added Y"). Use `reduce_key: "_all"` to aggregate everything. Read `docs/operators/reduce.md`.

#### Split

Divides long text into chunks (no LLM). Read `docs/operators/split.md`.

#### Unnest

Flattens list fields into rows (no LLM). Read `docs/operators/unnest.md`.

#### Resolve

Pairwise deduplication. Set `optimize: true` and run `docetl build` for blocking rules (without blocking it's O(n²)). Uses `comparison_prompt` + `resolution_prompt`. Read `docs/operators/resolve.md`.

#### Code Operations

Deterministic Python: `code_map`, `code_reduce`, `code_filter`. Each defines a `transform` function. Read `docs/operators/code.md`.

#### Retrievers

Augment operations with LanceDB retrieval context (`{{ retrieval_context }}`). Supports full-text, embedding, or hybrid search. Read `docs/retrievers.md`.

## Step 6: Environment Setup

Before running, verify API keys exist:

```bash
# Check for .env file
cat .env
```

Required keys depend on the model:
- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Google: `GEMINI_API_KEY`

If missing, prompt user to create `.env`:
```
OPENAI_API_KEY=sk-...
```

## Step 7: Execution

**Always test on a sample first, then run full pipeline.**

### Before First Run: Start the Feedback Server

```bash
# Start once per session — survives across pipeline runs
docetl serve &
PORT=$(cat .docetl_server_port)
```

### Test Run (Required)
Add `sample: 10-20` to your first operation, then run:
```bash
docetl run pipeline.yaml

# Wait for human feedback (blocks until they click "Done reviewing")
FEEDBACK=$(curl -s http://localhost:$PORT/feedback/wait)
echo "$FEEDBACK"
```

**Inspect the test results AND human feedback before proceeding:**
```python
import json
from collections import Counter

# Load intermediate results
data = json.load(open("intermediates/step_name/operation_name.json"))

print(f"Processed: {len(data)} docs")

# Check distributions
if "domain" in data[0]:
    print("Domain distribution:")
    for k, v in Counter(d["domain"] for d in data).most_common():
        print(f"  {k}: {v}")

# Show sample outputs
print("\nSample output:")
print(json.dumps(data[0], indent=2))
```

### Full Run
Once test results look good:
1. Remove the `sample` parameter from the pipeline
2. Ask user for permission (estimate cost based on test run)
3. Run full pipeline: `docetl run pipeline.yaml`
4. **Wait for feedback**: `curl -s http://localhost:$PORT/feedback/wait`
5. **Read and act on feedback** before declaring success

Options:
- `--max_threads N` - Control parallelism

Check intermediate results in the `intermediate_dir` folder to debug each step.

## Step 8: Optimization (Optional)

Use MOAR optimizer to find the Pareto frontier of **cost vs. accuracy** tradeoffs. MOAR experiments with different pipeline rewrites and models to find optimal configurations.

Add to pipeline YAML:

```yaml
optimizer_config:
  type: moar
  save_dir: ./optimization_results
  available_models:
    - gpt-5-nano
    - gpt-4o-mini
    - gpt-4o
  evaluation_file: evaluate.py  # User must provide
  metric_key: score
  max_iterations: 20
  model: gpt-5-nano
```

Create evaluation file (`evaluate.py`):
```python
def evaluate(outputs: list[dict]) -> dict:
    # Score the outputs (0-1 scale recommended)
    correct = sum(1 for o in outputs if is_correct(o))
    return {"score": correct / len(outputs)}
```

Run optimization:
```bash
docetl build pipeline.yaml --optimizer moar
```

MOAR will produce multiple pipeline variants on the Pareto frontier - user can choose based on their cost/accuracy preferences.

## Output Schemas

**Keep schemas minimal and simple** unless the user explicitly requests more fields. Default to 1-3 output fields per operation. Only add more fields if the user specifically asks for them.

**Nesting limit:** Maximum 2 levels deep (e.g., `list[{field: str}]` is allowed, but no deeper).

```yaml
# Good - minimal, focused on the core task
output:
  schema:
    summary: string

# Good - a few fields when task requires it
output:
  schema:
    topic: string
    keywords: list[string]

# Acceptable - 2 levels of nesting (list of objects)
output:
  schema:
    items: "list[{name: str, value: int}]"

# Bad - too many fields (unless user explicitly requested all of these)
output:
  schema:
    conflicts_found: bool
    num_conflicts: int
    conflicts: "list[{claim_a: str, source_a: str, claim_b: str, source_b: str}]"
    analysis_summary: str

# Bad - more than 2 levels of nesting (not supported)
output:
  schema:
    data: "list[{nested: {too: {deep: str}}}]"
```

**Guidelines:**
- Start with the minimum fields needed to answer the user's question
- Avoid complex nested objects unless explicitly requested
- If you need structured data, prefer multiple simple operations over one complex schema
- Complex schemas increase LLM failures and costs

Supported types: `string`, `int`, `float`, `bool`, `list[type]`, `enum`

## Validation

**Always add validation to LLM-powered operations** (map, reduce, filter, resolve). Validation catches malformed outputs and retries automatically.

```yaml
- name: extract_keywords
  type: map
  prompt: |
    Extract 3-5 keywords from: {{ input.text }}
  output:
    schema:
      keywords: list[string]
  validate:
    - len(output["keywords"]) >= 3
    - len(output["keywords"]) <= 5
  num_retries_on_validate_failure: 2
```

Common validation patterns:
```yaml
# List length constraints
- len(output["items"]) >= 1
- len(output["items"]) <= 10

# Enum/allowed values
- output["sentiment"] in ["positive", "negative", "neutral"]

# String not empty
- len(output["summary"].strip()) > 0

# Numeric ranges
- output["score"] >= 0
- output["score"] <= 100
```

## Jinja2 Templating

For map operations, use `input`:
```yaml
prompt: |
  Document: {{ input.text }}
  {% if input.metadata %}
  Context: {{ input.metadata }}
  {% endif %}
```

For reduce operations, use `inputs` (list):
```yaml
prompt: |
  Summarize these {{ inputs | length }} items:
  {% for item in inputs %}
  - {{ item.summary }}
  {% endfor %}
```

## Troubleshooting

### Pipeline won't run
- Check `.env` has correct API keys
- Verify dataset file exists and is valid JSON/CSV
- Check YAML syntax

### Bad outputs
- Read more input data examples to improve prompt specificity
- Add `validate` rules with retries
- Simplify output schema
- Add concrete examples to prompt

### High costs
- Use `gpt-5-nano` or `gpt-4o-mini`
- Add `sample: 10` to test on subset first
- Run MOAR optimizer to find cost-efficient rewrites

### Check intermediate results
Look in `intermediate_dir` folder to debug each step.

## Human-in-the-Loop Feedback Workflow

**When an LLM agent authors a pipeline, always set `ui: "web"` in the pipeline YAML.** This opens a browser-based feedback UI where the human can watch outputs stream in, click on any operation to inspect its results, give per-document or pipeline-level feedback, and kill the pipeline. The agent orchestrates the feedback loop.

```yaml
# Add to pipeline YAML top level (next to default_model)
ui: "web"       # Opens browser feedback UI automatically
default_model: gpt-5-nano
```

Three modes:
- `ui: "web"` — Browser-based feedback UI (use when agent runs the pipeline)
- `ui: "tui"` — Terminal dashboard (use for manual CLI runs)
- `ui: "none"` — No interactive UI (default)

### Starting the Feedback Server

**Always start a persistent feedback server before running any pipeline.** This keeps the browser UI alive across multiple pipeline runs so the human can continuously review results and give feedback.

```bash
# Start the server in the background (run this ONCE at the start of the session)
docetl serve &
```

This opens a browser and prints the server port. The port is saved to `.docetl_server_port`. All subsequent `docetl run` commands with `ui: "web"` automatically detect the server and push results to it — no restart, no broken browser connections.

### Automatic Feedback Detection

A project hook (`.claude/hooks/poll-feedback.sh`) automatically checks the feedback server for new human feedback. It fires:
- **After every Bash command** (PostToolUse)
- **After every agent turn** (Stop event)

The hook deduplicates — it only injects feedback you haven't seen yet. When new feedback arrives, it appears as `additionalContext` in your next turn automatically.

**The human can submit feedback at any time** — not just right after a pipeline run. The hook catches it on your next turn.

If you need to explicitly block until feedback arrives (e.g., right after a run):
```bash
PORT=$(cat .docetl_server_port)
curl -s http://localhost:$PORT/feedback/wait
```

The `/feedback/wait` endpoint blocks until the human signals they're done, then returns all collected feedback as JSON:
```json
{
  "doc_feedback": [{"operation": "summarize", "doc_index": 3, "feedback": "...", "timestamp": "..."}],
  "pipeline_feedback": [{"feedback": "...", "timestamp": "..."}],
  "killed": false
}
```

**Never skip the feedback wait.** If you run the pipeline and immediately move to the next step, the human's feedback is lost. The pattern is always: **run → wait → read feedback → iterate**.

To poll without blocking (e.g., to check if any feedback has come in yet):
```bash
curl -s http://localhost:$PORT/feedback/poll
```

### Sending Messages to the Human (Toasts)

Send toast notifications to the web UI so the human sees your status or questions:

```bash
curl -X POST http://localhost:<PORT>/message \
  -H "Content-Type: application/json" \
  -d '{"text": "I noticed 3 docs got negative feedback. Updating the prompt to add more detail.", "type": "info"}'
```

The `type` field can be `"info"`, `"success"`, `"warning"`, or `"error"`. The port is printed to stdout when the UI starts.

### Toasts with Action Buttons

For decisions that need explicit user confirmation, add `actions` — the toast will show buttons the user must click:

```bash
curl -X POST http://localhost:<PORT>/message \
  -H "Content-Type: application/json" \
  -d '{"text": "I updated the prompt to require quantitative results. Re-run the pipeline?", "type": "info", "actions": ["Confirm re-run", "Dismiss"]}'
```

Toasts with actions stay on screen until the user clicks a button. The response prints to stdout:

```
[TOAST:response] id=3 action=Confirm re-run
```

Use action toasts for:
- Confirming re-runs after prompt changes
- Approving pipeline structural changes (adding/removing operations)
- Approving model upgrades that increase cost

Use plain toasts (no actions) for:
- Acknowledging feedback ("Got it, looking at this...")
- Status updates ("Analyzing 5 feedback items...")
- Asking for more feedback ("Could you review a few more docs?")

### Feedback Loop Strategy

When you receive feedback, follow this cycle:

1. **Acknowledge** — Send a plain toast so the human knows you saw their feedback.
2. **Diagnose** — Categorize the feedback:
   - *Prompt quality*: outputs are wrong, vague, missing info, or in the wrong format.
   - *Schema mismatch*: output fields don't match what the human expects.
   - *Pipeline structure*: wrong operations, missing steps, or wrong order.
   - *Data quality*: input data has issues the pipeline can't fix.
3. **Propose a fix** — Send a toast with actions (e.g., `["Confirm re-run", "Dismiss"]`) describing what you plan to change. Wait for the `[TOAST:response]` line on stdout.
4. **Apply the fix** — Only after confirmation, edit the YAML pipeline and/or prompts.
5. **Re-run** — Execute the pipeline again so the human sees updated results.
6. **Check** — Monitor for new feedback. If quality improves, send a success toast. If not, iterate.

### Common Fix Patterns

**Updating prompts** (most common fix for doc-level feedback):
- Read the specific feedback and the current prompt in the YAML
- Add concrete examples, clarify instructions, or constrain the output format
- If multiple docs got similar feedback, address the pattern, not each individually

**Asking the human to label more docs**:
```bash
curl -X POST http://localhost:<PORT>/message \
  -d '{"text": "I see feedback on 2 docs but need more examples to understand the pattern. Could you review a few more docs and leave feedback?", "type": "info"}'
```

**Proposing pipeline changes** (for structural feedback):
```bash
curl -X POST http://localhost:<PORT>/message \
  -d '{"text": "Based on your feedback, I think we need a filter step before summarize to remove irrelevant sections. I will update the pipeline — send feedback if you disagree.", "type": "warning"}'
```

**Confirming before big changes**:
```bash
curl -X POST http://localhost:<PORT>/message \
  -d '{"text": "This would require changing the model from gpt-4o-mini to gpt-4o, which costs ~10x more. Should I proceed?", "type": "warning"}'
```
Wait for pipeline-level feedback from the human before proceeding.

### When to Stop Iterating

- The human sends positive pipeline feedback ("looks good", "ship it")
- No new negative feedback after a re-run
- The human kills the pipeline (you'll see `PipelineKilled` in the output)

## Quick Reference

```bash
# Start persistent feedback server (do this first for agent workflows)
docetl serve &

# Run pipeline (auto-connects to persistent server if running)
docetl run pipeline.yaml

# Run with more parallelism
docetl run pipeline.yaml --max_threads 16

# Poll for human feedback
curl http://localhost:<PORT>/feedback/poll

# Send a message/toast to the human
curl -X POST http://localhost:<PORT>/message \
  -H "Content-Type: application/json" \
  -d '{"text": "Updating prompt based on feedback...", "type": "info"}'

# Optimize pipeline (cost/accuracy tradeoff)
docetl build pipeline.yaml --optimizer moar

# Clear LLM cache
docetl clear-cache

# Check version
docetl version
```

