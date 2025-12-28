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
- Clickable links that open a modal/popup with source content
- Modal should show: extracted fields + original source text
- Original text can be collapsed by default with "Show original" toggle
- Embed source data as JSON in the page for JavaScript access

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

### Map Operation

Applies an LLM transformation to each document independently.

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

### Filter Operation

Keeps or removes documents based on LLM criteria. Output schema must have exactly one boolean field.

```yaml
- name: filter_relevant
  type: filter
  skip_on_error: true
  prompt: |
    Document: {{ input.text }}

    Is this document relevant to climate change?
    Respond true or false.
  output:
    schema:
      is_relevant: boolean
```

### Reduce Operation

Aggregates documents by a key using an LLM.

**Always include `fold_prompt` and `fold_batch_size`** for reduce operations. This handles cases where the group is too large to fit in context.

```yaml
- name: summarize_by_category
  type: reduce
  reduce_key: category  # use "_all" to aggregate everything
  skip_on_error: true
  prompt: |
    Summarize these {{ inputs | length }} items for category "{{ inputs[0].category }}":

    {% for item in inputs %}
    - {{ item.title }}: {{ item.description }}
    {% endfor %}

    Provide a 2-3 sentence summary of the key themes.
  fold_prompt: |
    You have a summary based on previous items, and new items to incorporate.

    Previous summary (based on {{ output.item_count }} items):
    {{ output.summary }}

    New items ({{ inputs | length }} more):
    {% for item in inputs %}
    - {{ item.title }}: {{ item.description }}
    {% endfor %}

    Write a NEW summary that covers ALL items (previous + new).

    IMPORTANT: Output a clean, standalone summary as if describing the entire dataset.
    Do NOT mention "updated", "added", "new items", or reference the incremental process.
  fold_batch_size: 100
  output:
    schema:
      summary: string
      item_count: int
  validate:
    - len(output["summary"].strip()) > 0
  num_retries_on_validate_failure: 2
```

**Critical: Writing Good Fold Prompts**

The `fold_prompt` is called repeatedly as batches are processed. Its output must:
1. **Reflect ALL data seen so far**, not just the latest batch
2. **Be a clean, standalone output** - no "updated X" or "added Y items" language
3. **Match the same schema** as the initial `prompt` output

Bad fold_prompt output: "Added 50 new projects. The updated summary now includes..."
Good fold_prompt output: "Developers are building privacy-focused tools and local-first apps..."

**Estimating `fold_batch_size`:**
- **Use 100+ for most cases** - larger batches = fewer LLM calls = lower cost
- For very long documents, reduce to 50-75
- For short documents (tweets, titles), can use 150-200
- Models like gpt-4o-mini have 128k context, so batch size is rarely the bottleneck

**Key parameters:**
- `reduce_key`: Field to group by (or list of fields, or `_all`)
- `fold_prompt`: Template for incrementally adding items to existing output (required)
- `fold_batch_size`: Number of items per fold iteration (required, use 100+)
- `associative`: Set to `false` if order matters

### Split Operation

Divides long text into smaller chunks. No LLM call.

```yaml
- name: split_document
  type: split
  split_key: content
  method: token_count  # or "delimiter"
  method_kwargs:
    num_tokens: 500
    model: gpt-5-nano
```

**Output adds:**
- `{split_key}_chunk`: The chunk content
- `{op_name}_id`: Original document ID
- `{op_name}_chunk_num`: Chunk number

### Unnest Operation

Flattens list fields into separate rows. No LLM call.

```yaml
- name: unnest_items
  type: unnest
  unnest_key: items  # field containing the list
  keep_empty: false  # optional
```

**Example:** If a document has `items: ["a", "b", "c"]`, unnest creates 3 documents, each with `items: "a"`, `items: "b"`, `items: "c"`.

### Resolve Operation

Deduplicates and canonicalizes entities. Uses pairwise comparison.

```yaml
- name: dedupe_names
  type: resolve
  optimize: true  # let optimizer find blocking rules
  skip_on_error: true
  comparison_prompt: |
    Are these the same person?

    Person 1: {{ input1.name }} ({{ input1.email }})
    Person 2: {{ input2.name }} ({{ input2.email }})

    Respond true or false.
  resolution_prompt: |
    Standardize this person's name:

    {% for entry in inputs %}
    - {{ entry.name }}
    {% endfor %}

    Return the canonical name.
  output:
    schema:
      name: string
```

**Important:** Set `optimize: true` and run `docetl build` to generate efficient blocking rules. Without blocking, this is O(n²).

### Code Operations

Deterministic Python transformations without LLM calls.

**code_map:**
```yaml
- name: compute_stats
  type: code_map
  code: |
    def transform(doc) -> dict:
        return {
            "word_count": len(doc["text"].split()),
            "char_count": len(doc["text"])
        }
```

**code_reduce:**
```yaml
- name: aggregate
  type: code_reduce
  reduce_key: category
  code: |
    def transform(items) -> dict:
        total = sum(item["value"] for item in items)
        return {"total": total, "count": len(items)}
```

**code_filter:**
```yaml
- name: filter_long
  type: code_filter
  code: |
    def transform(doc) -> bool:
        return len(doc["text"]) > 100

```

### Retrievers (LanceDB)

Augment LLM operations with retrieved context from a LanceDB index. Useful for:
- Finding related documents to compare against
- Providing additional context for extraction/classification
- Cross-referencing facts across a dataset

**Define a retriever:**
```yaml
retrievers:
  facts_index:
    type: lancedb
    dataset: extracted_facts  # dataset to index
    index_dir: workloads/wiki/lance_index
    build_index: if_missing  # if_missing | always | never
    index_types: ["fts", "embedding"]  # or "hybrid"
    fts:
      index_phrase: "{{ input.fact }}: {{ input.source }}"
      query_phrase: "{{ input.fact }}"
    embedding:
      model: openai/text-embedding-3-small
      index_phrase: "{{ input.fact }}"
      query_phrase: "{{ input.fact }}"
    query:
      mode: hybrid
      top_k: 5
```

**Use in operations:**
```yaml
- name: find_conflicts
  type: map
  retriever: facts_index
  prompt: |
    Check if this fact conflicts with any retrieved facts:

    Current fact: {{ input.fact }} (from {{ input.source }})

    Related facts from other articles:
    {{ retrieval_context }}

    Return whether there's a genuine conflict.
  output:
    schema:
      has_conflict: boolean
```

**Key points:**
- `{{ retrieval_context }}` is injected into prompts automatically
- Index is built on first use (when `build_index: if_missing`)
- Supports full-text (`fts`), vector (`embedding`), or `hybrid` search
- Use `save_retriever_output: true` to debug what was retrieved
- **Can index intermediate outputs**: Retriever can index the output of a previous pipeline step, enabling patterns like "extract facts → index facts → retrieve similar facts for each"

## Documentation Reference

For detailed parameters, advanced features, and more examples, read the docs:
- **Operations**: `docs/operators/` folder (map.md, reduce.md, filter.md, etc.)
- **Concepts**: `docs/concepts/` folder (pipelines.md, operators.md, schemas.md)
- **Examples**: `docs/examples/` folder
- **Optimization**: `docs/optimization/` folder

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

### Test Run (Required)
Add `sample: 10-20` to your first operation, then run:
```bash
docetl run pipeline.yaml
```

**Inspect the test results before proceeding:**
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
3. Run full pipeline
4. **Show final results** - distributions, key insights, trends

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

## Quick Reference

```bash
# Run pipeline
docetl run pipeline.yaml

# Run with more parallelism
docetl run pipeline.yaml --max_threads 16

# Optimize pipeline (cost/accuracy tradeoff)
docetl build pipeline.yaml --optimizer moar

# Clear LLM cache
docetl clear-cache

# Check version
docetl version
```

