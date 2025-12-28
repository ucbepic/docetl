---
name: docetl
description: Build and run LLM-powered data processing pipelines with DocETL. Use when users say "docetl", want to analyze unstructured data, process documents, extract information, or run ETL tasks on text. Helps with data collection, pipeline creation, execution, and optimization.
---

# DocETL Pipeline Development

DocETL is a system for creating LLM-powered data processing pipelines. This skill helps you build end-to-end pipelines: from data preparation to execution and optimization.

## Workflow Overview

1. **Understand the task** - What data? What processing?
2. **Prepare data** - Transform into DocETL dataset (JSON/CSV)
3. **Read and understand the data** - Examine actual documents to write specific prompts
4. **Author pipeline** - Create YAML configuration with detailed, data-specific prompts
5. **Verify environment** - Check API keys in `.env`
6. **Execute pipeline** - Run with `docetl run`
7. **Optimize (optional)** - Use MOAR optimizer for cost/accuracy tradeoffs

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

- **default_model**: Use `gpt-5-nano` unless user specifies otherwise
- **intermediate_dir**: Always set to log intermediate results
- **system_prompt**: Describe the data based on what you actually observed

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
  fold_prompt: |
    You have an existing summary and new items to incorporate.

    Existing summary:
    {{ output.summary }}

    New items to add:
    {% for item in inputs %}
    - {{ item.title }}: {{ item.description }}
    {% endfor %}

    Update the summary to include the new information.
  fold_batch_size: 10  # Estimate based on doc size and model context window
  output:
    schema:
      summary: string
  validate:
    - len(output["summary"].strip()) > 0
  num_retries_on_validate_failure: 2
```

**Estimating `fold_batch_size`:**
- Consider document size and model context window
- For gpt-4o-mini (128k context): ~50-100 small docs, ~10-20 medium docs
- For gpt-4o (128k context): similar to gpt-4o-mini
- Use WebSearch to look up context window sizes for unfamiliar models

**Key parameters:**
- `reduce_key`: Field to group by (or list of fields, or `_all`)
- `fold_prompt`: Template for incrementally adding items to existing output (required)
- `fold_batch_size`: Number of items per fold iteration (required)
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

**Always ask user before running** - LLM calls cost money.

```bash
docetl run pipeline.yaml
```

Options:
- `--max_threads N` - Control parallelism

Check intermediate results in the `intermediate_dir` folder to debug.

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

