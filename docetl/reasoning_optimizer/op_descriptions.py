from typing import Optional

from pydantic import BaseModel, Field


class Operator(BaseModel):
    """Operator model for each DocETL operator."""

    # Fields matching your spreadsheet columns
    name: str = Field(..., description="Operation name")
    type_llm_or_not: str = Field(
        ..., description="Type (LLM-powered or not LLM-powered)"
    )
    description: str = Field(..., description="Description")
    when_to_use: str = Field(..., description="When to Use")
    required_parameters: str = Field(..., description="Required Parameters")
    optional_parameters: Optional[str] = Field(None, description="Optional Parameters")
    returns: str = Field(..., description="Returns")
    minimal_example_configuration: str = Field(
        ..., description="Minimal Example Configuration"
    )

    def to_string(self) -> str:
        """Serialize operator for prompts."""
        parts = [
            f"## {self.name} ({self.type_llm_or_not})",
            f"**Description:** {self.description}",
            f"**When to Use:** {self.when_to_use}",
            f"**Required Parameters:**\n{self.required_parameters}",
        ]

        if self.optional_parameters:
            parts.append(f"**Optional Parameters:**\n{self.optional_parameters}")

        parts.append(f"**Returns:** {self.returns}")
        parts.append(
            f"**Example Configuration:**\n{self.minimal_example_configuration}\n"
        )
        return "\n\n".join(parts)


op_map = Operator(
    name="Map",
    type_llm_or_not="LLM-powered",
    description="Processes each document independently by making an LLM call with your prompt template. Creates one output for each input document, with the output conforming to your specified schema.",
    when_to_use="Use when you need to process each document individually - like extracting, summarizing, classifying, or generating new fields based on document content.",
    required_parameters="""
    name: Unique name for the operation
    type: Must be "map"
    model: LLM to use to execute the prompt
    prompt: Jinja2 template with {{ input.key }} for each document field you want to reference
    output.schema: Dictionary defining the structure and types of the output""",
    optional_parameters="""
    gleaning: Iteratively refines outputs that don't meet quality criteria. The LLM reviews its initial output and improves it based on validation feedback. Config requires:
    - if: Python expression for when to refine; e.g., "len(output[key] == 0)" (optional)
    - num_rounds: Max refinement iterations
    - validation_prompt: What to improve
    - model: LLM to use to execute the validation prompt and provide feedback (defaults to same model as operation model)
    (Default: gleaning is not enabled)

    calibrate: (bool) Processes a sample of documents first to create reference examples, then uses those examples in all subsequent prompts to ensure consistent outputs across the dataset
    (Default: calibrate is False)

    num_calibration_docs: Number of docs to use for calibration (default: 10)""",
    returns="Each original document, augmented with new keys specified in output_schema",
    minimal_example_configuration="""
    name: gen_insights
    type: map
    model: gpt-4o-mini
    prompt: From the user log below, list 2-3 concise insights (1-2 words each) and 1-2 supporting actions per insight. Return as a list of dictionaries with 'insight' and 'supporting_actions'. Log: {{ input.log }}
    output:
        schema:
            insights_summary: "string"
    """,
)

op_extract = Operator(
    name="Extract",
    type_llm_or_not="LLM-powered",
    description="""Pulls out specific portions of text exactly as they appear in the source document. The LLM identifies which parts to extract by providing line number ranges or regex patterns. Extracted text is saved to the original field name with "_extracted" suffix (e.g., report_text → report_text_extracted).""",
    when_to_use="Use when you need exact text from documents - like pulling out direct quotes, specific contract clauses, key findings, or any content that must be preserved word-for-word without LLM paraphrasing.",
    required_parameters="""
    name: Unique name for the operation
    type: Must be "extract"
    prompt: Instructions for what to extract (this is NOT a Jinja template; we will run the prompt independently for each key in document_keys)
    document_keys: List of document fields to extract from
    model: LLM to use to execute the extraction
    """,
    optional_parameters="""
    extraction_method: How the LLM specifies what to extract from each document:
    - "line_number" (default): The LLM outputs the line numbers or ranges in the document to extract. Use when relevant information is best identified by its position within each document.
    - "regex": The LLM generates a custom regex pattern for each document to match the target text. Use when the information varies in location but follows identifiable text patterns (e.g., emails, dates, or structured phrases).
    """,
    returns="""Each original document, augmented with {key}_extracted for each key in the specified document_keys""",
    minimal_example_configuration="""
    name: findings
    type: extract
    prompt: Extract all sections that discuss key findings, results, or conclusions from this research report. Focus on paragraphs that:
    - Summarize experimental outcomes
    - Present statistical results
    - Describe discovered insights
    - State conclusions drawn from the research
    Only extract the most important and substantive findings.
    document_keys: ["report_text"]
    model: gpt-4o-mini
    """,
)

op_parallel_map = Operator(
    name="Parallel Map",
    type_llm_or_not="LLM-powered",
    description="Runs multiple independent map operations concurrently on each document. Each prompt generates specific fields, and all outputs are combined into a single result per input document. More efficient than sequential maps when transformations are independent.",
    when_to_use="Use when you need multiple independent analyses of the same document - like extracting different types of information, running multiple classifications, or generating various summaries from the same input.",
    required_parameters="""
    name: Unique name for the operation
    type: Must be "parallel_map"
    prompts: List of prompt configs, each with:
    - prompt: Jinja2 template (should reference document fields using {{ input.key }})
    - output_keys: List of fields this prompt generates
    output.schema: Combined schema for all outputs. This must be the union of the output_keys from all prompts—every key in the output schema should be generated by at least one prompt, and no keys should be missing.
    """,
    optional_parameters="""
    model: Default LLM for all prompts
    Per-prompt options:
    - model: (optional) Override the default LLM for this specific prompt
    - gleaning: (optional) Validation and refinement configuration for this prompt, akin to gleaning in map operation
    """,
    returns="Each original document augmented with new keys specified in output_schema",
    minimal_example_configuration="""
    name: analyze_resume
    type: parallel_map
    prompts:
    - prompt: Extract skills from: {{ input.resume }}
        output_keys: [skills]
        model: gpt-4o-mini
    - prompt: Calculate years of experience from: {{ input.resume }}
        output_keys: [years_exp]
    - prompt: Rate writing quality 1-10: {{ input.cover_letter }}
        output_keys: [writing_score]
    output:
        schema:
            skills: list[string]
            years_exp: float
            writing_score: integer
    """,
)

op_filter = Operator(
    name="Filter",
    type_llm_or_not="LLM-powered",
    description="Evaluates each document with an LLM prompt and only keeps documents where the output is true. Documents evaluating to false are removed from the dataset entirely.",
    when_to_use="Use when you need to keep only documents meeting specific criteria - like filtering high-impact articles, relevant records, quality content, or documents matching complex conditions that require LLM judgment.",
    required_parameters="""
    name: Unique name for the operation
    type: Must be "filter"
    model: LLM to use to execute the prompt
    prompt: Jinja2 template that guides the LLM to return true or false (should reference document fields using {{ input.key }})
    output.schema: Must contain exactly one boolean field
    """,
    optional_parameters="""
    gleaning: Iteratively refines outputs that don't meet quality criteria. The LLM reviews its initial output and improves it based on validation feedback. Config requires:
    - if: Python expression for when to refine; e.g., "len(output[key] == 0)" (optional)
    - num_rounds: Max refinement iterations
    - validation_prompt: What to improve
    - model: LLM to use to execute the validation prompt and provide feedback (defaults to same model as operation model)
    (Default: gleaning is not enabled)

    calibrate: (bool) Processes a sample of documents first to create reference examples, then uses those examples in all subsequent prompts to ensure consistent outputs across the dataset
    (Default: calibrate is False)

    num_calibration_docs: Number of docs to use for calibration (default: 10)
    """,
    returns="Subset of documents; each document has same keys as before",
    minimal_example_configuration="""
    name: filter_insightful_comments
    type: filter
    prompt: Is this comment insightful?
    Comment: {{ input.comment }}
    Consider whether the comment adds a new perspective, explains reasoning, or deepens the discussion. Return true if it is insightful, false otherwise.
    output:
        schema:
            is_insightful: boolean
        model: gpt-4o-mini
    """,
)

op_reduce = Operator(
    name="Reduce",
    type_llm_or_not="LLM-powered",
    description="Aggregates multiple documents with the same key value(s) into a single output. Groups documents by reduce_key, then applies an LLM prompt to each group to create one aggregated output per unique key combination.",
    when_to_use="Use when you need to summarize, consolidate, or analyze groups of related documents - like combining all reviews for a product, summarizing feedback by department, or aggregating patient records by ID.",
    required_parameters="""
    name: Unique name for the operation
    type: Must be "reduce"
    model: LLM used to execute the prompts
    reduce_key: Key or keys to group by (can be a string, a list of strings, or "_all" to aggregate over all documents)
    prompt: Jinja2 template that references {{ inputs }} (the list of grouped documents, where each document is a dictionary) and {{ reduce_key }}

    fold_prompt: Template for incrementally processing large groups by folding batches into the current state. The template should reference:
    - {{ inputs }}: The current batch of documents to process
    - {{ output }}: The current aggregated state (matches output.schema)
    The LLM processes the data in batches, updating the state after each batch using the fold_prompt.

    fold_batch_size: Number of documents to process in each fold batch

    output.schema: Schema for the aggregated output
    """,
    optional_parameters="""
    value_sampling: Run the reduce operation only on a sample, to reduce processing cost and time. Specify a method:
    - random: Randomly select a subset of items from each group.
    - first_n: Select the first N items in each group.
    - cluster: Use clustering (e.g., K-means) to select a diverse, representative subset.
    - semantic_similarity: Select items most relevant to a provided query, based on embeddings.

    Optional parameters for value_sampling:
    - enabled: true to activate value sampling (default: false)
    - method: One of the above sampling methods
    - sample_size: Number of items to sample from each group
    - For cluster: no additional parameters required
    - For semantic_similarity:
        - embedding_model: Embedding model to use (e.g., text-embedding-3-small)
        - embedding_keys: List of fields to embed (e.g., [review])
        - query_text: Text to focus sampling (e.g., "battery life and performance")
    """,
    returns="""A set of fewer documents; each document has reduce_keys and all the keys specified in the output schema. Note that many keys in the original documents, if not specified in reduce_keys (i.e., the groupby), will get dropped.""",
    minimal_example_configuration="""
    name: summarize_feedback
    type: reduce
    reduce_key: department
    model: gpt-4o-mini
    prompt: Summarize the customer feedback for {{ reduce_key.department }}:
    {% for item in inputs %}
    Feedback {{ loop.index }}: {{ item.feedback }}
    {% endfor %}
    Provide main points and overall sentiment.

    fold_prompt: Incrementally update the summary and sentiment based on a batch of new feedback for {{ reduce_key.department }}.
    Current summary: {{ output.summary }}
    Current sentiment: {{ output.sentiment }}
    New feedback batch:
    {% for item in inputs %}
    Feedback {{ loop.index }}: {{ item.feedback }}
    {% endfor %}
    Return the updated summary and sentiment after incorporating the new feedback.

    fold_batch_size: 25

    output:
        schema:
            summary: string
            sentiment: string
    """,
)

op_split = Operator(
    name="Split",
    type_llm_or_not="Not LLM-powered",
    description="Divides long text into smaller chunks based on token count or delimiters. Creates multiple output documents from each input, one per chunk. Adds chunk metadata including chunk ID and sequence number.",
    when_to_use="""Use when documents exceed LLM token limits, when processing long transcripts/reports/contracts and we need to read every portion of the document for the task. E.g., "extract all mentions of X from this document." """,
    required_parameters="""
    name: Unique name for the operation
    type: Must be "split"
    split_key: Field containing the text to split
    method: "token_count" or "delimiter" (defaults to "token_count")
    method_kwargs:
    - For "token_count": num_tokens (integer) — Number of tokens per split
    - For "delimiter": delimiter (string) — Delimiter to use for splitting the text
    """,
    optional_parameters=None,
    returns="""
    The Split operation generates multiple output items for each input item. Each output includes:
    - All original key-value pairs from the input item
    - {split_key}_chunk: The content of the split chunk
    - {op_name}_id: A unique identifier for the original document
    - {op_name}_chunk_num: The sequential number of the chunk within its original document
    """,
    minimal_example_configuration="""
    name: split_transcript
    type: split
    split_key: transcript
    method: token_count
    method_kwargs:
        num_tokens: 500
    """,
)

op_gather = Operator(
    name="Gather",
    type_llm_or_not="Not LLM-powered",
    description="""Adds context from surrounding chunks to each chunk after splitting. Includes content from previous/next chunks and maintains document structure through header hierarchies. Creates a "rendered" version of each chunk with its context.""",
    when_to_use="Use after Split when chunks need context for accurate processing - essential for legal documents, technical manuals, or any content where references span chunks. Helps maintain document structure and cross-references.",
    required_parameters="""
    name: Unique name for the operation
    type: Must be "gather"
    content_key: Field containing the chunk content to gather
    doc_id_key: Field linking all chunks from the same original document
    order_key: Field specifying the order/sequence of each chunk within the document

    peripheral_chunks: Configuration for including context from surrounding (previous and/or next) chunks. This helps each chunk retain important context that may be necessary for accurate downstream processing.
    - You can specify previous and/or next context, and control how much content to include before and after each chunk.
    - For each (previous/next), you can define:
        - head: The first chunk(s) in the section (e.g., the chunk immediately before/after the current one)
        - middle: Chunks between head and tail (e.g., summarized versions of further-away chunks)
        - tail: The last chunk(s) in the section (e.g., furthest before/after the current chunk you want to include)
        - For each section, specify:
            - count: Number of chunks to include (head and tail only)
            - content_key: Which field to use for context (e.g., full text or summary)

    Example:
        peripheral_chunks:
        previous:
            head:
            count: 1
            content_key: full_content
            middle:
            content_key: summary_content
            tail:
            count: 2
            content_key: full_content
        next:
            head:
            count: 1
            content_key: full_content

    - This config means:
        • Include the full content of 1 chunk before, summaries of all in-between previous chunks, and the full content of the 2 furthest-back previous chunks.
        • Include the full content of the 1 chunk immediately after the current chunk.
    - Use full content for immediate context (head/tail) and summaries for middle sections to balance completeness and token efficiency.
    - Only include next chunks if future context is important; by default, focus on previous for most text documents.
    """,
    optional_parameters="""
    doc_header_key: (optional) Field containing extracted headers for each chunk.
    - This field provides the hierarchical structure of document sections, enabling the Gather operation to reconstruct header context for each chunk.
    - To use this, you must first run a map operation that extracts headers from each chunk, using the following schema:
        headers: list of {header: string, level: integer}
    - Example map operation:

        name: extract_headers
        type: map
        input:
            - agreement_text_chunk
        prompt: Extract any section headers from the following merger agreement chunk:
        {{ input.agreement_text_chunk }}
        Return the headers as a list, preserving their hierarchy.
        output.schema:
            headers: list[{header: string, level: integer}]
    """,
    returns="""
    The Gather operation produces one output item for each input chunk. Each output includes:

    - All original key-value pairs from the input document
    - {content_key}_rendered: The content of the chunk, enriched with:
        • Reconstructed header hierarchy (if doc_header_key is provided)
        • Previous context (chunks before the current chunk, if configured)
        • The main chunk, clearly marked
        • Next context (chunks after the current chunk, if configured)
        • Indications of skipped content between included sections (e.g., "[... 500 characters skipped ...]")

    For example, if your content_key is agreement_text_chunk, the Gather operation adds:
        agreement_text_chunk_rendered

    Note: No additional unique identifier or chunk number fields are created by Gather. (Those are typically added by Split.) Gather focuses on adding the rendered, context-enriched content field.
    """,
    minimal_example_configuration="""
    name: add_context
    type: gather
    content_key: text_chunk
    doc_id_key: split_doc_id
    order_key: split_chunk_num

    peripheral_chunks:
        previous:
            head:
                count: 1
                content_key: text_chunk
        tail:
            count: 2
            content_key: text_chunk
    """,
)

op_unnest = Operator(
    name="Unnest",
    type_llm_or_not="Not LLM-powered",
    description="Expands arrays into multiple documents (one per element) or flattens dictionary fields into the parent document. For arrays, replaces the array with individual elements. For dicts, adds specified fields to parent while keeping original dict.",
    when_to_use="Use when you need to process array elements individually, or when flattening nested data structures for easier analysis. Essentially, this operation is for normalizing nested data.",
    required_parameters="""
    name: Unique name for the operation
    type: Must be "unnest"
    unnest_key: Field containing the array or dictionary to expand
    """,
    optional_parameters="""
    keep_empty: If true, empty arrays being exploded will be kept in the output (with value None). Default: false
    recursive: If true, the unnest operation will be applied recursively to nested arrays. Default: false
    depth: The maximum depth for recursive unnesting (only applicable if recursive is true)
    """,
    returns="Returns one output document for each element in the unnested array or dictionary. Each output preserves all original fields from the input document, and adds fields from the expanded element. For arrays, each item becomes its own document. For dictionaries, specified expand_fields are added as top-level fields in the output.",
    minimal_example_configuration="""
    name: expand_user
    type: unnest
    unnest_key: user_info
    """,
)

op_sample = Operator(
    name="Sample",
    type_llm_or_not="Not LLM-powered",
    description="Selects a subset of documents from the input according to the specified sampling method. Used to generate a representative sample for further analysis or processing.",
    when_to_use="Use when you want to work with a smaller subset of your data for debugging, rapid prototyping, or to reduce compute cost. Also useful for sampling before downstream processing. Stratification can be applied to uniform, first, outliers, top_embedding, and top_fts methods. It ensures that the sample maintains the distribution of specified key(s) in the data or retrieves top items from each stratum.",
    required_parameters="""
    name: Unique name for the operation
    type: Must be "sample"
    method: The sampling method to use ("uniform", "outliers", "custom", "first", "top_embedding", or "top_fts")
    samples: Either a list of key-value pairs representing document ids and values, an integer count of samples, or a float fraction of samples.
    """,
    optional_parameters="""
    method: The sampling method to use. Options:
    - uniform: Randomly select the specified number or fraction of documents. When combined with stratification, maintains the distribution of the stratified groups.
    - first: Select the first N documents from the dataset. When combined with stratification, takes proportionally from each group.
    - top_embedding: Select top documents based on embedding similarity to a query. Requires the following in method_kwargs: keys: A list of keys to use for creating embeddings, query: The query string to match against (supports Jinja templates), embedding_model: (Optional) The embedding model to use. Defaults to "text-embedding-3-small".
    - top_fts: Retrieves the top N items using full-text search with BM25 algorithm. Requires the following in method_kwargs: keys: A list of keys to search within, query: The query string for keyword matching (supports Jinja templates).
    - outliers: Select or remove documents considered outliers based on embedding distance. Requiresthe following in method_kwargs: embedding_keys (fields to embed), std (standard deviation cutoff) or samples (number/fraction of outlier samples), and keep (true to keep, false to remove outliers; default false). Optionally, method_kwargs.center can specify the center point.
    - custom: Samples specific items by matching key-value pairs. Stratification is not supported with custom sampling.
    
    samples: The number of samples to select (integer), fraction of documents to sample (float), or explicit list of document IDs (for custom).

    random_state: Integer to seed the random generator for reproducible results (default: random each run).

    stratify_key: Field or list of fields to stratify by (for uniform method stratified sampling)
    samples_per_group: When stratifying, sample N items per group vs. dividing total (for uniform method)
    
    method_kwargs: Additional parameters required by the chosen sampling method, such as:
    - embedding_keys: List of fields to embed (for outliers)
    - std: Number of standard deviations for outlier cutoff (for outliers)
    - keep: true to keep or false to remove outliers (for outliers; default false)
    - center: Dictionary specifying the center for distance calculations (for outliers)
    """,
    returns="A subset of input documents, with the same schema as the original input.",
    minimal_example_configuration="""
    name: stratified_sample
    type: sample
    method: uniform
    samples: 0.2
    stratify_key: category
    """,
)

op_resolve = Operator(
    name="Resolve",
    type_llm_or_not="LLM-powered",
    description="Identifies and canonicalizes duplicate or matching entities across your dataset using LLM-driven pairwise comparison and resolution prompts. Useful for data cleaning, deduplication, and standardizing variations created by LLMs in preceding map operations.",
    when_to_use="Use when you need to standardize documents that may refer to the same real-world entity but have inconsistent or duplicated fields (e.g., names, product titles, organizations) due to extraction, human error, or LLM variation.",
    required_parameters="""
    name: Unique name for the operation
    type: Must be "resolve"
    comparison_prompt: Jinja2 template for comparing two candidate records (refer to as {{ input1 }}, {{ input2 }})
    resolution_prompt: Jinja2 template for consolidating/mapping a set of matched records (refer to as {{ inputs }})
    output.schema: Dictionary defining the structure and types of the resolved output
    embedding_model: Model to use for creating embeddings for blocking (default: falls back to default_model)
    comparison_model: LLM to use for comparisons (default: falls back to default_model)
    resolution_model: LLM to use for final resolution (default: falls back to default_model)
    """,
    optional_parameters="""
    blocking_keys: List of fields for blocking—records must match on at least one key to be compared (default: all input keys)
    blocking_threshold: Embedding similarity threshold for blocking (only compare above this value)
    blocking_conditions: List of Python expressions for custom blocking logic (e.g., "left['ssn'][-4:] == right['ssn'][-4:]")
    embedding_batch_size: Number of records sent to embedding model per batch (default: 1000)
    compare_batch_size: Number of record pairs compared per batch (default: 500)
    limit_comparisons: Maximum number of pairwise comparisons (default: no limit)
    """,
    returns="One output document per input document, preserving the original document structure, but with specified fields in the output schema updated to their resolved (standardized) values. All other fields remain unchanged.",
    minimal_example_configuration="""
    name: standardize_patient_names
    type: resolve
    comparison_model: gpt-4o-mini
    resolution_model: gpt-4o
    embedding_model: text-embedding-3-small
    comparison_prompt: |
    Compare the following two patient name entries:
    Patient 1: {{ input1.patient_name }}
    Date of Birth 1: {{ input1.date_of_birth }}
    Patient 2: {{ input2.patient_name }}
    Date of Birth 2: {{ input2.date_of_birth }}
    Are these entries likely referring to the same patient? Respond "True" or "False".
    resolution_prompt: |
    Standardize these patient names into a single, consistent format:
    {% for entry in inputs %}
    Patient Name {{ loop.index }}: {{ entry.patient_name }}
    {% endfor %}
    Provide a single, standardized patient name.
    output:
    schema:
        patient_name: string
    blocking_keys:
    - last_name
    - date_of_birth
    blocking_threshold: 0.8
    blocking_conditions:
    - "left['last_name'][:2].lower() == right['last_name'][:2].lower()"
    - "left['date_of_birth'] == right['date_of_birth']"
    """,
)

op_code_map = Operator(
    name="Code Map",
    type_llm_or_not="not LLM-powered",
    description="Applies a Python function to each document independently using custom code. Returns a dictionary of key-value pairs to UPDATE the original document with. Useful for deterministic transformations, regex processing, calculations, or leveraging external Python libraries.",
    when_to_use="Use when you need deterministic processing, complex calculations, regex/pattern matching, or want to leverage existing Python libraries. Ideal for structured data transformations that don't require LLM reasoning.",
    required_parameters="""
    name: Unique name for the operation
    type: Must be "code_map"
    code: Python code defining a function named 'transform' that takes an input document and returns a dictionary of updates. Must include all necessary imports within the function. Format: def transform(input_doc): ...""",
    optional_parameters="""
    drop_keys: List of keys to remove from output (default: None)
    concurrent_thread_count: Number of threads to use (default: number of logical CPU cores)""",
    returns="Each original document, updated with the key-value pairs returned by the transform function",
    minimal_example_configuration="""
    name: extract_keywords_deterministic
    type: code_map
    code: |
        def transform(input_doc):
            import re
            text = input_doc.get('content', '')
            # Extract words that are all caps (potential keywords)
            keywords = re.findall(r'\\b[A-Z]{2,}\\b', text)
            # Extract email addresses
            emails = re.findall(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b', text)
            return {
                'keywords': list(set(keywords)),
                'email_count': len(emails),
                'word_count': len(text.split())
            }
    """,
)

op_code_filter = Operator(
    name="Code Filter",
    type_llm_or_not="not LLM-powered",
    description="Filters documents based on custom Python logic. Uses a Python function that returns True to keep documents and False to filter them out. Useful for deterministic filtering based on calculations, regex patterns, or structured data conditions.",
    when_to_use="Use when you need deterministic filtering logic that doesn't require LLM reasoning - like filtering based on numeric thresholds, text patterns, data completeness, or complex boolean conditions.",
    required_parameters="""
    name: Unique name for the operation
    type: Must be "code_filter"
    code: Python code defining a function named 'transform' that takes an input document and returns a boolean (True to keep, False to filter out). Must include all necessary imports within the function. Format: def transform(input_doc): ...""",
    optional_parameters="""
    concurrent_thread_count: Number of threads to use (default: number of logical CPU cores)""",
    returns="Subset of input documents where the transform function returned True. Documents retain all original fields.",
    minimal_example_configuration="""
    name: filter_valid_scores
    type: code_filter
    code: |
        def transform(input_doc):
            score = input_doc.get('confidence_score', 0)
            text_length = len(input_doc.get('content', ''))
            # Keep documents with high confidence and sufficient content
            return score >= 0.8 and text_length >= 100
    """,
)

op_topk = Operator(
    name="TopK",
    type_llm_or_not="LLM-powered or not LLM-powered",
    description="Retrieves the most relevant items from your dataset using semantic similarity, full-text search, or LLM-based comparison. Provides a specialized interface for retrieval tasks where you need to find and rank the best matching documents based on specific criteria.",
    when_to_use="Use when you need to find the most relevant documents for a query, filter large datasets to the most important items, implement retrieval-augmented generation (RAG) pipelines, or build recommendation systems. Choose this over general sampling when you specifically need the 'best' matches according to some criteria.",
    required_parameters="""
    name: Unique name for the operation
    type: Must be "topk"
    method: Retrieval method to use ("embedding" for semantic similarity, "fts" for full-text search, or "llm_compare" for LLM-based ranking)
    k: Number of items to retrieve (integer) or percentage (float between 0 and 1)
    keys: List of document fields to use for matching/comparison
    query: Query or ranking criteria (Jinja templates supported for embedding and fts methods only)
    """,
    optional_parameters="""
    embedding_model: Model for embeddings (default: "text-embedding-3-small"). Used for embedding and llm_compare methods.

    model: LLM model for comparisons (required for llm_compare method)

    batch_size: Batch size for LLM ranking (default: 10, only for llm_compare method)

    stratify_key: Key(s) for stratified retrieval - ensures you retrieve top items from each group (string or list of strings). Not supported with llm_compare method.

    Method-specific notes:
    - embedding: Uses semantic similarity via embeddings. Supports Jinja templates in query.
    - fts: Uses BM25 full-text search algorithm. No API costs. Supports Jinja templates in query.
    - llm_compare: Uses LLM for complex ranking based on multiple criteria. Most expensive but most flexible. Does NOT support Jinja templates in query (ranking criteria must be consistent across all documents).
    """,
    returns="Top k documents based on the specified method and query, with the same schema as the original input",
    minimal_example_configuration="""
    name: find_relevant_tickets
    type: topk
    method: embedding
    k: 5
    keys:
        - subject
        - description
        - customer_feedback
    query: "payment processing errors with international transactions"
    embedding_model: text-embedding-3-small
    """,
)


# List of all operators
ALL_OPERATORS = [
    op_map,
    op_extract,
    op_parallel_map,
    op_filter,
    op_reduce,
    op_split,
    op_gather,
    op_unnest,
    op_sample,
    op_resolve,
    op_code_map,
    op_code_filter,
    op_topk,
]


def get_all_operator_descriptions() -> str:
    """
    Generate a comprehensive string containing all operator descriptions.
    This is useful for providing context about available operators in prompts.

    Returns:
        str: Formatted string containing all operator descriptions
    """
    descriptions = []
    descriptions.append("# Available DocETL Operators\n")
    descriptions.append(
        "Below are all the operators available in the DocETL pipeline:\n"
    )

    for op in ALL_OPERATORS:
        descriptions.append(op.to_string())

    return "\n".join(descriptions)
