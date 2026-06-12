# TopK Operation

The TopK operation retrieves the most relevant items from your dataset. Use it for retrieval and ranking: finding documents for a query, filtering datasets to the most important items, RAG pipelines, or recommendations.

```mermaid
flowchart LR
    in["all docs"] --> t["score against query"] --> out["top k docs"]
```
Three retrieval methods are supported:

- **embedding**: semantic similarity — use when meaning matters more than exact words
- **fts**: keyword-based retrieval with BM25 — use when specific terms are important
- **llm_compare**: an LLM ranks documents — use for criteria that require reasoning or multi-factor comparison

## Configuration

### Core Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `method` | `"embedding"` \| `"fts"` \| `"llm_compare"` | Retrieval method to use | Yes |
| `k` | `int` or `float` | Number of items to retrieve (float = percentage) | Yes |
| `keys` | `list[str]` | Document fields to use for matching/comparison | Yes |
| `query` | `str` | Query or ranking criteria (Jinja templates supported for `embedding` and `fts` only) | Yes |

### Method-Specific Parameters

| Parameter | Type | Methods | Description | Default |
|-----------|------|---------|-------------|---------|
| `embedding_model` | `str` | `embedding`, `llm_compare` | Model for embeddings | `"text-embedding-3-small"` |
| `model` | `str` | `llm_compare` | LLM model for comparisons | Required for `llm_compare` |
| `batch_size` | `int` | `llm_compare` | Batch size for LLM ranking | `10` |
| `stratify_key` | `str` or `list[str]` | `embedding`, `fts` | Keys for stratified retrieval | `None` |

!!! note "Python API"

    TopK has no dedicated Frame method, so in Python you construct the pipeline as a config dict and run it with `DSLRunner`. The first example below shows the full pattern; subsequent examples show the operation config dict to place in `config["operations"]`.

## Examples

### Semantic Search with Embeddings

Find support tickets semantically similar to payment processing issues:

=== "YAML"

    ```yaml
    - name: find_relevant_tickets
      type: topk
      method: embedding
      k: 5
      keys: 
        - subject
        - description
        - customer_feedback
      query: "payment processing errors with international transactions"
      embedding_model: text-embedding-3-small
    ```

=== "Python"

    ```python
    from docetl.runner import DSLRunner

    config = {
        "default_model": "gpt-4o-mini",
        "datasets": {
            "tickets": {"type": "file", "path": "tickets.json"},
        },
        "operations": [
            {
                "name": "find_relevant_tickets",
                "type": "topk",
                "method": "embedding",
                "k": 5,
                "keys": ["subject", "description", "customer_feedback"],
                "query": "payment processing errors with international transactions",
                "embedding_model": "text-embedding-3-small",
            }
        ],
        "pipeline": {
            "steps": [
                {
                    "name": "retrieve",
                    "input": "tickets",
                    "operations": ["find_relevant_tickets"],
                }
            ],
            "output": {"type": "file", "path": "out.json"},
        },
    }
    runner = DSLRunner(config)
    results, _ = runner.run()
    ```

### Keyword Search with FTS

Keyword matching with no API costs:

=== "YAML"

    ```yaml
    - name: search_products
      type: topk
      method: fts
      k: 20
      keys:
        - product_name
        - description
        - category
        - tags
      query: "wireless noise cancelling headphones bluetooth"
    ```

=== "Python"

    ```python
    # Add to config["operations"] and reference in a pipeline step
    {
        "name": "search_products",
        "type": "topk",
        "method": "fts",
        "k": 20,
        "keys": ["product_name", "description", "category", "tags"],
        "query": "wireless noise cancelling headphones bluetooth",
    }
    ```

### Complex Ranking with LLM Compare

Rank items by multi-factor or subjective criteria. This method requires consistent criteria across all documents and doesn't support Jinja templates:

=== "YAML"

    ```yaml
    - name: screen_resumes
      type: topk
      method: llm_compare
      k: 10
      keys:
        - skills
        - experience
        - education
      query: |
        Rank candidates based on their fit for a Senior Backend Engineer role requiring:
        - 5+ years Python experience
        - Distributed systems expertise
        - Strong knowledge of PostgreSQL and Redis
        - Experience with microservices architecture
        - Leadership experience is a plus
        
        Prioritize hands-on technical experience over academic credentials.
      model: gpt-4o
      batch_size: 5
    ```

=== "Python"

    ```python
    # Add to config["operations"] and reference in a pipeline step
    {
        "name": "screen_resumes",
        "type": "topk",
        "method": "llm_compare",
        "k": 10,
        "keys": ["skills", "experience", "education"],
        "query": """Rank candidates based on their fit for a Senior Backend Engineer role requiring:
    - 5+ years Python experience
    - Distributed systems expertise
    - Strong knowledge of PostgreSQL and Redis
    - Experience with microservices architecture
    - Leadership experience is a plus

    Prioritize hands-on technical experience over academic credentials.""",
        "model": "gpt-4o",
        "batch_size": 5,
    }
    ```

### Dynamic Queries with Templates

The embedding and FTS methods support Jinja templates for queries that adapt based on input data:

=== "YAML"

    ```yaml
    - name: personalized_search
      type: topk
      method: embedding
      k: 10
      keys:
        - content
        - tags
      query: |
        {{ input.user_preferences }} 
        Focus on {{ input.topic_of_interest }}
        Exclude anything related to {{ input.blocked_topics }}
    ```

=== "Python"

    ```python
    # Add to config["operations"] and reference in a pipeline step
    {
        "name": "personalized_search",
        "type": "topk",
        "method": "embedding",
        "k": 10,
        "keys": ["content", "tags"],
        "query": """{{ input.user_preferences }}
    Focus on {{ input.topic_of_interest }}
    Exclude anything related to {{ input.blocked_topics }}""",
    }
    ```

### Stratified Retrieval

The embedding and FTS methods support stratification, retrieving the top items from each group:

=== "YAML"

    ```yaml
    - name: recommendations_by_category
      type: topk
      method: fts
      k: 3  # Get top 3 from each category
      keys:
        - product_name
        - description
      query: "premium quality bestseller"
      stratify_key: category
    ```

=== "Python"

    ```python
    # Add to config["operations"] and reference in a pipeline step
    {
        "name": "recommendations_by_category",
        "type": "topk",
        "method": "fts",
        "k": 3,  # Get top 3 from each category
        "keys": ["product_name", "description"],
        "query": "premium quality bestseller",
        "stratify_key": "category",
    }
    ```

## Common Patterns

### Single-Document RAG Pipeline

Retrieve the most relevant chunks, then synthesize them into an answer with reduce:

=== "YAML"

    ```yaml
    # Step 1: Retrieve most relevant document chunks
    - name: retrieve_context
      type: topk
      method: embedding
      k: 5
      keys: [content]
      query: "{{ input.user_question }}"

    # Step 2: Generate comprehensive answer from all retrieved chunks
    - name: generate_answer
      type: reduce
      reduce_key: user_question  # Group by the question
      prompt: |
        Based on the following document excerpts, provide a comprehensive answer to the question.
        
        Question: {{ inputs[0].user_question }}
        
        Retrieved context from document:
        {% for chunk in inputs %}
        - {{ chunk.content }}
        {% endfor %}
        
        Synthesize the information from all excerpts into a single, coherent answer.
      output_schema:
        answer: string
    ```

=== "Python"

    ```python
    from docetl.runner import DSLRunner

    config = {
        "default_model": "gpt-4o-mini",
        "datasets": {
            "chunks": {"type": "file", "path": "chunks.json"},
        },
        "operations": [
            # Step 1: Retrieve most relevant document chunks
            {
                "name": "retrieve_context",
                "type": "topk",
                "method": "embedding",
                "k": 5,
                "keys": ["content"],
                "query": "{{ input.user_question }}",
            },
            # Step 2: Generate comprehensive answer from all retrieved chunks
            {
                "name": "generate_answer",
                "type": "reduce",
                "reduce_key": "user_question",  # Group by the question
                "prompt": """Based on the following document excerpts, provide a comprehensive answer to the question.

    Question: {{ inputs[0].user_question }}

    Retrieved context from document:
    {% for chunk in inputs %}
    - {{ chunk.content }}
    {% endfor %}

    Synthesize the information from all excerpts into a single, coherent answer.""",
                "output_schema": {"answer": "string"},
            },
        ],
        "pipeline": {
            "steps": [
                {
                    "name": "rag",
                    "input": "chunks",
                    "operations": ["retrieve_context", "generate_answer"],
                }
            ],
            "output": {"type": "file", "path": "answers.json"},
        },
    }
    runner = DSLRunner(config)
    results, _ = runner.run()
    ```

### Multi-Stage Filtering

Combine multiple TopK operations with different methods, progressively refining results:

=== "YAML"

    ```yaml
    # Cast a wide net with keyword search
    - name: initial_search
      type: topk
      method: fts
      k: 100
      keys: [title, content]
      query: "machine learning"

    # Refine with semantic search
    - name: refine_results
      type: topk
      method: embedding
      k: 20
      keys: [title, content]
      query: "practical applications of deep learning in healthcare"

    # Final ranking with LLM
    - name: final_ranking
      type: topk
      method: llm_compare
      k: 5
      keys: [title, abstract, impact_factor]
      query: "Rank by potential clinical impact and implementation feasibility"
      model: gpt-4o
    ```

=== "Python"

    ```python
    # Add to config["operations"] and reference in a pipeline step
    [
        # Cast a wide net with keyword search
        {
            "name": "initial_search",
            "type": "topk",
            "method": "fts",
            "k": 100,
            "keys": ["title", "content"],
            "query": "machine learning",
        },
        # Refine with semantic search
        {
            "name": "refine_results",
            "type": "topk",
            "method": "embedding",
            "k": 20,
            "keys": ["title", "content"],
            "query": "practical applications of deep learning in healthcare",
        },
        # Final ranking with LLM
        {
            "name": "final_ranking",
            "type": "topk",
            "method": "llm_compare",
            "k": 5,
            "keys": ["title", "abstract", "impact_factor"],
            "query": "Rank by potential clinical impact and implementation feasibility",
            "model": "gpt-4o",
        },
    ]
    ```

## Performance Considerations

- **fts**: fastest, no API costs (local BM25 scoring)
- **embedding**: API calls to embed documents and queries; similarity matching is fast once embeddings are computed
- **llm_compare**: highest cost and slowest, due to multiple LLM calls

To reduce cost: preprocess embeddings offline, use FTS for initial filtering before more expensive methods, and tune `batch_size` for llm_compare.

## Implementation Details

### Embedding Method

Embeds documents and the query with the specified embedding model (default text-embedding-3-small), then returns the k documents with highest cosine similarity to the query. Text normalization and truncation are handled automatically; with stratification, the process runs independently per stratum.

### FTS Method

Uses the BM25 ranking algorithm (via the rank-bm25 library). Documents are tokenized and lowercase-normalized with special characters removed. BM25 weighs term frequency (with saturation), inverse document frequency, and document length normalization.

### LLM Compare Method

Delegates to the [rank operation](rank.md): an initial embedding-based ordering, then sliding windows of documents compared by the LLM in batches (controlled by `batch_size`). Jinja templates are not supported because the LLM must compare all documents using the same criteria.

## Error Handling

- If k exceeds the number of available documents, all available items are returned.
- If specified keys are missing from some documents, whatever fields are available are used.
- FTS handles documents with empty text after normalization; embedding API failures are retried with exponential backoff.
