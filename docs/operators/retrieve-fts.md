# Retrieve FTS Operation

The Retrieve FTS (Full-Text Search) operation uses LanceDB to perform semantic full-text search on documents. It can use either pure text search or combine it with embeddings for enhanced semantic search capabilities.

## Overview

This operation is ideal for:
- Full-text search with semantic understanding
- Keyword-based search with relevance ranking
- Hybrid search combining text matching and semantic similarity
- Information retrieval from large document collections

## Required Parameters

- `type`: Must be set to "retrieve_fts"
- `query`: The search query text
- `embedding_keys`: List of document fields to index and search

## Optional Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `embedding_model` | The embedding model for semantic search | `"text-embedding-3-small"` |
| `num_chunks` | Number of top results to retrieve | `10` |
| `table_name` | Name of the LanceDB table | `"docetl_fts"` |
| `db_path` | Path to the LanceDB database | `"./.lancedb"` |
| `persist` | Whether to persist the database between runs | `false` |
| `rerank` | Whether to use embeddings for semantic reranking | `true` |
| `output_key` | Key to store retrieved documents | `"_retrieved"` |

## How It Works

1. **Document Indexing**: Indexes documents based on specified fields
2. **Optional Embedding**: If `rerank` is true, generates embeddings for semantic search
3. **Query Processing**: Processes the search query
4. **Search Execution**: 
   - With `rerank=true`: Performs semantic search using embeddings
   - With `rerank=false`: Performs keyword-based text search
5. **Result Compilation**: Returns top matching documents with relevance scores

## Examples

### Basic Full-Text Search

```yaml
- name: search_documents
  type: retrieve_fts
  query: "climate change impacts"
  embedding_keys:
    - title
    - content
  rerank: false  # Pure text search
```

### Semantic Search with Reranking

```yaml
- name: semantic_search
  type: retrieve_fts
  query: "What are the benefits of renewable energy?"
  embedding_keys:
    - title
    - abstract
    - content
  rerank: true  # Use embeddings for better relevance
  num_chunks: 15
```

### Knowledge Base Search

```yaml
- name: search_knowledge_base
  type: retrieve_fts
  query: "troubleshooting network connectivity issues"
  embedding_model: "text-embedding-ada-002"
  embedding_keys:
    - question
    - answer
    - tags
  table_name: "support_kb"
  persist: true
  output_key: "relevant_articles"
```

### Research Paper Search

```yaml
- name: find_papers
  type: retrieve_fts
  query: "neural architecture search automated machine learning"
  embedding_keys:
    - title
    - abstract
    - keywords
  num_chunks: 25
  rerank: true
```

## Output Format

Each document in the output contains:
- All original fields from the input document
- A new field (default `_retrieved`) with an array of retrieved documents
- Each retrieved document includes either:
  - `_distance`: When using embeddings (lower is more similar)
  - `_score`: When using text search (higher is more relevant)

Example output:
```json
{
  "id": 1,
  "title": "Original Document",
  "_retrieved": [
    {
      "id": 99,
      "title": "Highly Relevant Result",
      "content": "...",
      "_distance": 0.089
    },
    {
      "id": 156,
      "title": "Related Result",
      "content": "...",
      "_distance": 0.134
    }
  ]
}
```

## Search Modes

### Pure Text Search (`rerank: false`)
- Faster performance
- Good for exact keyword matching
- Lower computational cost
- Suitable for well-structured queries

### Semantic Search (`rerank: true`)
- Better understanding of query intent
- Finds conceptually similar documents
- Higher quality results for natural language queries
- Requires embedding generation (higher cost)

## Best Practices

1. **Field Selection**: Include fields with rich textual content in `embedding_keys`
2. **Query Formulation**: 
   - For text search: Use specific keywords
   - For semantic search: Use natural language questions
3. **Reranking Decision**: Enable reranking for queries requiring semantic understanding
4. **Performance Tuning**: 
   - Use `persist: true` for frequently accessed data
   - Adjust `num_chunks` based on downstream needs
5. **Cost Management**: Disable reranking for simple keyword searches to save on embedding costs

## Performance Considerations

- **Text Search**: Fast, minimal computational overhead
- **Semantic Search**: Requires embedding generation for documents and queries
- **Index Persistence**: Persistent indexes improve performance for repeated searches
- **Memory Usage**: Larger document collections require more memory
- **Batch Efficiency**: Documents are processed in batches for optimal performance

## Comparison with Retrieve Vector

| Feature | Retrieve FTS | Retrieve Vector |
|---------|--------------|-----------------|
| Search Type | Text + Optional Semantic | Pure Semantic |
| Best For | Keyword queries, mixed search | Similarity search |
| Query Format | Keywords or natural language | Natural language |
| Performance | Faster without reranking | Consistent speed |
| Flexibility | Can disable embeddings | Always uses embeddings |

## Integration Examples

### With Map Operation
```yaml
- name: search_and_summarize
  type: retrieve_fts
  query: "machine learning best practices"
  embedding_keys: ["content"]
  num_chunks: 5

- name: summarize_results
  type: map
  prompt: "Summarize these machine learning best practices: {{ _retrieved }}"
```

### With Filter Operation
```yaml
- name: filter_recent
  type: filter
  condition: "date > '2023-01-01'"

- name: search_recent_docs
  type: retrieve_fts
  query: "artificial intelligence trends"
  embedding_keys: ["title", "content"]
```