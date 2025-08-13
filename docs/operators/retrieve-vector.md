# Retrieve Vector Operation

The Retrieve Vector operation uses LanceDB to perform vector-based similarity search on documents. It embeds documents and retrieves the most similar items based on embedding distance.

## Overview

This operation is useful for:
- Semantic search - finding documents similar in meaning to a query
- Context retrieval for RAG (Retrieval-Augmented Generation) applications
- Finding related documents based on content similarity
- Building recommendation systems

## Required Parameters

- `type`: Must be set to "retrieve_vector"
- `query`: The query text to search for similar documents
- `embedding_keys`: List of document fields to embed and search

## Optional Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `embedding_model` | The embedding model to use | `"text-embedding-3-small"` |
| `num_chunks` | Number of top similar documents to retrieve (like top-k) | `10` |
| `table_name` | Name of the LanceDB table to use | `"docetl_vectors"` |
| `db_path` | Path to the LanceDB database | `"./.lancedb"` |
| `persist` | Whether to persist the vector database between runs | `false` |
| `distance_metric` | Distance metric for similarity search | `"cosine"` |
| `output_key` | Key to store retrieved documents in the output | `"_retrieved"` |

## How It Works

1. **Document Embedding**: The operation embeds all input documents using the specified embedding model and keys
2. **Index Creation**: Creates or updates a LanceDB vector index with the embeddings
3. **Query Embedding**: Embeds the query text using the same model
4. **Similarity Search**: Finds the top-k most similar documents based on vector distance
5. **Result Augmentation**: Each input document is augmented with the retrieved similar documents

## Examples

### Basic Semantic Search

```yaml
- name: find_similar_papers
  type: retrieve_vector
  query: "machine learning applications in healthcare"
  embedding_keys:
    - title
    - abstract
  num_chunks: 5
```

### RAG Context Retrieval

```yaml
- name: retrieve_context
  type: retrieve_vector
  query: "How does transformer architecture work?"
  embedding_model: "text-embedding-ada-002"
  embedding_keys:
    - content
  num_chunks: 10
  table_name: "knowledge_base"
  persist: true  # Keep the index for future queries
```

### Product Recommendation

```yaml
- name: find_similar_products
  type: retrieve_vector
  query: "comfortable running shoes for marathon training"
  embedding_keys:
    - name
    - description
    - features
  num_chunks: 20
  output_key: "recommended_products"
```

## Output Format

Each document in the output will contain:
- All original fields from the input document
- A new field (default `_retrieved`) containing an array of retrieved documents
- Each retrieved document includes a `_distance` field indicating similarity

Example output structure:
```json
{
  "id": 1,
  "title": "Original Document",
  "_retrieved": [
    {
      "id": 42,
      "title": "Similar Document 1",
      "content": "...",
      "_distance": 0.123
    },
    {
      "id": 87,
      "title": "Similar Document 2",
      "content": "...",
      "_distance": 0.156
    }
  ]
}
```

## Best Practices

1. **Choose Appropriate Embedding Keys**: Select fields that contain the most semantic information
2. **Model Selection**: Use models optimized for your domain (e.g., scientific embeddings for research papers)
3. **Persistence**: Enable `persist: true` when building a reusable knowledge base
4. **Chunk Size**: Balance between relevance and coverage - more chunks provide more context but may include less relevant items
5. **Query Crafting**: Write queries that capture the semantic intent of what you're searching for

## Performance Considerations

- **Embedding Cost**: Each document and query requires an API call to generate embeddings
- **Index Size**: Large document collections will create larger indexes
- **Memory Usage**: LanceDB loads indexes into memory for fast search
- **Batch Processing**: The operation embeds all documents in batches for efficiency

## Integration with Other Operations

Retrieve Vector works well with:
- **Map**: Process retrieved context with LLMs
- **Filter**: Pre-filter documents before retrieval
- **Reduce**: Aggregate information from retrieved documents
- **Rank**: Re-rank retrieved results based on additional criteria