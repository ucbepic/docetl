# Sample Operation

The Sample operation in DocETL selects a subset of items from the input data using various sampling methods. While it can be used as a debugging tool to limit data during pipeline development, it also serves as a powerful data selection mechanism for production pipelines.

## Overview

The Sample operation supports seven distinct sampling methods:

1. **Uniform**: Random sampling with equal probability for each item
2. **Stratify**: Maintains proportional representation of groups
3. **Outliers**: Identifies items based on embedding distance from a center
4. **Custom**: Selects specific items by matching key values
5. **First**: Takes the first N items (useful for debugging)
6. **Retrieve Vector**: Vector-based similarity search using LanceDB
7. **Retrieve FTS**: Full-text search with optional semantic reranking using LanceDB

## Required Parameters

- `name`: A unique name for the operation
- `type`: Must be set to "sample"
- `method`: The sampling method to use (`uniform`, `stratify`, `outliers`, `custom`, `first`, `retrieve_vector`, or `retrieve_fts`)
- `samples`: The number or fraction of samples to select (format depends on method)

## Optional Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `random_state` | Integer seed for reproducible random sampling | Uses global random state |
| `method_kwargs` | Additional parameters specific to the sampling method | `{}` |

## Sampling Methods in Detail

### Uniform Sampling

Uniform sampling randomly selects items with equal probability. This is the simplest and most common sampling method.

**How it works**: Each item has an equal chance of being selected. The selection is random but can be made reproducible with `random_state`.

**Parameters**:
- `samples`: Can be:
  - An integer (e.g., `100`) to select exactly that many items
  - A float between 0 and 1 (e.g., `0.1`) to select that fraction of items

**Example**:
```yaml
- name: uniform_sample
  type: sample
  method: uniform
  samples: 100  # Select 100 random items
  random_state: 42  # For reproducibility
```

### Stratified Sampling

Stratified sampling ensures proportional representation of different groups in your sample. This is crucial when you want to maintain the distribution of categories from your original data.

**How it works**: The data is divided into groups (strata) based on a specified key. The sampling then selects from each group proportionally to maintain the same distribution as the original data.

**Parameters**:
- `samples`: Same as uniform sampling
- `method_kwargs`:
  - `stratify_key`: The key to group by (can be a string or list of strings for compound stratification)
  - `samples_per_group`: If true, sample N items from each group instead of dividing total samples across groups

**Example - Single Key**:
```yaml
- name: stratified_sample
  type: sample
  method: stratify
  samples: 0.2  # Select 20% of data
  method_kwargs:
    stratify_key: category  # Maintain category proportions
  random_state: 42
```

**Example - Compound Keys**:
```yaml
- name: compound_stratified_sample
  type: sample
  method: stratify
  samples: 100
  method_kwargs:
    stratify_key: [category, region]  # Stratify by combination of category AND region
    samples_per_group: true  # Get 100 samples from each category-region combination
```

If your data has 70% "A" category and 30% "B" category, the sample will maintain this 70/30 ratio (unless using `samples_per_group`).

### Outlier Sampling

Outlier sampling uses embeddings to identify items that are either far from or close to a center point in the embedding space. This is powerful for finding unusual items or filtering to similar items.

**How it works**: 
1. Generates embeddings for specified fields in each document
2. Calculates distances from a center point (either computed or specified)
3. Selects items based on distance threshold

**Parameters**:
- `method_kwargs`:
  - `embedding_keys`: List of document fields to embed (required)
  - `embedding_model`: Model to use for embeddings (optional, uses default)
  - Either:
    - `std`: Number of standard deviations for threshold
    - `samples`: Number/fraction of items to consider as outliers
  - `keep`: Whether to keep outliers (`true`) or inliers (`false`, default)
  - `center`: Optional dictionary specifying the center point

**Example - Remove outliers**:
```yaml
- name: remove_outliers
  type: sample
  method: outliers
  method_kwargs:
    embedding_keys:
      - title
      - description
    std: 2  # Remove items > 2 standard deviations from center
    keep: false  # Keep only inliers
```

**Example - Find similar items**:
```yaml
- name: find_similar
  type: sample
  method: outliers
  method_kwargs:
    embedding_keys:
      - content
    center:
      content: "Machine learning applications in healthcare"
    samples: 50  # Keep 50 closest items to center
    keep: false  # Keep inliers (close to center)
```

### Custom Sampling

Custom sampling allows you to specify exactly which items to select by providing identifying key-value pairs.

**How it works**: Matches documents based on the keys and values you provide. All specified keys must match for a document to be selected.

**Parameters**:
- `samples`: A list of dictionaries, each containing key-value pairs to match

**Example**:
```yaml
- name: custom_sample
  type: sample
  method: custom
  samples:
    - id: 1
      type: article
    - id: 5
      type: paper
    - id: 10
      type: article
```

This will select documents where `id=1 AND type=article`, `id=5 AND type=paper`, or `id=10 AND type=article`.

### First Sampling

First sampling simply takes the first N items from the input. This is primarily useful for debugging and development.

**How it works**: Returns the first N items in the order they appear in the input.

**Parameters**:
- `samples`: Number of items to take from the beginning

**Example**:
```yaml
- name: debug_sample
  type: sample
  method: first
  samples: 10  # Take first 10 items
```

### Retrieve Vector

Retrieve Vector performs vector-based similarity search using LanceDB. It embeds documents and retrieves the most similar items based on embedding distance.

**Note**: This method requires LanceDB. If not already installed, run: `uv add lancedb` or `pip install lancedb`

**How it works**:
1. If `stratify_key` is specified, searches are performed within each stratum separately
2. Embeds all documents in each stratum using specified fields
3. Creates or updates a LanceDB vector index for each stratum
4. Embeds the query text
5. Finds the top-k most similar documents within each stratum
6. Each document receives the retrieved results from its own stratum

**Parameters in method_kwargs**:
- `query`: The query text to search for similar documents (required)
- `embedding_keys`: List of document fields to embed and search (required)
- `num_chunks`: Number of top similar documents to retrieve (default: 10)
- `embedding_model`: The embedding model to use (default: "text-embedding-3-small")
- `table_name`: Name of the LanceDB table (default: "docetl_vectors")
- `db_path`: Path to the LanceDB database (default: `~/.cache/docetl/lancedb/{operation_name}`)
- `persist`: Whether to persist the vector database between runs (default: false)
- `output_key`: Key to store retrieved documents in the output (default: "_retrieved")
- `stratify_key`: Optional key(s) to stratify searches by (can be string or list of strings)

**Example - Basic Vector Search**:
```yaml
- name: find_similar_papers
  type: sample
  method: retrieve_vector
  method_kwargs:
    query: "machine learning applications in healthcare"
    embedding_keys:
      - title
      - abstract
    num_chunks: 5
    output_key: "similar_papers"
```

**Example - Stratified Vector Search**:
```yaml
- name: find_similar_by_category
  type: sample
  method: retrieve_vector
  method_kwargs:
    query: "deep learning techniques"
    embedding_keys:
      - content
    stratify_key: category  # Search within each category separately
    num_chunks: 3
```

### Retrieve FTS

Retrieve FTS performs full-text search with support for pure text, vector, or hybrid search using LanceDB.

**Note**: This method requires LanceDB. If not already installed, run: `uv add lancedb` or `pip install lancedb`

**How it works**:
1. If `stratify_key` is specified, searches are performed within each stratum separately
2. Indexes documents based on specified fields
3. For vector or hybrid search, generates embeddings
4. Performs the specified type of search (FTS, vector, or hybrid)
5. Returns top matching documents with relevance scores

**Parameters in method_kwargs**:
- `query`: The search query text (required)
- `embedding_keys`: List of document fields to index and search (required)
- `num_chunks`: Number of top results to retrieve (default: 10)
- `embedding_model`: The embedding model for vector/hybrid search (default: "text-embedding-3-small")
- `table_name`: Name of the LanceDB table (default: "docetl_fts")
- `db_path`: Path to the LanceDB database (default: `~/.cache/docetl/lancedb/{operation_name}`)
- `persist`: Whether to persist the database between runs (default: false)
- `query_type`: Type of search - "fts", "vector", or "hybrid" (default: "hybrid")
- `output_key`: Key to store retrieved documents (default: "_retrieved")
- `stratify_key`: Optional key(s) to stratify searches by (can be string or list of strings)

**Example - Hybrid Search**:
```yaml
- name: hybrid_search
  type: sample
  method: retrieve_fts
  method_kwargs:
    query: "renewable energy benefits"
    embedding_keys:
      - title
      - content
    query_type: "hybrid"  # Combines FTS and vector search
    num_chunks: 15
```

**Example - Pure Text Search with Stratification**:
```yaml
- name: keyword_search_by_region
  type: sample
  method: retrieve_fts
  method_kwargs:
    query: "climate change impacts"
    embedding_keys:
      - title
      - content
    query_type: "fts"  # Pure keyword-based search
    stratify_key: [region, year]  # Search within each region-year combination
    num_chunks: 10
```

**Example - Vector Search**:
```yaml
- name: semantic_search
  type: sample
  method: retrieve_fts
  method_kwargs:
    query: "What are the effects of deforestation?"
    embedding_keys:
      - abstract
    query_type: "vector"  # Pure semantic search
    num_chunks: 20
```

## Use Cases

### 1. Debugging and Development
```yaml
# Limit data for faster iteration during development
- name: dev_sample
  type: sample
  method: uniform
  samples: 100
  random_state: 42
```

### 2. Balanced Dataset Creation
```yaml
# Ensure equal representation of all document types
- name: balanced_sample
  type: sample
  method: stratify
  samples: 1000
  method_kwargs:
    stratify_key: document_type
```

### 3. Compound Stratification
```yaml
# Sample from complex multi-dimensional groups
- name: regional_category_sample
  type: sample
  method: stratify
  samples: 50
  method_kwargs:
    stratify_key: [region, product_category]
    samples_per_group: true  # 50 samples from each region-category pair
```

### 4. Anomaly Detection
```yaml
# Find unusual customer feedback
- name: find_anomalies
  type: sample
  method: outliers
  method_kwargs:
    embedding_keys: [feedback_text]
    std: 3
    keep: true  # Keep the outliers
```

### 5. Similarity Filtering
```yaml
# Find documents similar to a query
- name: similar_docs
  type: sample
  method: outliers
  method_kwargs:
    embedding_keys: [title, abstract]
    center:
      title: "Climate Change Impact"
      abstract: "Study on global warming effects"
    samples: 100
    keep: false  # Keep items close to center
```

### 6. Specific Item Selection
```yaml
# Select specific test cases
- name: test_cases
  type: sample
  method: custom
  samples:
    - test_id: "TC001"
    - test_id: "TC005"
    - test_id: "TC010"
```

### 7. RAG Context Retrieval
```yaml
# Retrieve relevant context for RAG applications
- name: retrieve_context
  type: sample
  method: retrieve_vector
  method_kwargs:
    query: "How does transformer architecture work?"
    embedding_keys:
      - content
    num_chunks: 10
    persist: true  # Keep index for future queries
```

### 8. Knowledge Base Search
```yaml
# Search through documentation
- name: search_docs
  type: sample
  method: retrieve_fts
  method_kwargs:
    query: "troubleshooting network connectivity"
    embedding_keys:
      - question
      - answer
      - tags
    rerank: true
    output_key: "relevant_articles"
```

## Best Practices

1. **Use `random_state` for reproducibility**: Always set a random_state when using uniform or stratified sampling in production pipelines

2. **Choose the right method**:
   - Use `uniform` for general random sampling
   - Use `stratify` when maintaining group proportions is important
   - Use `outliers` for similarity-based filtering or anomaly detection
   - Use `custom` when you know exactly which items you want
   - Use `first` only for debugging
   - Use `retrieve_vector` for semantic similarity search
   - Use `retrieve_fts` for keyword or hybrid search

3. **Consider computational costs**: 
   - The `outliers`, `retrieve_vector`, and `retrieve_fts` methods require generating embeddings, which can be expensive for large datasets
   - Use `persist: true` for retrieval methods when querying the same dataset multiple times

4. **Validate sample sizes**: Ensure your sample size is appropriate for your downstream operations

5. **Combine with other operations**: Sample operations work well with filters and other transformations to create sophisticated data selection pipelines

6. **Stratification best practices**:
   - Use single keys for simple stratification
   - Use compound keys when you need to maintain complex group distributions
   - Use `samples_per_group` when you need a fixed number from each group regardless of group size
