# Sample Operation

The Sample operation in DocETL selects a subset of items from the input data using various sampling methods. While it can be used as a debugging tool to limit data during pipeline development, it also serves as a powerful data selection mechanism for production pipelines.

## Overview

The Sample operation supports five distinct sampling methods:

1. **Uniform**: Random sampling with equal probability for each item
2. **Stratify**: Maintains proportional representation of groups
3. **Outliers**: Identifies items based on embedding distance from a center
4. **Custom**: Selects specific items by matching key values
5. **First**: Takes the first N items (useful for debugging)

## Required Parameters

- `name`: A unique name for the operation
- `type`: Must be set to "sample"
- `method`: The sampling method to use (`uniform`, `stratify`, `outliers`, `custom`, or `first`)
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
  - `stratify_key`: The key to group by (required)

**Example**:
```yaml
- name: stratified_sample
  type: sample
  method: stratify
  samples: 0.2  # Select 20% of data
  method_kwargs:
    stratify_key: category  # Maintain category proportions
  random_state: 42
```

If your data has 70% "A" category and 30% "B" category, the sample will maintain this 70/30 ratio.

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

### 3. Anomaly Detection
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

### 4. Similarity Filtering
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

### 5. Specific Item Selection
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

## Best Practices

1. **Use `random_state` for reproducibility**: Always set a random_state when using uniform or stratified sampling in production pipelines

2. **Choose the right method**:
   - Use `uniform` for general random sampling
   - Use `stratify` when maintaining group proportions is important
   - Use `outliers` for similarity-based filtering or anomaly detection
   - Use `custom` when you know exactly which items you want
   - Use `first` only for debugging

3. **Consider computational costs**: The `outliers` method requires generating embeddings, which can be expensive for large datasets

4. **Validate sample sizes**: Ensure your sample size is appropriate for your downstream operations

5. **Combine with other operations**: Sample operations work well with filters and other transformations to create sophisticated data selection pipelines
