# Sample operation

The Sample operation in DocETL samples items from the input. It is meant mostly as a debugging tool:

Insert it before the last operation, the one you're currently trying to add to the end of a working pipeline, to limit the amount of data it will be fed, so that the run time is small enough to comfortably debug its prompt. Once it seems to be working, you can remove the sample operation. You can then repeat this for each operation you add while developing your pipeline!

## 🚀 Example:

```yaml
- name: sample_concepts
  type: sample
  method: uniform
  samples: 0.1
  stratify_key: category
  random_state: 42
```

This sample operation will return a pseudo-randomly selected 10% of the samples (samples: 0.1). The random selection will be seeded with a constant (42), meaning the same sample will be returned if you rerun the pipeline (If no random state is given, a different sample will be returned every time). Additionally, the random sampling will sample each value of the category key proportionally.

## Required Parameters

- name: A unique name for the operation.
- type: Must be set to "sample".
- method: The sampling method to use. Can be "uniform", "outliers", "custom", or "first".
- samples: Either a list of key-value pairs representing document ids and values, an integer count of samples, or a float fraction of samples.

## Optional Parameters

| Parameter         | Description                                                      | Default |
| ----------------- | ---------------------------------------------------------------- | ------- |
| random_state      | An integer to seed the random generator with                    | None    |
| stratify_key      | Key(s) to stratify by. Can be a string or list of strings      | None    |
| samples_per_group | When stratifying, sample N items per group vs. proportionally  | False   |
| method_kwargs     | Additional parameters for specific methods (e.g., outliers)    | {}      |

## Sampling Methods

### Uniform Sampling

Randomly samples items from the input data. When combined with stratification, maintains the distribution of the stratified groups.

```yaml
- name: uniform_sample
  type: sample
  method: uniform
  samples: 100
```

### First Sampling

Takes the first N items from the input. When combined with stratification, takes proportionally from each group.

```yaml
- name: first_sample
  type: sample
  method: first
  samples: 50
```

### Outlier Sampling

Samples based on distance from a center point in embedding space. Specify the following in method_kwargs:

- embedding_keys: A list of keys to use for creating embeddings.
- std: The number of standard deviations to use as the cutoff for outliers.
- samples: The number or fraction of samples to consider as outliers.
- keep: Whether to keep (true) or remove (false) the outliers. Defaults to false.
- center: (Optional) A dictionary specifying the center point for distance calculations.

You must specify either "std" or "samples" in the method_kwargs, but not both.

```yaml
- name: remove_outliers
  type: sample
  method: outliers
  method_kwargs:
    embedding_keys:
      - concept
      - description
    std: 2
    keep: false
```

### Custom Sampling

Samples specific items by matching key-value pairs. Stratification is not supported with custom sampling.

```yaml
- name: custom_sample
  type: sample
  method: custom
  samples:
    - id: 1
    - id: 5
```

## Stratification

Stratification can be applied to "uniform", "first", and "outliers" methods. It ensures that the sample maintains the distribution of specified key(s) in the data.

### Single Key Stratification

```yaml
- name: stratified_sample
  type: sample
  method: uniform
  samples: 0.2
  stratify_key: category
```

### Multiple Key Stratification

When using multiple keys, stratification is based on the combination of values:

```yaml
- name: multi_stratified_sample
  type: sample
  method: uniform
  samples: 50
  stratify_key: 
    - type
    - size
```

### Samples Per Group

Instead of proportional sampling, you can sample a fixed number from each stratum:

```yaml
- name: stratified_per_group
  type: sample
  method: uniform
  samples: 10  # Sample 10 items from each group
  stratify_key: category
  samples_per_group: true
```

This also works with fractions:

```yaml
- name: stratified_fraction_per_group
  type: sample
  method: uniform
  samples: 0.3  # Sample 30% from each group
  stratify_key: category
  samples_per_group: true
```

## Complete Examples

Stratified outlier detection:

```yaml
- name: stratified_outliers
  type: sample
  method: outliers
  stratify_key: document_type
  method_kwargs:
    embedding_keys:
      - title
      - content
    std: 1.5
    keep: false
```

Stratified first sampling with multiple keys:

```yaml
- name: stratified_first
  type: sample
  method: first
  samples: 100
  stratify_key:
    - category
    - priority
  samples_per_group: false  # Take proportionally from each combination
```

Outlier sampling with a custom center:

```yaml
- name: centered_outliers
  type: sample
  method: outliers
  method_kwargs:
    embedding_keys:
      - concept
      - description
    center:
      concept: Tree house
      description: A small house built among the branches of a tree for children to play in.
    samples: 20  # Keep the 20 furthest items from the center
    keep: true
```
