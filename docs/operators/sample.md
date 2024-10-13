# Sample operation

The Sample operation in DocETL samples items from the input. It is meant mostly as a debugging tool:

Insert it before the last operation, the one you're currently trying to add to the end of a working pipeline, to limit the amount of data it will be fed, so that the run time is small enough to comfortably debug its prompt. Once it seems to be working, you can remove the sample operation. You can then repeat this for each operation you add while developing your pipeline!

## 🚀 Example:

```yaml
- name: cluster_concepts
  type: sample
  method: stratify
  samples: 0.1
  method_kwargs:
    stratify_key: category
  random_state: 42
```

This sample operation will return a pseudo-randomly selected 10% of the samples (samples: 0.1). The random selection will be seeded with a constant (42), meaning the same sample will be returned if you rerun the pipeline (If no random state is given, a different sample will be returned every time). Additionally, the random sampling will sample each value of the category key equally.

## Required Parameters

- name: A unique name for the operation.
- type: Must be set to "sample".
- method: The sampling method to use. Can be "uniform", "stratify", "outliers", or "custom".
- samples: Either a list of key-value pairs representing document ids and values, an integer count of samples, or a float fraction of samples.

## Optional Parameters

| Parameter     | Description                                  | Default                             |
| ------------- | -------------------------------------------- | ----------------------------------- |
| random_state  | An integer to seed the random generator with | Use the (numpy) global random state |
| method_kwargs | Additional parameters for the chosen method  | {}                                  |

## Sampling Methods

### Uniform Sampling

For uniform sampling, no additional parameters are required in method_kwargs.

### Stratified Sampling

For stratified sampling, specify the following in method_kwargs:

- stratify_key: The key to stratify by

### Outlier Sampling

For outlier sampling, specify the following in method_kwargs:

- embedding_keys: A list of keys to use for creating embeddings.
- std: The number of standard deviations to use as the cutoff for outliers.
- samples: The number or fraction of samples to consider as outliers.
- keep: Whether to keep (true) or remove (false) the outliers. Defaults to false.
- center: (Optional) A dictionary specifying the center point for distance calculations. It should look like a document, with all the keys present in the embedding_keys list.

You must specify either "std" or "samples" in the outliers configuration, but not both.

### Custom Sampling

For custom sampling, provide a list of documents to sample in the "samples" parameter. Each document in the list should be a dictionary containing keys that match the keys in your input data.

## Examples:

Uniform sampling:

```yaml
- name: uniform_sample
  type: sample
  method: uniform
  samples: 100
```

Stratified sampling:

```yaml
- name: stratified_sample
  type: sample
  method: stratify
  samples: 0.2
  method_kwargs:
    stratify_key: category
```

Outlier sampling:

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

Custom sampling:

```yaml
- name: custom_sample
  type: sample
  method: custom
  samples:
    - id: 1
    - id: 5
```

Outlier sampling with a center:

```yaml
- name: remove_outliers
  type: sample
  method: outliers
  method_kwargs:
    embedding_keys:
      - concept
      - description
    center:
      concept: Tree house
      description: A small house built among the branches of a tree for children to play in.
```
