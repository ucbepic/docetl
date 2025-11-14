# Fallback Models

Fallback models provide automatic failover to alternative models when API errors or content errors occur with your primary model. This feature improves pipeline reliability and reduces failures due to temporary API issues or model unavailability.

## Overview

When configured, DocETL will automatically try fallback models in sequence if:

- The primary model encounters an API error (rate limits, service unavailability, etc.)
- The primary model returns invalid content that cannot be parsed
- The primary model fails to respond within expected timeframes

This ensures your pipelines continue running even when individual models experience issues.

## Configuration

Fallback models are configured at the global level in your pipeline YAML file. You can configure separate fallback models for:

- **Completion/Chat operations**: Used by `map`, `reduce`, `resolve`, `filter`, and other LLM-powered operations
- **Embedding operations**: Used by operations that generate embeddings (e.g., `cluster`, `rank`)

### Basic Configuration

The simplest way to configure fallback models is to provide a list of model names:

```yaml
# Default language model for all operations
default_model: gpt-4o-mini

# Fallback models for completion/chat operations
fallback_models:
  - gpt-3.5-turbo
  - claude-3-haiku-20240307

# Fallback models for embedding operations
fallback_embedding_models:
  - text-embedding-3-small
  - text-embedding-ada-002
```

Models will be tried in the order specified. If the primary model fails, DocETL will automatically try the first fallback model, then the second, and so on.

### Advanced Configuration

For more control, you can specify additional LiteLLM parameters for each fallback model:

```yaml
default_model: gpt-4o-mini

# Fallback models with custom parameters
fallback_models:
  - model_name: gpt-3.5-turbo
    litellm_params:
      temperature: 0.0
      max_tokens: 2000
  - model_name: claude-3-haiku-20240307
    litellm_params:
      temperature: 0.0

# Fallback embedding models
fallback_embedding_models:
  - model_name: text-embedding-3-small
    litellm_params: {}
  - model_name: text-embedding-ada-002
    litellm_params: {}
```

## How It Works

When an operation uses a model (either the `default_model` or an operation-specific model), DocETL will:

1. **Try the primary model first**: The operation's specified model (or `default_model`) is attempted first
2. **Fallback on error**: If an API error or content parsing error occurs, DocETL automatically tries the first fallback model
3. **Continue through fallbacks**: If the first fallback also fails, it tries the next fallback model in sequence
4. **Fail only if all models fail**: The operation only fails if all models (primary + all fallbacks) fail

## Example: Complete Pipeline with Fallback Models

Here's a complete example showing how to use fallback models in a pipeline:

```yaml
datasets:
  example_dataset:
    type: file
    path: example_data/example.json

# Default language model for all operations unless overridden
default_model: gpt-4o-mini

# Fallback models for completion/chat operations
# Models will be tried in order when API errors or content errors occur
fallback_models:
  # First fallback model
  - model_name: gpt-3.5-turbo
    litellm_params:
      temperature: 0.0
  # Second fallback model
  - model_name: claude-3-haiku-20240307
    litellm_params:
      temperature: 0.0

# Fallback models for embedding operations
# Separate configuration for embedding model fallbacks
fallback_embedding_models:
  - model_name: text-embedding-3-small
    litellm_params: {}
  - model_name: text-embedding-ada-002
    litellm_params: {}

operations:
  - name: example_map
    type: map
    prompt: "Extract key information from: {{ input.contents }}"
    output:
      schema:
        extracted_info: "str"

pipeline:
  steps:
    - name: process_data
      input: example_dataset
      operations:
        - example_map

  output:
    type: file
    path: example_output.json
```

## Best Practices

### Model Selection

- **Choose compatible models**: Select fallback models that produce similar output formats to your primary model
- **Consider cost**: Fallback models are only used when the primary fails, but choose models that fit your budget
- **Match capabilities**: Ensure fallback models can handle the same types of tasks as your primary model

### Order Matters

- **Place most reliable models first**: Order fallback models by reliability and compatibility with your primary model
- **Consider latency**: Faster models should generally come before slower ones
- **Test fallback behavior**: Verify that your fallback models produce acceptable results for your use case

### Error Handling

- **Monitor fallback usage**: Check logs to see how often fallbacks are triggered
- **Investigate frequent fallbacks**: If fallbacks are used frequently, investigate why the primary model is failing
- **Set appropriate timeouts**: Configure timeouts to avoid waiting too long on failing models

## Operation-Specific Models

You can still override the model for specific operations, and fallback models will apply to those operations as well:

```yaml
default_model: gpt-4o-mini

fallback_models:
  - gpt-3.5-turbo
  - claude-3-haiku-20240307

operations:
  - name: complex_analysis
    type: map
    model: gpt-4o  # This operation uses gpt-4o as primary
    # Fallback models will still be tried if gpt-4o fails
    prompt: "Perform complex analysis: {{ input.text }}"
```

## Limitations

- **LiteLLM dependency**: Fallback models require LiteLLM to be installed and properly configured
- **Model compatibility**: Not all models support the same parameters or produce identical output formats
- **Cost considerations**: Each fallback attempt may incur API costs, even if it fails
- **Latency impact**: Trying multiple models sequentially can increase overall operation time

## Troubleshooting

### Fallback Models Not Working

If fallback models aren't being used:

1. **Check LiteLLM installation**: Ensure LiteLLM is installed (`pip install litellm`)
2. **Verify configuration**: Check that `fallback_models` or `fallback_embedding_models` are correctly formatted in your YAML
3. **Review logs**: Look for warnings about Router creation or fallback model configuration

### Unexpected Model Usage

If you're seeing unexpected models being used:

1. **Check model order**: Verify the order of fallback models matches your expectations
2. **Review error logs**: Check if the primary model is failing and triggering fallbacks
3. **Validate model names**: Ensure all model names are valid and accessible with your API keys

## Example File

A complete working example is available at [`docs/examples/fallback_models_example.yaml`](../examples/fallback_models_example.yaml). You can use this as a starting point for your own pipelines.

## Related Documentation

- [Configuration Guide](configuration.md) - Learn about other optimization configuration options
- [Optimization Overview](overview.md) - Understand DocETL's optimization features
- [Best Practices](../best-practices.md) - General best practices for DocETL pipelines
