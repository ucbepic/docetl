# Troubleshooting MOAR

Common issues and solutions when using MOAR optimization.

## Error: Missing required accuracy metric

!!! error "Error Message"
    `KeyError: Missing required accuracy metric 'your_metric_key'`

**Solution:**

Check that:

1. Your evaluation function returns a dictionary with the `metric_key` you specified
2. The `metric_key` in `optimizer_config` matches the key in your evaluation results
3. Your evaluation function is working correctly (test it independently)

```python
# Test your function
result = evaluate_results("dataset.json", "results.json")
print(result)  # Verify your metric_key is present
```

## Error: Evaluation function takes wrong number of arguments

!!! error "Error Message"
    `TypeError: evaluate_results() takes 1 positional argument but 2 were given`

**Solution:**

Make sure your evaluation function has exactly this signature:

```python
def evaluate_results(dataset_file_path: str, results_file_path: str) -> Dict[str, Any]:
```

And that it's decorated with `@docetl.register_eval`.

## All accuracies showing as 0.0

!!! warning "Symptom"
    All solutions show 0.0 accuracy in the Pareto frontier.

**Possible causes:**

1. **Evaluation function failing silently** - Check the error logs
2. **Result files don't exist** - Make sure pipelines are executing successfully
3. **Metric key doesn't match** - Verify `metric_key` matches what your function returns

**Solution:**

Test your evaluation function independently and check MOAR logs for errors.

## Optimization taking too long

!!! tip "Speed Up Optimization"
    If optimization is taking too long, try:
    
    - Reduce `max_iterations` (e.g., from 40 to 20)
    - Use a smaller sample dataset via `dataset_path`
    - Reduce the number of models in `available_models`
    - Use a faster model for directive instantiation (`model` parameter)

## Best Practices

### Using Sample/Hold-Out Datasets

!!! tip "Avoid Overfitting"
    Always use a sample or hold-out dataset for optimization to avoid optimizing on your test set:
    
    ```yaml
    optimizer_config:
      dataset_path: data/sample_100.json  # Use sample/hold-out for optimization
    ```

### Choosing Models

!!! tip "Model Selection"
    Include a range of models in `available_models` to explore cost-accuracy trade-offs:
    
    ```yaml
    available_models:
      - gpt-4.1-nano      # Cheapest, lower accuracy
      - gpt-4.1-mini      # Low cost, decent accuracy
      - gpt-4.1           # Balanced
      - gpt-4o             # Higher cost, better accuracy
    ```

### Iteration Count

!!! tip "Iteration Guidelines"
    - **10-20 iterations**: Quick exploration, good for testing
    - **40 iterations**: Recommended for most use cases
    - **100+ iterations**: For complex pipelines or when you need the absolute best results

### Evaluation Function Performance

!!! tip "Keep Functions Fast"
    Your evaluation function will be called many times. Make sure it's efficient:
    
    - Avoid expensive computations
    - Cache results if possible
    - Keep the function simple and fast

## Getting Help

If you're still experiencing issues:

1. Check the MOAR logs for detailed error messages
2. Verify your evaluation function works independently
3. Test with a smaller `max_iterations` to isolate issues
4. Review the [Configuration Reference](configuration.md) to ensure all required fields are set

