# Additional Notes

Here are some additional notes to help you get the most out of your pipeline:

- **Sampling Operations**: If you want to run an operation on a random sample of your data, you can set the sample parameter for that operation. For example:

  ```yaml
  operations:
    extract_medications:
      sample: 100 # This will run the operation on a random sample of 100 items
      # ... rest of the operation configuration
  ```

- **Caching**: DocETL caches the results of operations by default. This means that if you run the same operation on the same data multiple times, the results will be retrieved from the cache rather than being recomputed. You can clear the cache by running docetl clear-cache.

- **The run Function**: The main entry point for running a pipeline is the run function in docetl/cli.py. Here's a description of its parameters and functionality:

::: docetl.cli.run
  handler: python
  options:
    members: - run
  show_root_full_path: true
  show_root_toc_entry: true
  show_root_heading: true
  show_source: false
  show_name: true

- **Intermediate Output**: If you provide an intermediate directory in your configuration, the outputs of each operation will be saved to this directory. This allows you to inspect the results of individual steps in the pipeline and can be useful for debugging or analyzing the pipeline's progress. Set the intermediate_dir parameter in your pipeline's output configuration to specify the directory where intermediate results should be saved; e.g.,

  ```yaml
  pipeline:
    output:
      type: file
      path: ...
      intermediate_dir: intermediate_results
  ```