# Getting Started with MOAR

This guide walks you through running your first MOAR optimization step by step.

## Step 1: Create Your Pipeline

Start with a standard DocETL pipeline. You can define it in Python or YAML.

=== "Python"

    ```python
    import docetl

    docetl.default_model = "gpt-4o-mini"

    frame = (
        docetl.read_json("data/transcripts.json")
        .map(
            prompt="Extract all medications mentioned in: {{ input.src }}",
            output={"schema": {"medication": "list[str]"}},
        )
    )
    ```

=== "YAML"

    ```yaml
    datasets:
      transcripts:
        path: data/transcripts.json
        type: file

    default_model: gpt-4o-mini

    operations:
      - name: extract_medications
        type: map
        output:
          schema:
            medication: list[str]
        prompt: |
          Extract all medications mentioned in: {{ input.src }}

    pipeline:
      steps:
        - name: medication_extraction
          input: transcripts
          operations:
            - extract_medications
      output:
        type: file
        path: results.json
    ```

!!! note "Standard Pipeline"
    Your pipeline doesn't need any special configuration for MOAR. Just create a normal DocETL pipeline.

## Step 2: Write an Evaluation Function

Write a Python function that scores pipeline output. MOAR calls this function for each pipeline configuration it explores.

!!! info "How Evaluation Works"
    - Your function receives the path to the pipeline's output JSON file
    - You load the results file and compute evaluation metrics
    - You return a dictionary of metrics
    - MOAR uses one specific key from this dictionary (specified by `metric_key`) as the accuracy metric to optimize

=== "Python API"

    Just define a regular Python function:

    ```python
    import json

    def evaluate(results_path):
        with open(results_path) as f:
            output = json.load(f)

        correct_count = 0
        for result in output:
            original_text = result.get("src", "").lower()
            for item in result.get("medication", []):
                if item.lower() in original_text:
                    correct_count += 1

        return {
            "medication_extraction_score": correct_count,
            "total_extracted": len(output),
        }
    ```

    If you need access to the original dataset, use a two-argument signature — the dataset path is passed automatically:

    ```python
    def evaluate(dataset_path, results_path):
        with open(results_path) as f:
            output = json.load(f)
        with open(dataset_path) as f:
            dataset = json.load(f)
        # compare output to dataset...
        return {"medication_extraction_score": computed_score}
    ```

=== "CLI (file-based)"

    For CLI usage, create a Python file with a `@register_eval` decorated function:

    ```python
    # evaluate_medications.py
    import json
    from docetl.utils_evaluation import register_eval

    @register_eval
    def evaluate_results(dataset_file_path: str, results_file_path: str) -> dict:
        with open(results_file_path) as f:
            output = json.load(f)

        correct_count = 0
        for result in output:
            original_text = result.get("src", "").lower()
            for item in result.get("medication", []):
                if item.lower() in original_text:
                    correct_count += 1

        return {
            "medication_extraction_score": correct_count,
            "total_extracted": len(output),
        }
    ```

For more details on evaluation functions, see the [Evaluation Functions guide](evaluation.md).

## Step 3: Run Optimization

=== "Python API (recommended)"

    `frame.optimize()` is the single entry point. Pass your evaluation function and the metric key — everything else has smart defaults. It returns an optimized `Frame` you can `.collect()` or `.write_json()`.

    ```python
    optimized = frame.optimize(
        eval_fn=evaluate,
        metric_key="medication_extraction_score",
    )
    ```

    All optional parameters:

    ```python
    optimized = frame.optimize(
        eval_fn=evaluate,
        metric_key="medication_extraction_score",
        models=None,                 # auto-detect from API keys
        agent_model=None,            # auto-select best available (or set docetl.agent_model)
        max_iterations=20,           # search budget
        save_dir=None,               # defaults to temp dir
        exploration_weight=1.414,    # UCB exploration constant
        dataset_path=None,           # sample dataset for optimization
        max_threads=None,            # max concurrent LLM calls per run
        max_concurrent_agents=3,     # parallel MCTS search agents
    )
    ```

    !!! tip "Auto-Detection"
        Models are auto-detected from your environment API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `AZURE_API_KEY`). The agent model is auto-selected as the best available model. No need to specify these unless you want to override the defaults.

=== "CLI"

    Add an `optimizer_config` to your YAML. Only `evaluation_file` and `metric_key` are required:

    ```yaml
    optimizer_config:
      evaluation_file: evaluate_medications.py
      metric_key: medication_extraction_score
    ```

    Then run:

    ```bash
    docetl build pipeline.yaml
    ```

    MOAR is the default optimizer -- no flag needed.

## Step 4: Review Results

=== "Python API"

    `optimize()` returns an optimized `Frame`. Access the full search results via `.search_results`:

    ```python
    # Run the optimized pipeline
    df = optimized.collect()
    print(f"Execution cost: ${optimized.total_cost:.4f}")

    # Access the MOAR search results
    results = optimized.search_results

    # Best pipeline by accuracy
    best = results.best()
    print(f"Accuracy: {best.accuracy}, Cost: ${best.cost:.4f}")

    # Cheapest pipeline on the frontier
    cheap = results.cheapest()

    # View all frontier solutions
    for plan in results.frontier:
        print(f"Accuracy: {plan.accuracy}, Cost: ${plan.cost:.4f}")

    # Get a DataFrame of all explored plans
    print(results.to_df())
    ```

=== "CLI"

    After optimization completes, check your `save_dir` for:

    - **`experiment_summary.json`** — High-level summary of the run
    - **`pareto_frontier.json`** — List of optimal solutions
    - **`evaluation_metrics.json`** — Detailed evaluation results
    - **`pipeline_*.yaml`** — Optimized pipeline configurations

For details on interpreting results, see [Understanding Results](results.md).

## Next Steps

- Learn about [configuration options](configuration.md)
- See [complete examples](examples.md)
- Read [troubleshooting tips](troubleshooting.md)
