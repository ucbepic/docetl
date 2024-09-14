# Running a Pipeline

In this section, we will walk through an example of a complex medical information extraction pipeline. This pipeline is designed to generate summaries of medications, including their side effects and therapeutic uses, from a collection of medical transcripts. You can download the example dataset [here](../assets/medical_transcripts.json).

## High-Level Pipeline Overview

1. **Data Input**: The pipeline starts by reading medical transcripts from a JSON file.
2. **Medication Extraction**: It analyzes each transcript to identify and list all mentioned medications.
3. **Unnesting**: The extracted medication list is unnested to process each medication individually.
4. **Medication Resolution**: Similar medication names are resolved to standardize the entries. _Note: If you are unsure about the optimal configuration for this operation, you can skip this step and move on to the optimizer section (covered in a later part of this documentation)._
5. **Summary Generation**: For each unique medication, the pipeline generates a summary of side effects and therapeutic uses based on information from all relevant transcripts.
6. **Output**: The final summaries are saved to a JSON file.

Now, let's look at the detailed configuration for this pipeline:

??? example "Side Effects and Therapeutic Uses Extraction Pipeline"

    ```yaml
    datasets:
        transcripts:
            path: medical_transcripts.json
            type: file

    default_model: gpt-4o-mini

    operations:
      - name: extract_medications
        output:
          schema:
            medication: list[str]
        prompt: |
          Analyze the following transcript of a conversation between a doctor and a patient:
          {{ input.src }}
          Extract and list all medications mentioned in the transcript.
          If no medications are mentioned, return an empty list.
        type: map

      - name: resolve_medications
        blocking_keys:
          - medication
        blocking_threshold: 0.6162
        comparison_prompt: |
          Compare the following two medication entries:
          Entry 1: {{ input1.medication }}
          Entry 2: {{ input2.medication }}
          Are these medications likely to be the same or closely related?
        embedding_model: text-embedding-3-small
        output:
          schema:
            medication: str
        resolution_prompt: |
          Given the following matched medication entries:
          {% for entry in matched_entries %}
          Entry {{ loop.index }}: {{ entry.medication }}
          {% endfor %}
          Determine the best resolved medication name for this group of entries. The resolved
          name should be a standardized, widely recognized medication name that best represents
          all matched entries.
        type: resolve

      - name: summarize_prescriptions
        output:
          schema:
            side_effects: str
            uses: str
        prompt: |
          Here are some transcripts of conversations between a doctor and a patient:

          {% for value in values %}
          Transcript {{ loop.index }}:
          {{ value.src }}
          {% endfor %}

          For the medication {{ reduce_key }}, please provide the following information based on all the transcripts above:

          1. Side Effects: Summarize all mentioned side effects of {{ reduce_key }}.
          2. Therapeutic Uses: Explain the medical conditions or symptoms for which {{ reduce_key }} was prescribed or recommended.

          Ensure your summary:
          - Is based solely on information from the provided transcripts
          - Focuses only on {{ reduce_key }}, not other medications
          - Includes relevant details from all transcripts
          - Is clear and concise
          - Includes quotes from the transcripts
        reduce_key:
          - medication
        type: reduce

      - name: unnest_medications
        type: unnest
        unnest_key: medication

    pipeline:
      output:
        path: medication_summaries.json
        type: file
      steps:
        - input: transcripts
          name: medical_info_extraction
          operations:
            - extract_medications
            - unnest_medications
            - resolve_medications
            - summarize_prescriptions
    ```

This example pipeline configuration demonstrates a complex medical information extraction task. It includes all the necessary components: datasets, default model, operations, and pipeline specification. When executed, this pipeline will process medical transcripts, extract medication information, resolve similar medications, and generate summaries for each medication, including side effects and therapeutic uses.

## Running the Pipeline

To run a pipeline in docetl, follow these steps:

Ensure your pipeline configuration includes all the required components as described in the [Pipelines](../concepts/pipelines.md) documentation. Your configuration should specify:

- Default model
- Datasets
- Operations
- Pipeline specification (steps and output)

Once you have your pipeline configuration ready, you can execute it using the `docetl run` command if you're confident that this pipeline is suitable for your task and data. This is typically the case when your documents are relatively small and your task is straightforward.

For example, to run a pipeline defined in `pipeline.yaml`, use the following command:

```bash
docetl run pipeline.yaml
```

!!! note

    If you're unsure about the optimal pipeline configuration or dealing with more complex scenarios, you may want to skip directly to the optimizer section (covered in a later part of this documentation).

As the pipeline runs, docetl will display progress information and eventually show the output. Here's an example of what you might see:

```
[Placeholder for pipeline execution output]
```

This pipeline configuration and execution process allows you to efficiently extract, process, and summarize medical information from transcripts using a series of well-defined operations.

## Additional Notes

Here are some additional notes to help you get the most out of your pipeline:

- **Sampling Operations**: If you want to run an operation on a random sample of your data, you can set the `sample` parameter for that operation. For example:

  ```yaml
  operations:
    extract_medications:
      sample: 100 # This will run the operation on a random sample of 100 items
      # ... rest of the operation configuration
  ```

- **The `run` Function**: The main entry point for running a pipeline is the `run` function in `docetl/cli.py`. Here's a description of its parameters and functionality:


::: docetl.cli.run
    handler: python
    options:
        members:
            - run
        show_root_full_path: true
        show_root_toc_entry: true
        show_root_heading: true
        show_source: false
        show_name: true

- **Raw Object Output**: We have not implemented this yet! But we are working on it. If you need access to the raw objects produced by the pipeline for debugging or further processing, use the `--write-raw-objects` flag.
