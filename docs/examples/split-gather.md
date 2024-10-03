# Split and Gather Example: Analyzing Trump Immunity Case

This example demonstrates how to use the Split and Gather operations in DocETL to process and analyze a large legal document, specifically the government's motion for immunity determinations in the case against former President Donald Trump. You can download the dataset we'll be using [here](https://github.com/ucbepic/docetl/blob/main/example_data/post_di_trump_motion.json). This dataset contains a single document.

## Problem Statement

We want to analyze a [lengthy legal document](https://storage.courtlistener.com/recap/gov.uscourts.dcd.258148/gov.uscourts.dcd.258148.252.0.pdf) to identify all people involved in the Trump vs. United States case regarding presidential immunity. The document is too long to process in a single operation, so we need to split it into manageable chunks and then gather context to ensure each chunk can be analyzed effectively.

## Chunking Strategy

When dealing with long documents, it's often necessary to break them down into smaller, manageable pieces. This is where the Split and Gather operations come in handy:

1. **Split Operation**: This divides the document into smaller chunks based on token count or delimiters. For legal documents, using a token count method is often preferable to ensure consistent chunk sizes.

2. **Gather Operation**: After splitting, we use the Gather operation to add context to each chunk. This operation can include content from surrounding chunks, as well as document-level metadata and headers, ensuring that each piece maintains necessary context for accurate analysis.

!!! note "Pipeline Overview"

    Our pipeline will follow these steps:

    1. Extract metadata from the full document
    2. Split the document into chunks
    3. Extract headers from each chunk
    4. Gather context for each chunk
    5. Analyze each chunk to identify people and their involvements in the case
    6. Reduce the results to compile a comprehensive list of people and their roles

## Example Pipeline

Here's a breakdown of the pipeline defined in trump-immunity_opt.yaml:

1.  **Dataset Definition**:
    We define a dataset (json file) with a single document.

2.  **Metadata Extraction**:
    We define a map operation to extract any document-level metadata that we want to pass to each chunk being processed.

3.  **Split Operation**:
    The document is split into chunks using the following configuration:

    ```yaml
    - name: split_find_people_and_involvements
      type: split
      method: token_count
      method_kwargs:
        num_tokens: 3993
      split_key: extracted_text
    ```

    This operation splits the document into chunks of approximately 3993 tokens each. This size is chosen to balance between maintaining context and staying within model token limits. `split_key` should be the field in the document that contains the text to split.

4.  **Header Extraction**:
    We define a map operation to extract headers from each chunk. Then, when rendering each chunk, we can also render the headers in levels above the headers in the chunk--ensuring that we can maintain hierarchical context, even when the headers are in other chunks.

5.  **Gather Operation**:
    Context is gathered for each chunk using the following configuration:

    ```yaml
    - name: gather_extracted_text_find_people_and_involvements
      type: gather
      content_key: extracted_text_chunk
      doc_header_key: headers
      doc_id_key: split_find_people_and_involvements_id
      order_key: split_find_people_and_involvements_chunk_num
      peripheral_chunks:
        next:
          head:
            count: 1
        previous:
          tail:
            count: 1
    ```

    This operation gathers context for each chunk, including the previous chunk, the current chunk, and the next chunk. We also render the headers populated by the previous operation.

    Note that `content_key` should be `_chunk` appended to the name of the field containing the text you are splitting. `doc_id_key` and `order_key` should be the `_id` and `_chunk_num` fields appended to the name of the prior split operation.
    !!! note

        You can define a gather operation without including headers. To do this, simply omit the `doc_header_key` from your gather operation configuration. This is useful when you don't need or haven't extracted hierarchical header information from your document chunks.

6.  **Chunk Analysis**:
    We define a map operation to analyze each chunk.

7.  **Result Reduction**:
    We define a reduce operation to reduce the results of the map operation (applied to each chunk) to a single list of people and their involvements in the case.

Here is the full pipeline configuration, with the split and gather operations highlighted. Assuming the sample dataset looks like this:

```json
[
  {
    "pdf_url": "https://storage.courtlistener.com/recap/gov.uscourts.dcd.258148/gov.uscourts.dcd.258148.252.0.pdf"
  }
]
```

??? example "Full Pipeline Configuration"

    ```yaml linenums="1" hl_lines="26-31 55-67"
    datasets:
      legal_doc:
        type: file
        path: /path/to/your/dataset.json
        parsing: # (1)!
          - function: azure_di_read
            input_key: pdf_url
            output_key: extracted_text
            function_kwargs:
              use_url: true
              include_line_numbers: true

    default_model: gpt-4o-mini

    operations:
      - name: extract_metadata_find_people_and_involvements
        type: map
        model: gpt-4o-mini
        prompt: |
          Given the document excerpt: {{ input.extracted_text }}
          Extract all the people mentioned and summarize their involvements in the case described.
        output:
          schema:
            metadata: str

      - name: split_find_people_and_involvements
        type: split
        method: token_count
        method_kwargs:
          num_tokens: 3993
        split_key: extracted_text

      - name: header_extraction_extracted_text_find_people_and_involvements
        type: map
        model: gpt-4o-mini
        output:
          schema:
            headers: "list[{header: string, level: integer}]"
        prompt: |
          Analyze the following chunk of a document and extract any headers you see.

          { input.extracted_text_chunk }

          Examples of headers and their levels based on the document structure:
          - "GOVERNMENT'S MOTION FOR IMMUNITY DETERMINATIONS" (level 1)
          - "Legal Framework" (level 1)
          - "Section I" (level 2)
          - "Section II" (level 2)
          - "Section III" (level 2)
          - "A. Formation of the Conspiracies" (level 3)
          - "B. The Defendant Knew that His Claims of Outcome-Determinative Fraud Were False" (level 3)
          - "1. Arizona" (level 4)
          - "2. Georgia" (level 4)

      - name: gather_extracted_text_find_people_and_involvements
        type: gather
        content_key: extracted_text_chunk
        doc_header_key: headers
        doc_id_key: split_find_people_and_involvements_id
        order_key: split_find_people_and_involvements_chunk_num
        peripheral_chunks:
          next:
            head:
              count: 1
          previous:
            tail:
              count: 1

      - name: submap_find_people_and_involvements
        type: map
        model: gpt-4o-mini
        output:
          schema:
            people_and_involvements: list[str]
        prompt: |
          Given the document excerpt: {{ input.extracted_text_chunk_rendered }}
          Extract all the people mentioned and summarize their involvements in the case described. Only process the main chunk.

      - name: subreduce_find_people_and_involvements
        type: reduce
        model: gpt-4o-mini
        associative: true
        pass_through: true
        synthesize_resolve: false
        output:
          schema:
            people_and_involvements: list[str]
        reduce_key:
          - split_find_people_and_involvements_id
        prompt: |
          Given the following extracted information about individuals involved in the case, compile a comprehensive list of people and their specific involvements in the case:

          {% for chunk in inputs %}
          {% for involvement in chunk.people_and_involvements %}
          - {{ involvement }}
          {% endfor %}
          {% endfor %}

          Make sure to include all the people and their involvements. If a person has multiple involvements, group them together.

    pipeline:
      steps:
        - name: analyze_document
          input: legal_doc
          operations:
            - extract_metadata_find_people_and_involvements
            - split_find_people_and_involvements
            - header_extraction_extracted_text_find_people_and_involvements
            - gather_extracted_text_find_people_and_involvements
            - submap_find_people_and_involvements
            - subreduce_find_people_and_involvements

      output:
        type: file
        path: /path/to/your/output/people_and_involvements.json
        intermediate_dir: /path/to/your/intermediates
    ```

    1. This is an example parsing function, as explained in the [Parsing](../examples/custom-parsing.md) docs. You can define your own parsing function to extract the text you want to split, or just have the text be directly in the json file.

Running the pipeline with `docetl run pipeline.yaml` will execute the pipeline and save the output to the path specified in the output section. It cost $0.05 and took 23.8 seconds with gpt-4o-mini.

## Optional: Compiling a Pipeline into a Split-Gather Pipeline

You can also compile a pipeline into a split-gather pipeline using the `docetl build` command. Say we had a much simpler pipeline for the same document analysis task as above, with just one map operation to extract people and their involvements.

```yaml
default_model: gpt-4o-mini

datasets:
  legal_doc: # (1)!
    path: /path/to/dataset.json
    type: file
    parsing:
      - input_key: pdf_url
        function: azure_di_read
        output_key: extracted_text
        function_kwargs:
          use_url: true
          include_line_numbers: true

operations:
  - name: find_people_and_involvements
    type: map
    optimize: true
    prompt: |
      Given this document, extract all the people and their involvements in the case described by the document.

      {{ input.extracted_text }}

      Return a list of people and their involvements in the case.
    output:
      schema:
        people_and_involvements: list[str]

pipeline:
  steps:
    - name: analyze_document
      input: legal_doc
      operations:
        - find_people_and_involvements

  output:
    type: file
    path: "/path/to/output/people_and_involvements.json"
```

1. This is an example parsing function, as explained in the [Parsing](../examples/custom-parsing.md) docs. You can define your own parsing function to extract the text you want to split, or just have the text be directly in the json file. If you want the text directly in the json file, you can have your json be a list of objects with a single field "extracted_text".

In the pipeline above, we don't have any split or gather operations. Running `docetl build pipeline.yaml [--model=gpt-4o-mini]` will output a new pipeline_opt.yaml file with the split and gather operations highlighted--like we had defined in the previous example. Note that this cost us $20 to compile, since we tried a bunch of different plans...
