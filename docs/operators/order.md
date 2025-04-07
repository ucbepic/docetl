# Order Operation

The Order operation in DocETL sorts documents based on specified criteria using embedding similarity and LLM-based ranking.

## 🚀 Example: Ranking Debates by Level of Controversy

Let's see a practical example of using the Order operation to rank political debates based on how controversial they are:

```yaml
- name: order_by_controversy
  type: order
  prompt: |
    Order these debate transcripts based on how controversial the discussion is.
    Consider factors like:
    - The level of disagreement between candidates
    - Discussion of divisive topics
    - Strong emotional language
    - Presence of conflicting viewpoints
    - Public reaction mentioned in the transcript
    
    Debates with the most controversial content should be ranked highest.
  input_keys: ["content", "title", "date"]
  direction: desc
  verbose: true
```

This Order operation ranks debate transcripts from most controversial to least controversial by:

1. First generating embeddings for the ordering criteria and each document
2. Creating an initial ranking based on embedding similarity
3. Using an LLM to perform more precise rankings on batches of documents
4. Merging the batch rankings into a coherent global ordering
5. Adding ranking information to each document

??? example "Sample Input and Output"

    Input:
    ```json
    [
      {
        "title": "Presidential Debate: Economy and Trade",
        "date": "2020-09-29",
        "content": "Moderator: Let's discuss trade policies. Candidate A, your response?\n\nCandidate A: My opponent's policies have shipped jobs overseas for decades! Our workers are suffering while other countries laugh at us.\n\nCandidate B: That's simply not true. The data shows our export growth has been strong. My opponent doesn't understand basic economics.\n\nCandidate A: [interrupting] You've been in government for 47 years and haven't fixed anything!\n\nCandidate B: If you'd let me finish... The manufacturing sector has actually added jobs under our policies.\n\nModerator: Please allow each other to finish. Let's move to healthcare..."
      },
      {
        "title": "Vice Presidential Debate: Foreign Policy",
        "date": "2020-10-07",
        "content": "Moderator: What would your administration's approach be to China?\n\nCandidate C: We need strategic engagement that protects American interests while avoiding unnecessary conflict. My opponent has proposed policies that would damage our diplomatic relationships.\n\nCandidate D: I respectfully disagree with my colleague. Our current approach has been too soft. We need to stand firm on human rights issues and trade imbalances.\n\nCandidate C: I think we actually agree on the goals, if not the methods. The question is how to achieve them without harmful escalation.\n\nCandidate D: That's a fair point. Perhaps there's a middle ground that maintains pressure while keeping dialogue open.\n\nModerator: Thank you both for that thoughtful exchange. Moving to the Middle East..."
      }
    ]
    ```

    Output:
    ```json
    [
      {
        "title": "Presidential Debate: Economy and Trade",
        "date": "2020-09-29",
        "content": "Moderator: Let's discuss trade policies. Candidate A, your response?\n\nCandidate A: My opponent's policies have shipped jobs overseas for decades! Our workers are suffering while other countries laugh at us.\n\nCandidate B: That's simply not true. The data shows our export growth has been strong. My opponent doesn't understand basic economics.\n\nCandidate A: [interrupting] You've been in government for 47 years and haven't fixed anything!\n\nCandidate B: If you'd let me finish... The manufacturing sector has actually added jobs under our policies.\n\nModerator: Please allow each other to finish. Let's move to healthcare...",
        "_rank": 1
      },
      {
        "title": "Vice Presidential Debate: Foreign Policy",
        "date": "2020-10-07",
        "content": "Moderator: What would your administration's approach be to China?\n\nCandidate C: We need strategic engagement that protects American interests while avoiding unnecessary conflict. My opponent has proposed policies that would damage our diplomatic relationships.\n\nCandidate D: I respectfully disagree with my colleague. Our current approach has been too soft. We need to stand firm on human rights issues and trade imbalances.\n\nCandidate C: I think we actually agree on the goals, if not the methods. The question is how to achieve them without harmful escalation.\n\nCandidate D: That's a fair point. Perhaps there's a middle ground that maintains pressure while keeping dialogue open.\n\nModerator: Thank you both for that thoughtful exchange. Moving to the Middle East...",
        "_rank": 2
      }
    ]
    ```

This example demonstrates how the Order operation can semantically sort documents based on complex criteria, providing a ranking that would be difficult to achieve with keyword matching or rule-based approaches.

## Algorithm and Implementation

The Order operation works in these steps:

1. **Embedding-Based Ranking**:
   - Create an embedding vector for the ordering criteria prompt
   - Create embedding vectors for each document using specified input_keys
   - Calculate cosine similarity between the criteria vector and each document vector
   - Sort documents by similarity score (high to low for desc, low to high for asc)

2. **Batch Processing for LLM Ranking**:
   - Divide documents into batches of size `batch_size`
   - For each batch:
     - Generate a prompt containing all documents in the batch
     - Ask the LLM to rank them according to the criteria
     - Parse the response into a ranked list of indices

3. **Merging**:
   - For datasets with <= 2*`num_grounding_examples` items:
     - Combine the batch rankings, removing duplicates
   - For larger datasets:
     - Split the initial embedding-based ranking into chunks of size `chunk_size`
     - For each chunk:
       - If first chunk: include `num_grounding_examples` from batch rankings
       - Ask LLM to rank the chunk
       - Add ranked chunk to final ranking
   - Add any missing document indices to the end

4. **Output Preparation**:
   - Reorder the input documents based on the final ranking
   - Add a `_rank` field to each document (1-indexed)

The operation handles fallback strategies when LLM ranking fails by reverting to embedding-based similarity for affected batches.

## Required Parameters

- `name`: A unique name for the operation.
- `type`: Must be set to "order".
- `prompt`: The prompt specifying the ordering criteria.
- `input_keys`: List of document keys to consider for ordering.
- `direction`: Either "asc" (ascending) or "desc" (descending).

## Optional Parameters

| Parameter                    | Description                                                                                | Default                       |
| ---------------------------- | ------------------------------------------------------------------------------------------ | ----------------------------- |
| `model`                      | The language model to use for LLM-based ranking                                            | Falls back to `default_model` |
| `embedding_model`            | The embedding model to use for similarity calculations                                     | "text-embedding-3-small"      |
| `num_grounding_examples`     | Number of examples from batch rankings to use for grounding chunk verification             | 5                             |
| `batch_size`                 | Maximum number of documents to process in a single LLM batch ranking                       | 10                            |
| `chunk_size`                 | Size of chunks for the verification phase                                                  | 5                             |
| `timeout`                    | Timeout for each LLM call in seconds                                                       | 120                           |
| `verbose`                    | Whether to log detailed ordering statistics                                                | False                         |
| `litellm_completion_kwargs`  | Additional parameters to pass to LiteLLM completion calls                                  | {}                            |
| `bypass_cache`               | If true, bypass the cache for this operation                                               | False                         |

## Two-Step Ordering Approach

For more complex ordering tasks, a two-step approach can be more effective:

1. First use a `map` operation to extract and structure relevant information
2. Then use the `order` operation to rank based on the extracted information

??? example "Two-Step Ordering Example"

    ```yaml
    operations:
      - name: extract_hostile_exchanges
        type: map
        output:
          schema:
            meanness_summary: "str"
            hostility_level: "int"
            key_examples: "list[str]"
        prompt: |
          Analyze the following debate transcript for {{ input.title }} on {{ input.date }}:

          {{ input.content }}

          Extract and summarize exchanges where candidates are mean or hostile to each other.
          [... prompt details ...]

      - name: order_by_meanness
        type: order
        prompt: |
          Order these debate transcripts based on how mean or hostile the candidates are to each other.
          Focus on the meanness summaries and examples that have been extracted.
          
          Consider:
          - The overall hostility level rating
          - Severity of personal attacks in the key examples
          [... prompt details ...]
        input_keys: ["meanness_summary", "hostility_level", "key_examples", "title", "date"]
        direction: desc

    pipeline:
      steps:
        - name: meanness_analysis
          input: debates
          operations:
            - extract_hostile_exchanges
            - order_by_meanness
    ```

This approach:
1. First extracts structured data about hostility in each debate
2. Then orders debates based on this pre-processed data
3. Results in more accurate ordering by working with focused, structured information

## Best Practices

1. **Craft Clear Ordering Criteria**: Write clear, specific prompts that guide the LLM to understand the ordering priorities.

2. **Choose Appropriate Input Keys**: Only include document fields that are relevant to the ordering criteria to reduce noise.

3. **Consider Pre-Processing**: For complex criteria, use a map operation first to extract structured data that makes ordering more effective.

4. **Tune Batch and Chunk Sizes**:
   - For larger documents, use smaller batch sizes
   - For simpler ordering criteria, larger batches may be more efficient
   - Start with defaults and adjust based on results

5. **Use Verbose Mode During Development**: Enable the `verbose` flag during development to understand how the ordering process works and verify the results.

6. **Direction Matters**: Choose "asc" or "desc" carefully based on your use case:
   - "desc" (descending) ranks the most matching items first
   - "asc" (ascending) ranks the least matching items first

7. **Mind Cost Considerations**: The ordering operation makes multiple LLM calls and embedding requests. For large datasets, consider sampling first to test your approach.

## Performance Considerations

- The order operation scales with O(n) for embedding generation
- LLM batch ranking requires O(n/batch_size) calls
- For large datasets (>100 documents), the operation will display a confirmation prompt
- Using a smaller model for the `embedding_model` can reduce costs without significantly impacting quality
- The `verbose` flag adds detailed logging but doesn't affect performance or results
