# Rank Operation

The Rank operation in DocETL sorts documents based on specified criteria.
Note that this operation is designed to sort documents along some (latent) attribute in the data. **It is not specifically meant for top-k or retrieval-like queries.**

We adapt algorithms from Human-Powered Sorts and Joins ([VLDB 2012](https://www.vldb.org/pvldb/vol5/p013_adammarcus_vldb2012.pdf)).

## Example: Ranking Debates by Level of Controversy

=== "YAML"

    ```yaml
    - name: rank_by_controversy
      type: rank
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
      rerank_call_budget: 10 # max number of LLM calls to use; also optional
      initial_ordering_method: "likert"
    ```

=== "Python"

    Rank has no dedicated Frame method, so construct the pipeline as a config dict and run it with `DSLRunner`:

    ```python
    from docetl.runner import DSLRunner

    config = {
        "default_model": "gpt-4o-mini",
        "datasets": {
            "debates": {"type": "file", "path": "debates.json"},
        },
        "operations": [
            {
                "name": "rank_by_controversy",
                "type": "rank",
                "prompt": """Order these debate transcripts based on how controversial the discussion is.
    Consider factors like:
    - The level of disagreement between candidates
    - Discussion of divisive topics
    - Strong emotional language
    - Presence of conflicting viewpoints
    - Public reaction mentioned in the transcript

    Debates with the most controversial content should be ranked highest.""",
                "input_keys": ["content", "title", "date"],
                "direction": "desc",
                "rerank_call_budget": 10,  # max number of LLM calls to use; also optional
                "initial_ordering_method": "likert",
            }
        ],
        "pipeline": {
            "steps": [
                {
                    "name": "controversy_ranking",
                    "input": "debates",
                    "operations": ["rank_by_controversy"],
                }
            ],
            "output": {"type": "file", "path": "ranked_debates.json"},
        },
    }
    runner = DSLRunner(config)
    results, _ = runner.run()
    ```

This operation:

1. Generates ordinal scores (Likert scale) for each document — one LLM call **per document**.
2. Creates an initial ranking from those scores.
3. Re-ranks a sliding window of documents with an LLM — `rerank_call_budget` calls.

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

## Algorithm and Implementation

The Rank operation works in these steps:

1. **Initial Ranking**:
    1. The algorithm begins with either an embedding-based or Likert-scale rating approach:
        1. **Embedding-based**: Creates embedding vectors for the ranking criteria and each document, then calculates cosine similarity
        2. **Likert-based** (default): Uses the LLM to rate each document on a 7-point Likert scale based on the criteria. We do this in batches of `batch_size` documents (defaults to 10), and the prompt includes a random sample of `num_calibration_docs` (defaults to 10) documents to calibrate the LLM with.
    2. Documents are initially sorted by their similarity scores or ratings (high to low for desc, low to high for asc)
2. **"Picky Window" Refinement**:
    1. Rather than processing all documents with equal focus, the algorithm employs a "picky window" approach
    2. Starting from the bottom of the currently ranked documents and working upward:
        1. A large window of documents is presented to the LLM
        2. The LLM is asked to identify only the top few documents (configured via `num_top_items_per_window`)
        3. These chosen documents are then moved to the beginning of the window
    3. The window slides upward through the document set with overlapping segments
3. **Resource Utilization**:
   
    1. The window size and step size are calculated from the call budget
    2. Windows overlap (see `overlap_fraction`)
    3. Document positions are tracked with unique identifiers

4. **Output Preparation**:
   
    1. After all windows have been processed, the algorithm assigns a `_rank` field to each document (1-indexed)
    2. Returns the documents in their final sorted order


## Required Parameters

- `name`: A unique name for the operation.
- `type`: Must be set to "rank".
- `prompt`: The prompt specifying the ranking criteria. This does **not** need to be a Jinja template.
- `input_keys`: List of document keys to consider for ranking.
- `direction`: Either "asc" (ascending) or "desc" (descending).

## Optional Parameters

| Parameter                    | Description                                                                                | Default                       |
| ---------------------------- | ------------------------------------------------------------------------------------------ | ----------------------------- |
| `model`                      | The language model to use for LLM-based ranking                                            | Falls back to `default_model` |
| `embedding_model`            | The embedding model to use for similarity calculations                                     | "text-embedding-3-small"      |
| `batch_size`                 | Maximum number of documents to process in a single LLM batch rating (used for the first pass)                       | 10                            |
| `timeout`                    | Timeout for each LLM call in seconds                                                       | 120                           |
| `verbose`                    | Whether to log detailed LLM call statistics                                                 | False                         |
| `num_calibration_docs`     | Number of documents to use for calibration (used for the first pass)                       | 10                            |
| `litellm_completion_kwargs`  | Additional parameters to pass to LiteLLM completion calls                                  | {}                            |
| `bypass_cache`               | If true, bypass the cache for this operation                                               | False                         |
| `initial_ordering_method`    | Method to use for initial ranking: "likert" (default) or "embedding"                       | "likert"                   |
| `k`                          | Number of top items to focus on in the final ranking                                       | None (ranks all items)        |
| `call_budget`                | Maximum number of LLM API calls to make during ranking                                     | 10                           |
| `num_top_items_per_window`   | Number of top items the LLM should select from each window                                 | 3                             |
| `overlap_fraction`           | Fraction of overlap between windows                                                        | 0.5                           |

## Two-Step Ranking Approach

For more complex ranking tasks, a two-step approach can be more effective:

1. First use a `map` operation to extract and structure relevant information
2. Then use the `rank` operation to rank based on the extracted information

??? example "Two-Step Ranking Example"

    === "YAML"

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

          - name: rank_by_meanness
            type: rank
            prompt: |
              Order these debate transcripts based on how mean or hostile the candidates are to each other.
              Focus on the meanness summaries and examples that have been extracted.
              
              Consider:
              - The overall hostility level rating
              - Severity of personal attacks in the key examples
              [... prompt details ...]
            input_keys: ["meanness_summary", "hostility_level", "key_examples", "title", "date"]
            direction: desc
            rerank_call_budget: 10

        pipeline:
          steps:
            - name: meanness_analysis
              input: debates
              operations:
                - extract_hostile_exchanges
                - rank_by_meanness
        ```

    === "Python"

        ```python
        from docetl.runner import DSLRunner

        config = {
            "default_model": "gpt-4o-mini",
            "datasets": {
                "debates": {"type": "file", "path": "debates.json"},
            },
            "operations": [
                {
                    "name": "extract_hostile_exchanges",
                    "type": "map",
                    "output": {
                        "schema": {
                            "meanness_summary": "str",
                            "hostility_level": "int",
                            "key_examples": "list[str]",
                        }
                    },
                    "prompt": """Analyze the following debate transcript for {{ input.title }} on {{ input.date }}:

        {{ input.content }}

        Extract and summarize exchanges where candidates are mean or hostile to each other.
        [... prompt details ...]""",
                },
                {
                    "name": "rank_by_meanness",
                    "type": "rank",
                    "prompt": """Order these debate transcripts based on how mean or hostile the candidates are to each other.
        Focus on the meanness summaries and examples that have been extracted.

        Consider:
        - The overall hostility level rating
        - Severity of personal attacks in the key examples
        [... prompt details ...]""",
                    "input_keys": [
                        "meanness_summary",
                        "hostility_level",
                        "key_examples",
                        "title",
                        "date",
                    ],
                    "direction": "desc",
                    "rerank_call_budget": 10,
                },
            ],
            "pipeline": {
                "steps": [
                    {
                        "name": "meanness_analysis",
                        "input": "debates",
                        "operations": [
                            "extract_hostile_exchanges",
                            "rank_by_meanness",
                        ],
                    }
                ],
                "output": {"type": "file", "path": "ranked.json"},
            },
        }
        runner = DSLRunner(config)
        results, _ = runner.run()
        ```

## Best Practices

1. **Choose Appropriate Input Keys**: Only include document fields relevant to the ranking criteria.

2. **Consider Pre-Processing**: For complex criteria, use a map operation first to extract structured data, then rank on that.

3. **Tune Window Parameters**:
   - Adjust `num_top_items_per_window` based on how selective the ranking needs to be
   - Modify `overlap_fraction` to balance redundancy and completeness
   - Start with defaults and adjust based on results

4. **Direction Matters**:
   - "desc" (descending) ranks the most matching items first
   - "asc" (ascending) ranks the least matching items first

5. **Mind Cost**: Ranking makes multiple LLM calls and embedding requests. For large datasets, sample first to test your approach; the embedding-based first pass (`initial_ordering_method: embedding`) reduces cost. Enable `verbose` during development to see call statistics.

## Performance Considerations

- The rank operation scales with O(n)
- The `verbose` flag adds detailed logging but doesn't affect performance or results
