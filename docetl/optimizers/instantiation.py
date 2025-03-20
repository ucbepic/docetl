import json

import yaml


def generate_rewrite_prompt(
    candidate, pipeline_config, expected_ops=None, sample_docs=None
):
    """
    Construct a prompt for the LLM agent.
    - Lists the full pipeline code (from pipeline_config).
    - Lists the candidate skeleton, including tags for synthesized operators and their original op names.
    - Instructs the agent to write Jinja template prompts and output schemas for each synthesized operation.
    - Provides examples and context.

    Args:
        candidate: The candidate skeleton
        pipeline_config: The original pipeline configuration
        expected_ops: Optional list of expected operations in execution order
        sample_docs: Optional sample documents for context
    """
    # Convert pipeline_config to YAML.
    pipeline_yaml = yaml.dump(pipeline_config, default_flow_style=False)

    # Generate an execution-order representation of the skeleton
    # This reverses the tree structure to show operations in execution order
    execution_order = []

    def collect_execution_order(node, depth=0):
        """Recursively collect nodes in execution order (depth-first)"""
        # Process children first (since the tree is reversed)
        for child in node.children:
            collect_execution_order(child, depth + 1)

        # Skip non-processing operations like scan and step_boundary
        if node.op_type in ["scan", "step_boundary"]:
            return

        # Then process the current node
        tag = " (synth)" if hasattr(node, "synthesized") and node.synthesized else ""
        orig_op = (
            f"orig:{node.original_op.config.get('type')}"
            if hasattr(node, "original_op") and node.original_op
            else "None"
        )
        execution_order.append((depth, f"{node.op_type}{tag}[{orig_op}]"))

    # Traverse the skeleton to get execution order
    collect_execution_order(candidate)

    # Format the execution order into a readable string
    execution_order_str = "Execution Order (First to Last):\n"
    for depth, node_str in execution_order:
        indent = "  " * depth
        execution_order_str += f"{indent}→ {node_str}\n"

    # If we have expected ops and this is a reprompt, add more explicit instructions
    reprompt_str = ""
    if expected_ops:
        reprompt_str = "\nYOUR PREVIOUS RESPONSE DID NOT MATCH THE EXPECTED EXECUTION ORDER. Please ensure your operations list strictly follows this order:\n"
        for i, op in enumerate(expected_ops):
            synthesis = "(synthesized)" if op["synthesized"] else "(original)"
            reprompt_str += f"{i+1}. {op['type']} {synthesis}\n"

    # Sample document information for context
    sample_docs_info = ""
    if sample_docs:
        sample_docs_info = "\nSample Documents Statistics:\n"
        for i, doc in enumerate(
            sample_docs[:2]
        ):  # Limit to first 2 documents for brevity
            sample_docs_info += f"Document {i+1}:\n"
            for key, value in doc.items():
                if isinstance(value, str):
                    # Limit text length
                    sample_value = value[:300] + ("..." if len(value) > 300 else "")
                    sample_docs_info += f"  {key}: {sample_value}\n"
                else:
                    sample_docs_info += f"  {key}: {type(value).__name__}\n"

    # Add examples for each operation type
    operation_examples = """
Examples of Different Operation Types:

1. Map Operation Example:
```yaml
- name: analyze_news_article
  type: map
  prompt: |
    Analyze the following news article:
    "{{ input.article }}"

    Provide the following information:
    1. Main topic (1-3 words)
    2. Summary (2-3 sentences)
    3. Key entities mentioned (list up to 5, with brief descriptions)
    4. Sentiment towards the main topic (positive, negative, or neutral)
    5. Potential biases or slants in reporting (if any)
    6. Relevant categories (e.g., politics, technology, environment; list up to 3)
    7. Credibility score (1-10, where 10 is highly credible)

  output:
    schema:
      main_topic: string
      summary: string
      key_entities: list[object]
      sentiment: string
      biases: list[string]
      categories: list[string]
      credibility_score: integer
```

2. Parallel Map Operation Example:
```yaml
- name: process_job_application
  type: parallel_map
  prompts:
    - name: extract_skills
      prompt: "Given the following resume: '{{ input.resume }}', list the top 5 relevant skills for a software engineering position."
      output_keys:
        - skills
      model: gpt-4o-mini
    - name: calculate_experience
      prompt: "Based on the work history in this resume: '{{ input.resume }}', calculate the total years of relevant experience for a software engineering role."
      output_keys:
        - years_experience
      model: gpt-4o-mini
    - name: evaluate_cultural_fit
      prompt: "Analyze the following cover letter: '{{ input.cover_letter }}'. Rate the candidate's potential cultural fit on a scale of 1-10, where 10 is the highest."
      output_keys:
        - cultural_fit_score
      model: gpt-4o-mini
  output:
    schema:
      skills: list[string]
      years_experience: float
      cultural_fit_score: integer
```

3. Reduce Operation Example:
```yaml
- name: summarize_feedback
  type: reduce
  reduce_key: department
  prompt: |
    Summarize the customer feedback for the {{ inputs[0].department }} department:

    {% for item in inputs %}
    Feedback {{ loop.index }}: {{ item.feedback }}
    {% endfor %}

    Provide a concise summary of the main points and overall sentiment.
  output:
    schema:
      summary: string
      sentiment: string
```

4. Split Operation Example:
```yaml
- name: split_transcript
  type: split
  split_key: transcript
  method: token_count
  method_kwargs:
    num_tokens: 500
    model: gpt-4o-mini
```

5. Gather Operation Example:
```yaml
- name: context_gatherer
  type: gather
  content_key: transcript
  doc_id_key: split_transcript_id  # _id from split op name
  order_key: split_transcript_chunk_num  # _chunk_num from split op name
  peripheral_chunks:
    previous:
      count: 1
      content_key: transcript
```

Gather operations can have more complex peripheral_chunks configurations, like:
```yaml
peripheral_chunks:
  previous:
    head:
      count: 1
      content_key: full_content
    tail:
      count: 2
      content_key: full_content
  next:
    head:
      count: 1
      content_key: full_content
```
But if in doubt, use 1 previous and 0 next.
"""

    prompt = f"""
You are an LLM agent tasked with generating detailed configuration for a newly optimized LLM-powered document processing pipeline.
The optimized pipeline must perform the same overall task as the original pipeline but achieve higher accuracy by decomposing complex operations into smaller subtasks.

Below is the original pipeline code (in YAML) provided in the variable "pipeline_config":
----------------
{pipeline_yaml}
----------------

Below is the candidate rewrite skeleton for the pipeline.
Each node is displayed with its operator type, a tag if it is synthesized, and the original operator name for context.
IMPORTANT: The operations must be executed in the order shown below, from top to bottom:
----------------
{execution_order_str}
----------------{reprompt_str}
{sample_docs_info}

{operation_examples}

Your task is to analyze the candidate skeleton and provide the complete list of operations that should be executed in order.
This includes both original operations and newly synthesized ones.

For each operation in the skeleton, I'll ask you to provide its configuration one by one.
First, please provide the full list of operations with their types and whether they need to be synthesized.

Important:
- You MUST follow the execution order shown above exactly - do not add new operations or change the order.
- Operations marked as (synth) in the skeleton need new configurations (synthesized = true).
- Original operations (not marked as synth) should keep their original configuration (synthesized = false).
- For operations that don't need synthesis (synthesize = false), please include the full configuration object from the original pipeline config.
- For split operations, provide a reasonable token count (usually between 500-1000 tokens) based on the nature of the documents.
- For gather operations, specify how context should be collected (previous/next chunks). If in doubt, use 1 previous chunk and 0 next chunks.
- For reduce operations, specify the reduce_key (usually the document ID created by a preceding split operation).
- All map operations should have a prompt and output schema.
- Output schema types for synthesized operations should be "string" for all fields, unless it's the final operation in a rewrite pattern (then use the original output schema).

Here are some example rewrite decomposition patterns to help you understand the context:
  1. Decomposition: {{"pattern": ["map"], "skeleton": ["split", "gather", "map", "reduce"]}}
     (Useful for long documents: split divides the document, gather collects context, map extracts refined info, reduce aggregates it.)
  2. Decomposition: {{"pattern": ["map"], "skeleton": ["split", "gather", "sample", "map", "reduce"]}}
     (Similar to above but includes sampling.)
  3. Decomposition: {{"pattern": ["map"], "skeleton": ["map*"]}}
     (Breaks a map into a chain of map operations for iterative refinement.)
  4. Decomposition: {{"pattern": ["map"], "skeleton": ["parallel_map", "map"]}}
     (Splits the map operation into independent parallel subtasks whose outputs are unified later.)

Please respond with an array of operations, each containing:
1. name: A descriptive name for the operation
2. type: The operation type (e.g., split, gather, map, reduce, parallel_map)
3. synthesize: Boolean indicating whether this operation needs to be synthesized (true) or can be reused from the original config (false)
4. original_op: The name of the original operation this is derived from (if applicable)
5. config: The complete configuration object if synthesize=false (copied from the original pipeline)
    """
    return prompt


def generate_op_config_prompt(
    op_name,
    op_type,
    original_op_name,
    pipeline_config,
    previous_ops,
    is_final_op=False,
    sample_docs=None,
):
    """
    Generate a prompt to get configuration for a specific operation.
    This function returns a concise prompt since the LLM can see the previous conversation.

    Args:
        op_name: The name of the operation
        op_type: The type of the operation
        original_op_name: The name of the original operation (if applicable)
        pipeline_config: The original pipeline configuration
        previous_ops: List of operations already configured
        is_final_op: Whether this is the final operation in a rewrite pattern
        sample_docs: Sample documents for context
    """
    # Find the original operation if it exists
    original_op = None
    if original_op_name and "ops" in pipeline_config:
        for op in pipeline_config["ops"]:
            if op["name"] == original_op_name:
                original_op = op
                break

    original_op_info = ""
    if original_op:
        original_op_info = f"\nThis operation is derived from: {original_op_name}"
        if is_final_op:
            original_op_info += f"\nThis is the final operation in a rewrite pattern. You should match the output schema of the original operation: {json.dumps(original_op.get('output', {}))}"

    type_specific_instructions = ""
    example_config = ""

    if op_type == "reduce":
        type_specific_instructions = """
For a reduce operation, your prompt MUST include proper Jinja syntax for looping over inputs:
```
{% for item in inputs %}
  Process item {{ loop.index }}: {{ item.field_name }}
{% endfor %}
```

And you must include a suitable reduce_key parameter that groups documents together, often based on a document ID field.
"""
        example_config = """
Example of a Reduce Operation:
```yaml
- name: summarize_feedback
  type: reduce
  reduce_key: department
  prompt: |
    Summarize the customer feedback for the {{ inputs[0].department }} department:

    {% for item in inputs %}
    Feedback {{ loop.index }}: {{ item.feedback }}
    {% endfor %}

    Provide a concise summary of the main points and overall sentiment.
  output:
    schema:
      summary: string
      sentiment: string
```
"""
    elif op_type == "map":
        type_specific_instructions = """
For a map operation, your prompt should use Jinja syntax to access input fields:
```
Analyze the following content: {{ input.content }}
```

Make sure to include all necessary fields that this operation needs to access.
"""
        example_config = """
Example of a Map Operation:
```yaml
- name: analyze_news_article
  type: map
  prompt: |
    Analyze the following news article:
    "{{ input.article }}"

    Provide the following information:
    1. Main topic (1-3 words)
    2. Summary (2-3 sentences)
    3. Key entities mentioned (list up to 5, with brief descriptions)
    4. Sentiment towards the main topic (positive, negative, or neutral)
    5. Potential biases or slants in reporting (if any)
    6. Relevant categories (e.g., politics, technology, environment; list up to 3)
    7. Credibility score (1-10, where 10 is highly credible)

  output:
    schema:
      main_topic: string
      summary: string
      key_entities: list[object]
      sentiment: string
      biases: list[string]
      categories: list[string]
      credibility_score: integer
```
"""
    elif op_type == "split":
        type_specific_instructions = """
For a split operation, specify:
- The split_key (what field to split)
- method (usually "token_count")
- method_kwargs with appropriate num_tokens (usually 500-1000) and model name
"""
        example_config = """
Example of a Split Operation:
```yaml
- name: split_transcript
  type: split
  split_key: transcript
  method: token_count
  method_kwargs:
    num_tokens: 500
    model: gpt-4o-mini
  output:
    schema:
      split_transcript_id: string
      split_transcript_chunk_num: integer
      transcript: string
      total_chunks: integer
```

Note that the output schema for a split operation typically includes fields with the pattern:
- split_{operation_name}_id
- split_{operation_name}_chunk_num
These IDs will be referenced by gather operations.
"""
    elif op_type == "gather":
        type_specific_instructions = """
For a gather operation, specify:
- content_key (the key containing content to gather)
- doc_id_key (ID field from split operation, typically split_{split_op_name}_id)
- order_key (chunk order field from split, typically split_{split_op_name}_chunk_num)
- peripheral_chunks (how to gather context from previous/next chunks)

If in doubt, for peripheral_chunks use:
  previous:
    count: 1
    content_key: same_as_content_key
  next:
    count: 0
"""
        example_config = """
Example of a Gather Operation:
```yaml
- name: context_gatherer
  type: gather
  content_key: transcript
  doc_id_key: split_transcript_id  # _id from split op name
  order_key: split_transcript_chunk_num  # _chunk_num from split op name
  peripheral_chunks:
    previous:
      count: 1
      content_key: transcript
    next:
      count: 0
  output:
    schema:
      doc_id: string
      chunk_id: integer
      transcript: string
      context: object
```

For more complex scenarios, the peripheral_chunks can be more detailed:
```yaml
peripheral_chunks:
  previous:
    head:
      count: 1
      content_key: full_content
    middle:
      content_key: summary_content
    tail:
      count: 2
      content_key: full_content
  next:
    head:
      count: 1
      content_key: full_content
```
"""
    elif op_type == "parallel_map":
        type_specific_instructions = """
For a parallel_map operation, you need to specify:
- A list of prompts, each with its own name, prompt template, and output keys
- Each prompt should focus on a specific subtask or aspect of the data
- The final output schema should combine the outputs from all prompts
"""
        example_config = """
Example of a Parallel Map Operation:
```yaml
- name: process_job_application
  type: parallel_map
  prompts:
    - name: extract_skills
      prompt: "Given the following resume: '{{ input.resume }}', list the top 5 relevant skills for a software engineering position."
      output_keys:
        - skills
      model: gpt-4o-mini
    - name: calculate_experience
      prompt: "Based on the work history in this resume: '{{ input.resume }}', calculate the total years of relevant experience for a software engineering role."
      output_keys:
        - years_experience
      model: gpt-4o-mini
    - name: evaluate_cultural_fit
      prompt: "Analyze the following cover letter: '{{ input.cover_letter }}'. Rate the candidate's potential cultural fit on a scale of 1-10, where 10 is the highest."
      output_keys:
        - cultural_fit_score
      model: gpt-4o-mini
  output:
    schema:
      skills: list[string]
      years_experience: float
      cultural_fit_score: integer
```
"""

    prompt = f"""
Now I need you to generate the configuration for the operation named: {op_name}
Type: {op_type}{original_op_info}

Based on the information from our conversation, generate a complete configuration for this operation.
Your response should be a valid JSON object containing the complete configuration.

{example_config}

IMPORTANT:
- The operation type MUST be "{op_type}" - do not use "unknown" or any other type.
- Output schema types should be "string" for all fields{"" if is_final_op else " (use simple string types)"}.
{type_specific_instructions}
"""
    return prompt


def invoke_rewrite_agent(candidate, pipeline_config, llm_client, sample_docs, console):
    """
    Given a candidate skeleton and the pipeline configuration, iteratively build
    configurations for all operations in the pipeline.

    Args:
        candidate: The candidate skeleton with operations to synthesize
        pipeline_config: Original pipeline configuration
        llm_client: The LLM client for generating configurations
        sample_docs: Optional sample documents for context (default: None)

    Returns:
        List of operation configurations in execution order
    """
    # Print the candidate skeleton for debugging
    console.print("\n[bold blue]Candidate Skeleton:[/bold blue]")
    console.print("=" * 80)
    console.print(f"{candidate}")
    console.print("=" * 80)

    # Generate an execution-order list for validation
    execution_order = []

    def collect_execution_order(node):
        """Recursively collect nodes in execution order (depth-first)"""
        # Process children first (since the tree is reversed)
        for child in node.children:
            collect_execution_order(child)

        # Then process the current node
        # Skip non-processing operations like scan and step_boundary for the LLM
        if node.op_type not in ["scan", "step_boundary"]:
            execution_order.append(
                {
                    "type": node.op_type,
                    "synthesized": hasattr(node, "synthesized") and node.synthesized,
                    "original_op": (
                        node.original_op
                        if hasattr(node, "original_op") and node.original_op
                        else None
                    ),
                }
            )
        # But keep track of ALL operations for our final pipeline
        all_execution_order.append(
            {
                "type": node.op_type,
                "synthesized": hasattr(node, "synthesized") and node.synthesized,
                "original_op": (
                    node.original_op
                    if hasattr(node, "original_op") and node.original_op
                    else None
                ),
            }
        )

    # Keep track of ALL operations including scan/step_boundary
    all_execution_order = []

    # Traverse the skeleton to get execution order
    collect_execution_order(candidate)

    # Print execution order for debugging
    console.print("\n[bold blue]Expected Execution Order:[/bold blue]")
    console.print("=" * 80)
    for i, op in enumerate(execution_order):
        synth_status = (
            "[yellow]Synthesized[/yellow]"
            if op["synthesized"]
            else "[green]Original[/green]"
        )
        original_op_info = ""
        if op["original_op"] and hasattr(op["original_op"], "config"):
            original_op_info = f" (original: {op['original_op'].config.get('name')})"
        console.print(f"{i+1}. {op['type']} - {synth_status}{original_op_info}")
    console.print("=" * 80)

    # Step 1: Get the ordered list of operations with metadata from the LLM
    initial_prompt = generate_rewrite_prompt(
        candidate, pipeline_config, sample_docs=sample_docs
    )
    system_prompt = "You are a helpful assistant specializing in optimizing document processing pipelines."

    # Define parameters schema for pipeline ops list with metadata
    pipeline_ops_schema = {
        "type": "object",
        "properties": {
            "operations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string"},
                        "synthesize": {"type": "boolean"},
                        "original_op": {"type": "string", "nullable": True},
                        "config": {
                            "type": "object",
                            "additionalProperties": True,
                            "nullable": True,
                        },
                    },
                    "required": ["name", "type", "synthesize"],
                },
                "description": "List of operations with metadata and full configuration for original operations",
            }
        },
        "required": ["operations"],
    }

    # Get initial response with pipeline ops list, with up to 2 retries for incorrect order
    max_attempts = 3
    for attempt in range(max_attempts):
        response = llm_client.generate_rewrite(
            [
                {
                    "role": "user",
                    "content": (
                        initial_prompt
                        if attempt == 0
                        else generate_rewrite_prompt(
                            candidate,
                            pipeline_config,
                            expected_ops=execution_order,
                            sample_docs=sample_docs,
                        )
                    ),
                }
            ],
            system_prompt,
            pipeline_ops_schema,
        )

        result = json.loads(response.choices[0].message.content)
        operations = result["operations"]

        # Validate operations match the expected execution order
        has_mismatch = False

        # Check length first
        if len(operations) != len(execution_order):
            console.print(
                f"[bold red]Warning:[/bold red] LLM returned {len(operations)} operations, but skeleton has {len(execution_order)} processing operations"
            )
            has_mismatch = True

        # Check operation types match in order
        for i, (llm_op, expected_op) in enumerate(zip(operations, execution_order)):
            if llm_op["type"] != expected_op["type"]:
                console.print(
                    f"[bold red]Warning:[/bold red] Operation {i+1} type mismatch: LLM returned '{llm_op['type']}', expected '{expected_op['type']}'"
                )
                has_mismatch = True

            # Check if synthesize flag matches expected
            if llm_op["synthesize"] != expected_op["synthesized"]:
                console.print(
                    f"[bold red]Warning:[/bold red] Operation {i+1} synthesize mismatch: LLM returned '{llm_op['synthesize']}', expected '{expected_op['synthesized']}'"
                )
                # Force correct synthesis status based on skeleton
                llm_op["synthesize"] = expected_op["synthesized"]

        # Break if everything matches, otherwise retry
        if not has_mismatch or attempt == max_attempts - 1:
            break

        console.print(
            f"[bold yellow]Attempt {attempt+1}/{max_attempts}:[/bold yellow] Reprompting LLM for correct operation order"
        )

    # Print operations to console in a nice format
    console.print("\n[bold blue]Pipeline Operations:[/bold blue]")
    console.print("=" * 80)

    for i, op in enumerate(operations):
        synthesis_status = (
            "[bold green]Original[/bold green]"
            if not op["synthesize"]
            else "[bold yellow]Synthesized[/bold yellow]"
        )
        original_info = f" (from: {op['original_op']})" if op.get("original_op") else ""

        console.print(
            f"[bold]{i+1}.[/bold] [cyan]{op['name']}[/cyan] - Type: [magenta]{op['type']}[/magenta] - {synthesis_status}{original_info}"
        )

    console.print("=" * 80)

    # Get mapping of original operations from pipeline config for any special operations like scan/step_boundary
    original_ops = {}
    if "ops" in pipeline_config:
        for op in pipeline_config["ops"]:
            original_ops[op["name"]] = op

    # Step 2: Iteratively build configurations for each operation
    complete_pipeline = []
    conversation = [
        {"role": "user", "content": initial_prompt},
        {"role": "assistant", "content": response.choices[0].message.content},
    ]

    # Define parameters schema for operation configuration
    op_config_schema = {
        "type": "object",
        "properties": {"op_config": {"type": "object", "additionalProperties": True}},
        "required": ["op_config"],
    }

    # First process any scan or step_boundary operations that come before processing operations
    for op_info in all_execution_order:
        if op_info["type"] in ["scan", "step_boundary"]:
            # Find this operation in the original config
            if op_info["original_op"] and hasattr(op_info["original_op"], "config"):
                original_name = op_info["original_op"].config.get("name")
                if original_name and original_name in original_ops:
                    complete_pipeline.append(original_ops[original_name])
                    console.print(
                        f"[bold green]Adding Initial Operation:[/bold green] [cyan]{original_name}[/cyan]"
                    )
        else:
            # Once we hit a processing operation, break out
            break

    # Process each processing operation in the pipeline
    for i, op_info in enumerate(operations):
        op_name = op_info["name"]
        op_type = op_info["type"]
        needs_synthesis = op_info["synthesize"]
        original_op_name = op_info.get("original_op")

        # Debug info
        console.print(
            f"[bold blue]Processing operation:[/bold blue] {op_name} (type: {op_type}, synthesis: {needs_synthesis})"
        )

        # Use the config provided by the LLM if available
        if not needs_synthesis and op_info.get("config"):
            op_config = op_info["config"]
            console.print(
                f"[bold green]Using original config:[/bold green] [cyan]{op_name}[/cyan]"
            )
            complete_pipeline.append(op_config)
            continue

        # Check if this is the final operation in a rewrite pattern
        is_final_op = False
        if original_op_name and i == len(operations) - 1:
            is_final_op = True

        # If operation doesn't need synthesis but no config was provided, try to find it in the original pipeline
        if not needs_synthesis:
            console.print(
                f"[bold yellow]Warning:[/bold yellow] LLM did not provide config for non-synthesized operation {op_name} - trying to find it"
            )

            # Try to find the operation in the original config
            if original_op_name and original_op_name in original_ops:
                op_config = original_ops[original_op_name]
                console.print(
                    f"[bold green]Found original operation:[/bold green] [cyan]{original_op_name}[/cyan]"
                )
                complete_pipeline.append(op_config)
                continue
            elif op_name in original_ops:
                op_config = original_ops[op_name]
                console.print(
                    f"[bold green]Found original operation by name:[/bold green] [cyan]{op_name}[/cyan]"
                )
                complete_pipeline.append(op_config)
                continue
            else:
                console.print(
                    f"[bold red]Warning:[/bold red] Could not find original config for {op_name} - will synthesize instead"
                )
                # Continue to synthesize

        # We need to generate a new configuration
        op_prompt = generate_op_config_prompt(
            op_name=op_name,
            op_type=op_type,
            original_op_name=original_op_name,
            pipeline_config=pipeline_config,
            previous_ops=complete_pipeline,
            is_final_op=is_final_op,
            sample_docs=sample_docs,
        )

        # Add to conversation and get response
        conversation.append({"role": "user", "content": op_prompt})
        response = llm_client.generate_rewrite(
            conversation, system_prompt, op_config_schema
        )

        # Parse the response and add to the complete pipeline
        op_result = json.loads(response.choices[0].message.content)
        op_config = op_result["op_config"]

        # Ensure the configuration has the correct name and type
        op_config["name"] = op_name
        op_config["type"] = op_type  # Force the correct type from the schema

        complete_pipeline.append(op_config)

        # Pretty print the synthesized operation to the console
        console.print(
            f"\n[bold green]Synthesized Operation:[/bold green] [cyan]{op_name}[/cyan]"
        )
        console.print("-" * 80)
        formatted_config = json.dumps(op_config, indent=2)
        console.print(formatted_config)
        console.print("-" * 80)

        # Add the response to the conversation history
        conversation.append(
            {"role": "assistant", "content": response.choices[0].message.content}
        )

    # Process any scan or step_boundary operations that come after processing operations
    found_operations = set(op["name"] for op in complete_pipeline)
    for op_info in all_execution_order:
        if op_info["type"] in ["scan", "step_boundary"]:
            # Find this operation in the original config
            if op_info["original_op"] and hasattr(op_info["original_op"], "config"):
                original_name = op_info["original_op"].config.get("name")
                if (
                    original_name
                    and original_name in original_ops
                    and original_name not in found_operations
                ):
                    complete_pipeline.append(original_ops[original_name])
                    console.print(
                        f"[bold green]Adding Final Operation:[/bold green] [cyan]{original_name}[/cyan]"
                    )

    return complete_pipeline
