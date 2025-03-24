import copy
import json
from itertools import product

import yaml

from docetl.containers import OpContainer
from docetl.utils import extract_jinja_variables


def generate_rewrite_prompt(
    pipeline_config, execution_order, expected_ops=None, sample_docs=None
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
        execution_order: Pre-calculated execution order (optional)
        expected_ops: Optional list of expected operations in execution order
        sample_docs: Optional sample documents for context
    """
    # Convert pipeline_config to YAML.
    pipeline_yaml = yaml.dump(pipeline_config, default_flow_style=False)

    # Format the execution order into a readable string
    execution_order_str = "Execution Order (First to Last):\n"
    for depth, node_str in execution_order:
        indent = "  " * depth
        execution_order_str += f"{indent}→ {node_str}\n"

    # Add explanation for map* if it appears in the execution order
    if any("map*" in node_str for _, node_str in execution_order):
        execution_order_str += "\nNOTE: When you see 'map*' in the execution order, this indicates that 2+ sequential map operations will be synthesized in place of a single map operation. This breaks down the original map operation into multiple steps that build on each other. Make sure to return 'map*' as the type for this special case.\n"

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

4. Split Operation Example (no output schema needed):
```yaml
- name: split_transcript
  type: split
  split_key: transcript
  method: token_count
  method_kwargs:
    num_tokens: 500
    model: gpt-4o-mini
```

5. Gather Operation Example (no output schema needed):
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
     (Breaks a map into a chain of 2+ map operations for iterative refinement. Still return map* as the type.)
  4. Decomposition: {{"pattern": ["map"], "skeleton": ["parallel_map", "map"]}}
     (Splits the map operation into independent parallel subtasks whose outputs are unified later.)

Please respond with an array of operations, each containing:
1. name: A descriptive name for the operation
2. type: The operation type (e.g., split, gather, map, reduce, parallel_map, map*)
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
    skeleton_metadata=None,
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
        skeleton_metadata: Optional dictionary containing op_skeleton_metadata and instantiated_rewrite
    """

    # Find the original operation if it exists
    original_op = None
    if original_op_name and "ops" in pipeline_config:
        for op in pipeline_config["ops"]:
            if op["name"] == original_op_name:
                original_op = op
                break

    if original_op is None and "operations" in pipeline_config:
        for op in pipeline_config["operations"]:
            if op["name"] == original_op_name:
                original_op = op
                break

    original_op_info = ""
    if original_op:
        original_op_info = f"\nThis operation is derived from: {original_op_name}"
        if is_final_op:
            original_op_info += f"\nThis is the final operation in a rewrite pattern. You should match the output schema of the original operation: {json.dumps(original_op.get('output', {}))}"

    # Add rewrite hints if available
    rewrite_hints = ""
    if skeleton_metadata:
        op_skeleton = skeleton_metadata.get("op_skeleton_metadata")
        rewrite = skeleton_metadata.get("instantiated_rewrite")

        if op_skeleton:
            rewrite_hints += "\n\nREWRITE HINTS FOR THIS OPERATION:\n"

            # Add purpose hint if available
            if hasattr(op_skeleton, "purpose") and op_skeleton.purpose:
                rewrite_hints += f"Purpose: {op_skeleton.purpose}\n"

            # Add implementation hints if available
            if (
                hasattr(op_skeleton, "implementation_hints")
                and op_skeleton.implementation_hints
            ):
                rewrite_hints += "Implementation Hints:\n"
                for hint in op_skeleton.implementation_hints:
                    rewrite_hints += f"- {hint}\n"

            # Add input/output fields if available
            if hasattr(op_skeleton, "input_fields") and op_skeleton.input_fields:
                rewrite_hints += "Input Fields to Reference:\n"
                for field in op_skeleton.input_fields:
                    rewrite_hints += f"- {field}\n"

            if hasattr(op_skeleton, "output_fields") and op_skeleton.output_fields:
                rewrite_hints += "Output Fields to Generate:\n"
                for field in op_skeleton.output_fields:
                    rewrite_hints += f"- {field}\n"

        # Add rewrite pattern information
        if rewrite:
            if hasattr(rewrite, "decomposition") and hasattr(
                rewrite.decomposition, "pattern"
            ):
                pattern = rewrite.decomposition.pattern
                skeleton = [op.op_type for op in rewrite.decomposition.skeleton]

                rewrite_hints += "\nThis operation is part of a rewrite pattern:\n"
                rewrite_hints += f"Original Pattern: {', '.join(pattern)}\n"
                rewrite_hints += f"Rewritten as: {', '.join(skeleton)}\n"

                # Find position in skeleton
                if hasattr(op_skeleton, "op_type") and op_skeleton.op_type == op_type:
                    position = (
                        skeleton.index(op_type) + 1
                        if op_type in skeleton
                        else "unknown"
                    )
                    total = len(skeleton)
                    rewrite_hints += f"This operation is #{position} of {total} in the rewrite pattern.\n"

            # Add specific rewrite description if available
            if hasattr(rewrite, "description") and rewrite.description:
                rewrite_hints += f"\nRewrite Description: {rewrite.description}\n"

    # Format previous operations for context
    previous_ops_info = ""
    if previous_ops and len(previous_ops) > 0:
        previous_ops_info = "\n\nPrevious Operations Context:\n"

        # First, display the operations in their execution order
        previous_ops_info += "\nExecution Order (First to Current):\n"
        for i, prev_op_list in enumerate(previous_ops):  # Show all operations in order
            prev_op = prev_op_list[-1]
            prev_op_name = prev_op.get("name", "unknown")
            prev_op_type = prev_op.get("type", "unknown")

            previous_ops_info += f"{i+1}. {prev_op_name} (Type: {prev_op_type})\n"

        # Then show more details about the most recent operations
        previous_ops_info += "\nRecent Operations Details:\n"
        for i, prev_op_list in enumerate(
            previous_ops[-3:]
        ):  # Show details of last 3 operations
            prev_op = prev_op_list[-1]
            # Include basic operation information
            prev_op_name = prev_op.get("name", "unknown")
            prev_op_type = prev_op.get("type", "unknown")

            previous_ops_info += f"\nOperation: {prev_op_name} (Type: {prev_op_type})\n"

            # Include output schema if available
            if "output" in prev_op and "schema" in prev_op["output"]:
                previous_ops_info += "Output Schema:\n"
                for field, field_type in prev_op["output"]["schema"].items():
                    previous_ops_info += f"  - {field}: {field_type}\n"

            # Include specific fields for different operation types
            if prev_op_type == "split":
                previous_ops_info += (
                    f"Split Key: {prev_op.get('split_key', 'unknown')}\n"
                )
            elif prev_op_type == "gather":
                previous_ops_info += (
                    f"Content Key: {prev_op.get('content_key', 'unknown')}\n"
                )
                previous_ops_info += (
                    f"Doc ID Key: {prev_op.get('doc_id_key', 'unknown')}\n"
                )
                previous_ops_info += (
                    f"Order Key: {prev_op.get('order_key', 'unknown')}\n"
                )

            # Include prompts for map operations (summarized if too long)
            if "prompt" in prev_op:
                prompt = prev_op["prompt"]
                if len(prompt) > 1000:
                    prompt = prompt[:1000] + "...[truncated]"
                previous_ops_info += f"Prompt: {prompt}\n"

        # Finally, add pattern detection information if applicable
        # Identify common patterns
        last_3_types = [
            op_list[-1].get("type", "unknown")
            for op_list in previous_ops[-3:]
            if op_list[-1].get("type", "unknown") != "step_boundary"
        ]

        # Check if this is part of a split-gather-map pattern
        if "split" in last_3_types and "gather" in last_3_types and op_type == "map":
            previous_ops_info += "\nIMPORTANT - PATTERN DETECTED: Split-Gather-Map\n"
            previous_ops_info += "This is a common pattern for processing large documents by splitting them into chunks, gathering context, and then analyzing the chunks with context.\n"

            # Find the gather operation (which should be the most recent)
            for p_op_list in reversed(previous_ops):
                p_op = p_op_list[-1]
                if p_op.get("type") == "gather":
                    content_key = p_op.get("content_key", "unknown")
                    previous_ops_info += f"\nCRITICAL INSTRUCTION: You MUST reference the gathered content as '{{{{ input.{content_key}_chunk_rendered }}}}' in your prompt.\n"
                    previous_ops_info += f"DO NOT use '{{{{ input.{content_key} }}}}' as this will not include the context from surrounding chunks.\n"
                    previous_ops_info += f"The '{content_key}_chunk_rendered' field contains the main content plus context from other chunks.\n"
                    break

        # Check if this is part of a map* chain
        elif op_type == "map" and any(
            p_op_list[-1].get("type") == "map" for p_op_list in previous_ops[-2:]
        ):
            previous_ops_info += (
                "\nIMPORTANT - PATTERN DETECTED: Sequential Map Chain\n"
            )
            previous_ops_info += "This operation should build upon the outputs of the previous map operations.\n"

            # Find the most recent map operation
            for p_op_list in reversed(previous_ops):
                p_op = p_op_list[-1]
                if (
                    p_op.get("type") == "map"
                    and "output" in p_op
                    and "schema" in p_op["output"]
                ):
                    previous_ops_info += f"\nYou should reference outputs from the previous map operation ('{p_op.get('name')}') such as:\n"
                    for field in p_op["output"]["schema"].keys():
                        previous_ops_info += f"- {{{{ input.{field} }}}}\n"
                    break

        previous_ops_info += "\nYour operation should work with the outputs of these previous operations.\n"

    # Format sample documents for context
    sample_docs_info = ""
    if sample_docs and len(sample_docs) > 0:
        sample_docs_info = "\n\nSample Documents for Context:\n"
        # Choose a representative document
        sample_doc = sample_docs[0]
        sample_docs_info += "Document Structure:\n"

        # Show the structure of the document (keys and value types)
        for key, value in sample_doc.items():
            value_type = type(value).__name__
            value_preview = str(value)
            if len(value_preview) > 50:
                value_preview = value_preview[:50] + "...[truncated]"
            sample_docs_info += f"  - {key} ({value_type}): {value_preview}\n"

        # Show more detailed first few documents if they're small
        if len(str(sample_doc)) < 1000:
            sample_docs_info += "\nSample Document (complete):\n"
            sample_docs_info += json.dumps(sample_doc, indent=2)

        # Add tips based on the document structure
        sample_docs_info += "\n\nWhen designing your operation, consider how to handle these document fields appropriately.\n"

    type_specific_instructions = ""
    example_config = ""

    if op_type == "map*":
        type_specific_instructions = """
For a map* operation, you need to generate 2+ sequential map operations that build on each other.
Each operation should:
1. Have a clear, specific name describing its subtask
2. Include appropriate prompt templates using Jinja syntax
3. Have a suitable output schema
4. Build on the output of the previous operation in the sequence

Your response should be a LIST of operations, not just one.
"""
        example_config = """
Example of a map* Operation (which generates multiple map operations):
```json
{
  "op_configs": [
    {
      "name": "extract_raw_entities",
      "type": "map",
      "prompt": "Extract all named entities from the following text: {{ input.text }}\\n\\nList each entity with its type (person, organization, location, etc.)",
      "output": {
        "schema": {
          "entities": "list[object]"
        }
      }
    },
    {
      "name": "enrich_entities",
      "type": "map",
      "prompt": "Enrich the following entities with additional information:\\n{{ input.entities }}\\n\\nFor each entity, add a brief description and relevance score (1-10).",
      "output": {
        "schema": {
          "enriched_entities": "list[object]"
        }
      }
    },
    {
      "name": "categorize_entities",
      "type": "map",
      "prompt": "Categorize the following enriched entities into logical groups:\\n{{ input.enriched_entities }}\\n\\nProvide group names and explain your categorization logic.",
      "output": {
        "schema": {
          "entity_categories": "list[object]",
          "categorization_logic": "string"
        }
      }
    }
  ]
}
```

Each map operation in the sequence should:
1. Process the output of the previous operation
2. Have a focused, specific task
3. Include clear output schema that will be used by the next operation
"""
    elif op_type == "reduce":
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

If this map operation follows a gather operation:
- You MUST reference the gathered content with '_chunk_rendered' suffix
  * Example: If the content_key in gather was 'transcript', use {{ input.transcript_chunk_rendered }}
  * NOT just {{ input.transcript }}
- The _chunk_rendered version includes the surrounding context from other chunks
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
  * This MUST be one of the keys present in the document (see sample document structure)
  * It should typically be a content field referred to in the original operation prompt
  * Choose the key containing the main text content that needs to be processed in chunks
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
  * This MUST match the split_key used in the preceding split operation
- doc_id_key (ID field from split operation)
  * This MUST be 'split_{split_op_name}_id' matching the ID field from the split operation
- order_key (chunk order field from split)
  * This MUST be 'split_{split_op_name}_chunk_num' matching the chunk number field from the split operation
- peripheral_chunks (how to gather context from previous/next chunks)
  * This defines how many neighboring chunks to include for context

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
Type: {op_type}{original_op_info}{rewrite_hints}

Based on the information from our conversation, generate a complete configuration for this operation.
Your response should be a valid JSON object containing the complete configuration.
{previous_ops_info}{sample_docs_info}

{example_config}

IMPORTANT:
- The operation type MUST be "{op_type}" - do not use "unknown" or any other type.
- Output schema types should be "string" for all fields{"" if is_final_op else " (use simple string types)"}.
- Your configuration should work well with the previous operations in the pipeline.
- Use the sample document structure to ensure your operation processes the correct fields.
{type_specific_instructions}
"""

    # print(prompt)  # Debug print, removing to clean up output
    return prompt


def invoke_rewrite_agent(
    candidate,
    pipeline_config,
    op_names_to_configs,
    llm_client,
    sample_docs,
    console,
    runner=None,
):
    """
    Given a candidate skeleton and the pipeline configuration, iteratively build
    configurations for all operations in the pipeline.

    Args:
        candidate: The candidate skeleton with operations to synthesize
        pipeline_config: Original pipeline configuration
        op_names_to_configs: A dictionary mapping operation names to their configurations
        llm_client: The LLM client for generating configurations
        sample_docs: Optional sample documents for context (default: None)
        console: The console for logging
        runner: The DSLRunner instance for syntax checking (default: None)

    Returns:
        List of operation configurations in execution order
    """
    # Create a visual separator for the beginning of the pipeline rewrite process
    console.rule("[bold]Pipeline Rewrite Process[/bold]")

    # Create reusable panel styles
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    # Define consistent panel styles
    INFO_STYLE = "blue"
    SUCCESS_STYLE = "green"
    WARNING_STYLE = "yellow"
    ERROR_STYLE = "red"

    def create_panel(title, content, style=INFO_STYLE):
        """Helper function to create consistently styled panels"""
        return Panel(content, title=title, border_style=style, expand=False)

    # Print the candidate skeleton in a panel
    console.print(
        create_panel(
            "Candidate Skeleton",
            Text(str(candidate), style="white"),
        )
    )

    # Generate an execution-order list for validation and formatted execution order for prompts
    execution_order = []
    formatted_execution_order = []

    def collect_execution_order(node, depth=0):
        """Recursively collect nodes in execution order (depth-first)"""
        # Process children first (since the tree is reversed)
        for child in node.children:
            collect_execution_order(child, depth + 1)

        # Then process the current node
        # Skip non-processing operations like scan and step_boundary for the LLM
        if node.op_type not in ["scan", "step_boundary"]:
            # Extract metadata from the skeleton node
            skeleton_metadata = None
            if hasattr(node, "op_skeleton_metadata") or hasattr(
                node, "instantiated_rewrite"
            ):
                skeleton_metadata = {
                    "op_skeleton_metadata": (
                        node.op_skeleton_metadata
                        if hasattr(node, "op_skeleton_metadata")
                        else None
                    ),
                    "instantiated_rewrite": (
                        node.instantiated_rewrite
                        if hasattr(node, "instantiated_rewrite")
                        else None
                    ),
                }

            # Add to execution order for validation
            execution_order.append(
                {
                    "type": node.op_type,
                    "synthesized": hasattr(node, "synthesized") and node.synthesized,
                    "original_op": (
                        node.original_op
                        if hasattr(node, "original_op") and node.original_op
                        else None
                    ),
                    "skeleton_metadata": skeleton_metadata,
                }
            )

            # Add to formatted execution order for prompt
            tag = (
                " (synth)" if hasattr(node, "synthesized") and node.synthesized else ""
            )
            orig_op = (
                f"orig:{node.original_op.config.get('type')} named {node.original_op.config.get('name')}"
                if hasattr(node, "original_op") and node.original_op
                else "None"
            )
            formatted_execution_order.append((depth, f"{node.op_type}{tag}[{orig_op}]"))

        # But keep track of ALL operations for our final pipeline
        # Extract metadata from the skeleton node for all operations
        skeleton_metadata = None
        if hasattr(node, "op_skeleton_metadata") or hasattr(
            node, "instantiated_rewrite"
        ):
            skeleton_metadata = {
                "op_skeleton_metadata": (
                    node.op_skeleton_metadata
                    if hasattr(node, "op_skeleton_metadata")
                    else None
                ),
                "instantiated_rewrite": (
                    node.instantiated_rewrite
                    if hasattr(node, "instantiated_rewrite")
                    else None
                ),
            }

        all_execution_order.append(
            {
                "type": node.op_type,
                "synthesized": hasattr(node, "synthesized") and node.synthesized,
                "original_op": (
                    node.original_op
                    if hasattr(node, "original_op") and node.original_op
                    else None
                ),
                "skeleton_metadata": skeleton_metadata,
            }
        )

    # Keep track of ALL operations including scan/step_boundary
    all_execution_order = []

    # Traverse the skeleton to get execution order
    collect_execution_order(candidate)

    # Create an execution order table
    execution_table = Table(title=None, box=None, expand=True)
    execution_table.add_column("#", style="bold")
    execution_table.add_column("Operation Type", style="magenta")
    execution_table.add_column("Status", style="bold")
    execution_table.add_column("Original Operation", style="dim")

    for i, op in enumerate(execution_order):
        synth_status = (
            "[yellow]Synthesized[/yellow]"
            if op["synthesized"]
            else "[green]Original[/green]"
        )
        original_op_info = ""
        if op["original_op"] and hasattr(op["original_op"], "config"):
            original_op_info = f"{op['original_op'].config.get('name')}"

        execution_table.add_row(str(i + 1), op["type"], synth_status, original_op_info)

    # Print execution order in a panel
    console.print(create_panel("Expected Execution Order", execution_table))

    # Step 1: Get the ordered list of operations with metadata from the LLM
    initial_prompt = generate_rewrite_prompt(
        pipeline_config,
        execution_order=formatted_execution_order,
        sample_docs=sample_docs,
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
                        "original_op_name": {"type": "string"},
                    },
                    "required": ["name", "type", "synthesize", "original_op_name"],
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
                            pipeline_config,
                            execution_order=formatted_execution_order,
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
        mismatch_messages = []

        # Check length first
        if len(operations) != len(execution_order):
            mismatch_messages.append(
                f"LLM returned {len(operations)} operations, but skeleton has {len(execution_order)} processing operations"
            )
            has_mismatch = True

        # Check operation types match in order
        for i, (llm_op, expected_op) in enumerate(zip(operations, execution_order)):
            if llm_op["type"] != expected_op["type"]:
                mismatch_messages.append(
                    f"Operation {i+1} type mismatch: LLM returned '{llm_op['type']}', expected '{expected_op['type']}'"
                )
                has_mismatch = True

            # Check if synthesize flag matches expected
            if llm_op["synthesize"] != expected_op["synthesized"]:
                mismatch_messages.append(
                    f"Operation {i+1} synthesize mismatch: LLM returned '{llm_op['synthesize']}', expected '{expected_op['synthesized']}'"
                )
                # Force correct synthesis status based on skeleton
                llm_op["synthesize"] = expected_op["synthesized"]

            # Add the skeleton node to the operation
            llm_op["skeleton_metadata"] = expected_op["skeleton_metadata"]

        # Break if everything matches, otherwise retry
        if not has_mismatch or attempt == max_attempts - 1:
            break

        # Show mismatch warnings in a panel
        warning_text = Text(
            f"Attempt {attempt+1}/{max_attempts}: Reprompting LLM for correct operation order\n\n"
        )
        for msg in mismatch_messages:
            warning_text.append("• " + msg + "\n", style="bold yellow")

        console.print(
            create_panel("Operation Order Mismatch", warning_text, style=WARNING_STYLE)
        )

    # Create operations table
    pipeline_table = Table(title=None, box=None, expand=True)
    pipeline_table.add_column("#", style="bold")
    pipeline_table.add_column("Operation Name", style="cyan")
    pipeline_table.add_column("Type", style="magenta")
    pipeline_table.add_column("Status", style="bold")
    pipeline_table.add_column("Original Operation", style="dim")

    for i, op in enumerate(operations):
        synthesis_status = (
            "[yellow]Synthesized[/yellow]"
            if op["synthesize"]
            else "[green]Original[/green]"
        )
        original_info = op.get("original_op_name", "")

        pipeline_table.add_row(
            str(i + 1), op["name"], op["type"], synthesis_status, original_info
        )

    # Print the pipeline operations in a panel
    console.print(create_panel("Pipeline Operations", pipeline_table))

    # Get mapping of original operations from pipeline config for any special operations like scan/step_boundary
    original_ops = {}
    if "ops" in pipeline_config:
        for op in pipeline_config["ops"]:
            original_ops[op["name"]] = op

    # Step 2: Iteratively build configurations for each operation
    # Modified: complete_pipeline will now contain lists of alternative configs for each position
    complete_pipeline = []
    conversation = [
        {"role": "user", "content": initial_prompt},
        {"role": "assistant", "content": response.choices[0].message.content},
    ]

    # Define parameters schema for operation configuration
    op_config_schema = {
        "type": "object",
        "properties": {
            "op_config": {"type": "object", "additionalProperties": True},
            "op_configs": {
                "type": "array",
                "items": {"type": "object", "additionalProperties": True},
                "description": "List of operation configurations for map* operations",
            },
        },
        "anyOf": [{"required": ["op_config"]}, {"required": ["op_configs"]}],
        "description": "Either a single operation configuration or a list of operation configurations for map*",
    }

    # Process section panel and progress
    progress_panel = None

    # First process any scan or step_boundary operations that come before processing operations
    initial_ops = []
    for op_info in all_execution_order:
        if op_info["type"] in ["scan", "step_boundary"]:
            # Find this operation in the original config
            if op_info["original_op"] and hasattr(op_info["original_op"], "config"):
                original_name = op_info["original_op"].config.get("name")
                if original_name and original_name in original_ops:
                    # Modified: Add as a list with one item
                    complete_pipeline.append([original_ops[original_name]])
                    initial_ops.append(original_name)
        else:
            # Once we hit a processing operation, break out
            break

    if initial_ops:
        console.print(
            create_panel(
                "Initial Operations",
                Text("Adding initial operations: " + ", ".join(initial_ops)),
                style=SUCCESS_STYLE,
            )
        )

    # Get model choices from runner if available, otherwise use default
    model_choices = runner.optimizer.model_choices

    # Process each processing operation in the pipeline
    for i, op_info in enumerate(operations):
        op_name = op_info["name"]
        op_type = op_info["type"]
        needs_synthesis = op_info["synthesize"]
        original_op_name = op_info.get("original_op_name")

        # Create operation status panel
        op_status_text = Text()
        op_status_text.append("Operation: ", style="bold")
        op_status_text.append(op_name, style="cyan bold")
        op_status_text.append("\nType: ", style="bold")
        op_status_text.append(op_type, style="magenta")
        op_status_text.append("\nSynthesis: ", style="bold")
        op_status_text.append(
            str(needs_synthesis), style="yellow" if needs_synthesis else "green"
        )

        if original_op_name:
            op_status_text.append("\nDerived from: ", style="bold")
            op_status_text.append(original_op_name, style="dim")

        progress_panel = create_panel(
            f"Processing Operation ({i+1}/{len(operations)})", op_status_text
        )
        console.print(progress_panel)

        # Use the config from op_names_to_configs
        if not needs_synthesis and op_name in op_names_to_configs:
            op_config = op_names_to_configs[op_name]
            # Modified: Create variations for original config based on model choices
            op_configs = []

            # Always include the original configuration
            op_configs.append(op_config)

            # For map and reduce operations, create model variations
            if op_type in ["map", "reduce"]:
                original_model = op_config.get(
                    "model",
                    (
                        runner.optimizer.default_model
                        if runner and hasattr(runner, "optimizer")
                        else "gpt-4o-mini"
                    ),
                )
                for model in model_choices:
                    if model != original_model:
                        model_config = copy.deepcopy(op_config)
                        model_config["model"] = model
                        op_configs.append(model_config)

            # Add the list of configs to the pipeline
            complete_pipeline.append(op_configs)

            console.print(
                create_panel(
                    "Using Original Configuration with Variations",
                    Text(
                        f"{op_name} - Generated {len(op_configs)} variations based on model choices",
                        style="cyan bold",
                    ),
                    style=SUCCESS_STYLE,
                )
            )
            continue

        # Check if this is the final operation in a rewrite pattern
        is_final_op = False
        if original_op_name and i == len(operations) - 1:
            is_final_op = True

        # If operation doesn't need synthesis but no config was provided, try to find it in the original pipeline
        if not needs_synthesis:
            warning_text = Text(
                f"Could not find original config for {op_name} - will synthesize instead"
            )
            console.print(
                create_panel("Configuration Warning", warning_text, style=WARNING_STYLE)
            )
            # Continue to synthesize

        # If the operation is a split, gather or sample, we'll create multiple configurations
        if op_type == "split":
            # Find the longest content field in the document that is also in the original operation prompt
            keys_in_map_prompt = extract_jinja_variables(
                op_names_to_configs[original_op_name]["prompt"]
            )
            keys_in_map_prompt = list(
                set(k.replace("input.", "") for k in keys_in_map_prompt)
            )
            longest_key = max(
                {
                    k: len(v)
                    for k, v in sample_docs[0].items()
                    if k in keys_in_map_prompt
                },
                key=lambda x: x[1],
            )
            avg_longest_key_length = sum(
                len(sample_docs[i][longest_key]) for i in range(len(sample_docs))
            ) / len(sample_docs)

            # Create multiple split configurations with different token counts
            split_configs = []

            # Generate multiple split configurations with different settings
            token_counts = [
                max(5, int((avg_longest_key_length / 4) * 0.1)),  # Smaller chunks
                max(
                    5, int((avg_longest_key_length / 4) * 0.3)
                ),  # Medium chunks (default)
                max(5, int((avg_longest_key_length / 4) * 0.5)),  # Larger chunks
                max(5, int((avg_longest_key_length / 4) * 0.7)),  # Larger chunks
            ]
            token_counts = list(set(token_counts))

            for token_count in token_counts:
                split_config = {
                    "name": op_name,
                    "type": op_type,
                    "split_key": longest_key,
                    "method": "token_count",
                    "method_kwargs": {
                        "num_tokens": token_count,
                    },
                }
                split_configs.append(split_config)

            # Add the split configurations to the pipeline
            complete_pipeline.append(split_configs)

            # Create a formatted table of configurations
            split_table = Table(title=None, box=None, expand=True)
            split_table.add_column("#", style="bold")
            split_table.add_column("Configuration", style="cyan")
            split_table.add_column("Token Count", style="magenta")

            for j, cfg in enumerate(split_configs):
                split_table.add_row(
                    str(j + 1),
                    f"{cfg['name']} (split_key: {cfg['split_key']})",
                    str(cfg["method_kwargs"]["num_tokens"]),
                )

            console.print(
                create_panel(
                    "Generated Split Configurations", split_table, style=SUCCESS_STYLE
                )
            )
            continue

        # If the operation is a gather, create multiple gather configurations
        if op_type == "gather":
            # Look for the most recent split operation
            split_op_configs = None
            split_op_name = None
            for prev_configs in reversed(complete_pipeline):
                if prev_configs[0]["type"] == "split":
                    split_op_configs = prev_configs
                    split_op_name = prev_configs[0]["name"]
                    break

            if not split_op_configs:
                raise ValueError(
                    "Could not find a preceding split operation for gather. A gather operation must follow a split operation in the pipeline."
                )

            split_key = split_op_configs[0]["split_key"]

            # Create multiple gather configurations with different context settings
            gather_configs = []
            context_settings = []

            # If there was a preceding map, use the summary key as the previous content_key
            for prev_configs in reversed(complete_pipeline):
                if (
                    prev_configs[0]["type"] == "map"
                    and prev_configs[0] != split_op_configs[0]
                ):
                    if (
                        "output" in prev_configs[0]
                        and "schema" in prev_configs[0]["output"]
                    ):
                        summary_key = list(prev_configs[0]["output"]["schema"].keys())[
                            0
                        ]
                        context_settings.extend(
                            [
                                {
                                    "previous": {"content_key": summary_key},
                                },
                                {
                                    "previous": {
                                        "head": {"content_key": split_key},
                                        "middle": {"content_key": summary_key},
                                    },
                                },
                            ]
                        )

            # Generate multiple gather configurations with different context settings
            if len(context_settings) == 0:
                context_settings = [
                    {},
                    {
                        "previous": {
                            "count": 1,
                            "content_key": split_key,
                        }
                    },  # No context
                    {
                        "previous": {"count": 1, "content_key": split_key},
                        "next": {"count": 1, "content_key": split_key},
                    },  # 1 previous, 1 next
                    {
                        "previous": {
                            "head": {
                                "count": 1,
                                "content_key": split_key,
                            },
                            "tail": {
                                "count": 1,
                                "content_key": split_key,
                            },
                        }
                    },  # 1 head, 1 tail of previous
                ]

            for context in context_settings:
                gather_config = {
                    "name": op_name,
                    "type": op_type,
                    "doc_id_key": f"{split_op_name}_id",
                    "order_key": f"{split_op_name}_chunk_num",
                    "content_key": split_key,
                    "peripheral_chunks": context,
                }

                gather_configs.append(gather_config)

            # Add the gather configurations to the pipeline
            complete_pipeline.append(gather_configs)

            # Create a formatted table of configurations
            gather_table = Table(title=None, box=None, expand=True)
            gather_table.add_column("#", style="bold")
            gather_table.add_column("Configuration", style="cyan")
            gather_table.add_column("Context", style="magenta")

            for j, cfg in enumerate(gather_configs):
                context_info = []

                # Handle previous chunks context
                if "previous" in cfg["peripheral_chunks"]:
                    if isinstance(cfg["peripheral_chunks"]["previous"], dict):
                        if "count" in cfg["peripheral_chunks"]["previous"]:
                            prev = cfg["peripheral_chunks"]["previous"]["count"]
                            context_info.append(f"Previous: {prev}")
                        elif (
                            "head" in cfg["peripheral_chunks"]["previous"]
                            and "tail" in cfg["peripheral_chunks"]["previous"]
                        ):
                            head = cfg["peripheral_chunks"]["previous"]["head"]["count"]
                            tail = cfg["peripheral_chunks"]["previous"]["tail"]["count"]
                            context_info.append(f"Head: {head}, Tail: {tail}")

                # Handle next chunks context
                if "next" in cfg["peripheral_chunks"]:
                    next_count = cfg["peripheral_chunks"]["next"]["count"]
                    context_info.append(f"Next: {next_count}")

                # Join context information or use "None" if empty
                context_str = ", ".join(context_info) if context_info else "None"

                gather_table.add_row(
                    str(j + 1),
                    f"{cfg['name']} (content: {cfg['content_key']})",
                    context_str,
                )

            console.print(
                create_panel(
                    "Generated Gather Configurations", gather_table, style=SUCCESS_STYLE
                )
            )
            continue

        # If the operation is a sample, create multiple sample configurations
        if op_type == "sample":
            # Look for the preceding gather operation to get the doc_id_key
            gather_op_configs = None
            for prev_configs in reversed(complete_pipeline):
                if prev_configs[0]["type"] == "gather":
                    gather_op_configs = prev_configs
                    break

            doc_id_key = (
                gather_op_configs[0]["doc_id_key"] if gather_op_configs else "id"
            )

            # Create multiple sample configurations with different sampling rates
            sample_configs = []

            # Generate multiple sample configurations with different sampling rates
            sampling_rates = [0.05, 0.2, 0.5]  # 5%, 20%, and 50%

            for rate in sampling_rates:
                sample_config = {
                    "name": op_name,
                    "type": op_type,
                    "method": "stratify",
                    "samples": rate,
                    "method_kwargs": {
                        "stratify_key": doc_id_key,
                        "random": False,
                    },
                }
                sample_configs.append(sample_config)

            # Add the sample configurations to the pipeline
            complete_pipeline.append(sample_configs)

            # Create a formatted table of configurations
            sample_table = Table(title=None, box=None, expand=True)
            sample_table.add_column("#", style="bold")
            sample_table.add_column("Configuration", style="cyan")
            sample_table.add_column("Sampling Rate", style="magenta")

            for j, cfg in enumerate(sample_configs):
                sample_table.add_row(
                    str(j + 1),
                    f"{cfg['name']} (method: {cfg['method']})",
                    f"{int(cfg['samples'] * 100)}%",
                )

            console.print(
                create_panel(
                    "Generated Sample Configurations", sample_table, style=SUCCESS_STYLE
                )
            )
            continue

        # We need to generate a new configuration
        op_prompt = generate_op_config_prompt(
            op_name=op_name,
            op_type=op_type,
            original_op_name=original_op_name,
            pipeline_config=pipeline_config,
            previous_ops=complete_pipeline,  # Use first variant for context
            is_final_op=is_final_op,
            sample_docs=sample_docs,
            skeleton_metadata=op_info.get("skeleton_metadata"),
        )

        # Add to conversation and get response
        conversation.append({"role": "user", "content": op_prompt})

        # Track syntax check success
        syntax_check_success = False
        max_syntax_attempts = 3
        syntax_attempt = 0
        current_op_config = None
        syntax_error = None

        while not syntax_check_success and syntax_attempt < max_syntax_attempts:
            if syntax_attempt > 0:
                # Add error feedback for reprompting
                error_prompt = f"""
The operation configuration you provided failed the syntax check with the following error:
{syntax_error}

Please fix the configuration for operation '{op_name}' of type '{op_type}'.
Make sure to:
1. Include all required fields for this operation type
2. Use proper Jinja syntax for templates
3. Ensure output schema has valid types
4. All JSON/YAML is properly formatted
                """
                conversation.append({"role": "user", "content": error_prompt})

                # Show syntax error in a panel
                console.print(
                    create_panel(
                        f"Syntax Check Failed (Attempt {syntax_attempt}/{max_syntax_attempts})",
                        Text(syntax_error),
                        style=ERROR_STYLE,
                    )
                )

            # Generate operation config
            response = llm_client.generate_rewrite(
                conversation, system_prompt, op_config_schema
            )

            # Parse the response
            op_result = json.loads(response.choices[0].message.content)

            # Handle the case of map* where we get multiple operations
            if "op_configs" in op_result:
                op_configs = op_result["op_configs"]

                # For map*, verify each sub-operation
                all_valid = True
                invalid_configs = []

                # Only perform syntax check if runner is provided
                if runner:
                    for j, config in enumerate(op_configs):
                        # Ensure each configuration has a valid name and type
                        if "name" not in config:
                            config["name"] = f"{op_name}_{j+1}"
                        if "type" not in config:
                            config["type"] = "map"

                        # Create temporary OpContainer for syntax check
                        temp_container = OpContainer(
                            f"temp/{config['name']}", runner, config
                        )

                        try:
                            # Run syntax check
                            temp_container.syntax_check()
                        except Exception as e:
                            all_valid = False
                            invalid_configs.append((j, str(e)))
                            syntax_error = f"Error in sub-operation {j+1} ({config['name']}): {str(e)}"

                if all_valid or not runner:
                    syntax_check_success = True
                    current_op_config = op_configs
            else:
                # Standard single operation case
                op_config = op_result["op_config"]

                # Ensure the configuration has the correct name and type
                op_config["name"] = op_name
                op_config["type"] = op_type

                # Only perform syntax check if runner is provided
                if runner:
                    # Create temporary OpContainer for syntax check
                    temp_container = OpContainer(f"temp/{op_name}", runner, op_config)

                    try:
                        # Run syntax check
                        temp_container.syntax_check()
                        syntax_check_success = True
                        current_op_config = op_config
                    except Exception as e:
                        syntax_error = str(e)
                else:
                    syntax_check_success = True
                    current_op_config = op_config

            # Add the response to the conversation history
            conversation.append(
                {"role": "assistant", "content": response.choices[0].message.content}
            )

            syntax_attempt += 1

        # If we reach max attempts and still failed, use the last config anyway but warn
        if not syntax_check_success and syntax_attempt >= max_syntax_attempts:
            warning_text = Text(
                f"Could not generate valid config for {op_name} after {max_syntax_attempts} attempts. Using last generated config."
            )
            console.print(
                create_panel("Syntax Check Failed", warning_text, style=WARNING_STYLE)
            )

        # Handle the case of map* where we get multiple operations
        if isinstance(current_op_config, list):
            # For map*, we have multiple operations to add
            op_configs_list = current_op_config

            # Create a table for the map* operations
            map_star_table = Table(title=None, box=None, expand=True)
            map_star_table.add_column("Sub-Op #", style="bold")
            map_star_table.add_column("Name", style="cyan")
            map_star_table.add_column("Type", style="magenta")
            map_star_table.add_column("Variations", style="green")

            # Each map* operation expands to multiple sub-operations
            # For each sub-operation position, we'll have a list of variations
            for sub_op_idx, base_config in enumerate(op_configs_list):
                # Ensure each configuration has a valid name and type
                if "name" not in base_config:
                    base_config["name"] = f"{op_name}_{sub_op_idx+1}"
                if "type" not in base_config:
                    base_config["type"] = "map"  # map* is composed of map operations

                # Generate variations for this sub-operation based on model choices
                sub_op_configs = []

                # Include the base configuration
                sub_op_configs.append(base_config)

                # For map operations, create model variations
                if base_config["type"] in ["map", "reduce"]:
                    base_model = base_config.get(
                        "model",
                        (
                            runner.optimizer.default_model
                            if runner and hasattr(runner, "optimizer")
                            else "gpt-4o-mini"
                        ),
                    )
                    for model in model_choices:
                        if model != base_model:
                            model_config = copy.deepcopy(base_config)
                            model_config["model"] = model
                            sub_op_configs.append(model_config)

                # Add to the table
                map_star_table.add_row(
                    str(sub_op_idx + 1),
                    base_config["name"],
                    base_config["type"],
                    str(len(sub_op_configs)),
                )

                # Add this sub-operation's configurations to the pipeline
                complete_pipeline.append(sub_op_configs)

            # Print the map* operations summary
            console.print(
                create_panel(
                    f"Synthesized Map* Operations ({len(op_configs_list)} sub-operations)",
                    map_star_table,
                    style=SUCCESS_STYLE,
                )
            )
        else:
            # Standard single operation case
            op_config = current_op_config

            # Generate variations for this operation based on model choices
            op_configs = []

            # Include the base configuration
            op_configs.append(op_config)

            # For map and reduce operations, create model variations
            if op_type in ["map", "reduce"]:
                base_model = op_config.get(
                    "model",
                    (
                        runner.optimizer.default_model
                        if runner and hasattr(runner, "optimizer")
                        else "gpt-4o-mini"
                    ),
                )
                for model in model_choices:
                    if model != base_model:
                        model_config = copy.deepcopy(op_config)
                        model_config["model"] = model
                        op_configs.append(model_config)

            # Add the operation configurations to the pipeline
            complete_pipeline.append(op_configs)

            # Create a formatted table of model variations
            models_table = Table(title=None, box=None, expand=True)
            models_table.add_column("#", style="bold")
            models_table.add_column("Configuration", style="cyan")
            models_table.add_column("Model", style="magenta")

            for j, cfg in enumerate(op_configs):
                model = cfg.get("model", "default")
                models_table.add_row(str(j + 1), cfg["name"], model)

            console.print(
                create_panel(
                    f"Synthesized {op_type.capitalize()} Configurations",
                    models_table,
                    style=SUCCESS_STYLE,
                )
            )

    # Process any scan or step_boundary operations that come after processing operations
    final_ops = []
    found_operations = set()
    for op_list in complete_pipeline:
        for op in op_list:
            found_operations.add(op["name"])

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
                    # Add as a list with one item
                    complete_pipeline.append([original_ops[original_name]])
                    final_ops.append(original_name)

    if final_ops:
        console.print(
            create_panel(
                "Final Operations",
                Text("Adding final operations: " + ", ".join(final_ops)),
                style=SUCCESS_STYLE,
            )
        )

    # Generate all possible pipeline combinations using the cross product
    def generate_pipelines(config_lists):
        return list(product(*config_lists))

    pipeline_combinations = generate_pipelines(complete_pipeline)

    # Print pipeline combinations summary
    total_combinations = len(pipeline_combinations)
    console.print(
        create_panel(
            "Pipeline Combinations",
            Text(f"Generated {total_combinations} possible pipeline configurations"),
            style=SUCCESS_STYLE,
        )
    )

    # Print a summary of a few representative pipelines
    sample_size = min(3, total_combinations)
    if sample_size > 0:
        samples_table = Table(title=None, box=None, expand=True)
        samples_table.add_column("Pipeline #", style="bold")
        samples_table.add_column("Operations", style="cyan")
        samples_table.add_column("Models Used", style="magenta")

        for i in range(sample_size):
            pipeline = pipeline_combinations[i]
            op_names = [op["name"] for op in pipeline]
            models = [
                op.get("model", "default")
                for op in pipeline
                if op.get("type") in ["map", "reduce"]
            ]

            samples_table.add_row(
                str(i + 1),
                ", ".join(op_names),
                ", ".join(models) if models else "No LLM models",
            )

        console.print(
            create_panel(
                f"Sample Pipeline Configurations (showing {sample_size}/{total_combinations})",
                samples_table,
                style=INFO_STYLE,
            )
        )

    # Convert pipeline combinations to list of complete pipelines
    pipelines = [list(pipeline) for pipeline in pipeline_combinations]

    # Print summary of the complete pipelines
    pipeline_summary = Table(title=None, box=None, expand=True)
    pipeline_summary.add_column("Total Pipelines", style="bold")
    pipeline_summary.add_column("Operations Per Pipeline", style="cyan")

    pipeline_summary.add_row(
        str(len(pipelines)), str(len(pipelines[0]) if pipelines else 0)
    )

    console.print(
        create_panel("Complete Pipeline Summary", pipeline_summary, style=SUCCESS_STYLE)
    )

    return pipelines
