import json
from copy import deepcopy
from typing import Dict, List, Type

from pydantic import BaseModel, Field

from docetl.reasoning_optimizer.instantiate_schemas import SwapWithCodeInstantiateSchema

from .agent_utils import AgenticDirectiveRunner
from .base import Directive, DirectiveTestCase


class SwapWithCodeDirective(Directive):
    name: str = Field(default="swap_with_code", description="The name of the directive")
    formal_description: str = Field(default="Reduce => Code Reduce + Map")
    nl_description: str = Field(
        default="Replaces a Reduce operation with a Code Reduce operation for deterministic logic plus an optional Map operation to format the output. The Code Reduce handles the core reduction logic (like counting, collecting, or aggregating) while the optional Map operation converts the result to match the expected schema format."
    )
    when_to_use: str = Field(
        default="When a reduce operation performs logic that can be implemented more efficiently or deterministically with code rather than an LLM. Examples include: counting distinct values, finding most common elements, basic aggregations, set operations, or mathematical computations. Particularly useful when the reduction logic is straightforward but the output needs to be formatted in a specific way for downstream operations."
    )

    instantiate_schema_type: Type[BaseModel] = Field(
        default=SwapWithCodeInstantiateSchema
    )

    example: str = Field(
        default="""
        Target Operation:
        - name: summarize_locations
          type: reduce
          reduce_key: "_all"
          prompt: |
            Summarize all distinct locations from these documents:
            {% for input in inputs %}{{ input.locations }}{% endfor %}
          output:
            schema:
              summary: "str"
              distinct_locations: "list[str]"

        The agent might convert this to:
        1. Code Reduce that collects distinct locations: {"locations": ["NYC", "SF", "LA"]}
        2. Optional Map that formats this as: {"summary": "Found 3 distinct locations: NYC, SF, LA", "distinct_locations": ["NYC", "SF", "LA"]}

        Example InstantiateSchema (what the agent should output):
        SwapWithCodeInstantiateSchema(
            code_reduce_name="collect_distinct_locations",
            code="def transform(inputs):\n    locations = set()\n    for item in inputs:\n        if 'locations' in item and isinstance(item['locations'], list):\n            locations.update(item['locations'])\n    return {'distinct_locations': sorted(list(locations))}",
            map_prompt="Create a summary of the locations: {{ input.distinct_locations }}. Output format: summary (string describing the count and locations), distinct_locations (the original list)."
        )
        """
    )

    test_cases: List[DirectiveTestCase] = Field(
        default_factory=lambda: [
            DirectiveTestCase(
                name="basic_reduce_to_code_reduce",
                description="Should convert reduce operation to code reduce + optional map",
                input_config={
                    "name": "count_items",
                    "type": "reduce",
                    "reduce_key": "_all",
                    "prompt": "Count the total items: {% for input in inputs %}{{ input.count }}{% endfor %}",
                    "output": {"schema": {"total": "int"}},
                },
                target_ops=["count_items"],
                expected_behavior="Should replace reduce with code reduce that performs counting logic and optional map to format output",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="reduce_with_grouping",
                description="Should handle reduce operations with grouping keys",
                input_config={
                    "name": "group_by_category",
                    "type": "reduce",
                    "reduce_key": "category",
                    "prompt": "List all items for this category: {% for input in inputs %}{{ input.name }}{% endfor %}",
                    "output": {"schema": {"category": "str", "items": "list[str]"}},
                },
                target_ops=["group_by_category"],
                expected_behavior="Should preserve reduce_key grouping in the code reduce operation",
                should_pass=True,
            ),
        ]
    )

    def __eq__(self, other):
        return isinstance(other, SwapWithCodeDirective)

    def __hash__(self):
        return hash("SwapWithCodeDirective")

    def to_string_for_instantiate(
        self, target_ops_configs: List[Dict], pipeline_code: Dict = None
    ) -> str:
        """
        Generate a prompt that asks the agent to analyze sample data and create a code reduce + optional map replacement.
        """
        assert (
            len(target_ops_configs) == 1
        ), "SwapWithCode directive only supports single target operation"

        op = target_ops_configs[0]
        original_prompt = op.get("prompt", "")
        reduce_key = op.get("reduce_key", "_all")
        output_schema = op.get("output", {}).get("schema", {})

        # Build pipeline context
        pipeline_context = ""
        if pipeline_code:
            pipeline_context = f"""
Pipeline Context:
{json.dumps(pipeline_code, indent=2)}

The target reduce operation '{op['name']}' fits into this broader pipeline. Consider:
- What data flows into this operation from previous steps
- How this operation's output will be used by subsequent operations
- The overall goal of the pipeline when designing your code reduce + map solution
"""

        return (
            f"You are an expert at analyzing reduce operations and implementing efficient code-based alternatives.\n\n"
            f"Target Reduce Operation:\n"
            f"{json.dumps(op, indent=2)}\n\n"
            f"Original Prompt: {original_prompt}\n"
            f"Reduce Key: {reduce_key}\n"
            f"Expected Output Schema: {json.dumps(output_schema, indent=2)}\n\n"
            f"{pipeline_context}\n"
            f"Your task is to replace this reduce operation with:\n"
            f"1. A Code Reduce operation that implements the core reduction logic deterministically\n"
            f"2. An optional Map operation that formats the code reduce output to match the expected schema\n\n"
            f"You will be given access to sample input data through a read_next_docs() function. Use this to:\n"
            f"1. Understand the actual structure and patterns in the input data\n"
            f"2. Identify what the reduce operation is trying to accomplish\n"
            f"3. Design efficient Python code that performs the same reduction logic\n"
            f"4. Determine if a follow-up map operation is needed to format the output correctly\n"
            f"5. Consider edge cases and data variations in your implementation\n\n"
            f"Guidelines for the replacement:\n"
            f"- The Code Reduce must implement a 'transform' function that takes a list of inputs and returns a dictionary\n"
            f"- Include all necessary imports within the transform function\n"
            f"- The code should handle the same reduce_key grouping as the original operation\n"
            f"- If the code reduce output doesn't match the expected schema, provide a map_prompt to format it\n"
            f"- The map operation (if needed) should reference fields from the code reduce output using {{{{ input.field_name }}}}\n"
            f"- Focus on correctness, efficiency, and handling edge cases found in the sample data\n\n"
            f"Examples of good candidates for code reduce:\n"
            f"- Counting, summing, or basic mathematical operations\n"
            f"- Collecting distinct values or creating sets\n"
            f"- Finding min/max values or sorting\n"
            f"- Simple aggregations or list operations\n"
            f"- Deterministic text processing\n\n"
            f"Example transformation:\n"
            f"{self.example}\n\n"
            f"Analyze samples strategically to understand the data patterns and reduction requirements.\n"
            f"When you have enough information to create an efficient code-based solution, output your result."
        )

    def llm_instantiate(
        self,
        target_ops_configs: List[Dict],
        input_file_path: str,
        agent_llm: str,
        message_history: list = [],
        pipeline_code: Dict = None,
    ):
        """
        Use agentic approach to analyze sample data and generate code reduce + optional map replacement.
        """
        # Load sample input data
        try:
            with open(input_file_path, "r") as f:
                input_data = json.load(f)

            if not isinstance(input_data, list) or len(input_data) == 0:
                raise ValueError(
                    "Input file must contain a non-empty list of sample data"
                )

        except Exception as e:
            raise Exception(
                f"Failed to load input data from {input_file_path}: {str(e)}"
            )

        # Create validation function
        def validate_code_reduce_schema(schema_instance):
            # Basic validation is handled by Pydantic validators
            # Could add additional validation here if needed
            pass

        # Set up agentic runner with validation
        runner = AgenticDirectiveRunner(
            input_data=input_data,
            agent_llm=agent_llm,
            validation_func=validate_code_reduce_schema,
        )

        # Create system prompt for the agentic runner
        system_prompt = (
            "You are an expert at analyzing reduce operations and designing efficient code-based alternatives. "
            "Your goal is to examine input samples to understand the reduction logic, then implement it as "
            "efficient Python code with optional formatting. You consider both performance and correctness "
            "while ensuring the output matches the expected schema through code reduce + optional map pattern."
        )

        # Create initial user message
        initial_message = self.to_string_for_instantiate(
            target_ops_configs, pipeline_code
        )

        # Run the agentic loop
        try:
            schema, updated_message_history, call_cost = runner.run_agentic_loop(
                system_prompt=system_prompt,
                initial_user_message=initial_message,
                response_schema=SwapWithCodeInstantiateSchema,
            )

            # Update message history
            message_history.extend(updated_message_history)

            return schema, message_history, call_cost

        except Exception as e:
            raise Exception(f"Failed to instantiate swap_with_code directive: {str(e)}")

    def apply(
        self,
        global_default_model: str,
        ops_list: List[Dict],
        target_ops: List[str],
        rewrite: SwapWithCodeInstantiateSchema,
    ) -> List[Dict]:
        """
        Apply the directive by replacing the reduce operation with code reduce + optional map.
        """
        new_ops_list = deepcopy(ops_list)

        # Find the target operation
        target_pos = None
        target_op = None
        for i, op in enumerate(new_ops_list):
            if op["name"] in target_ops:
                target_pos = i
                target_op = op
                break

        if target_pos is None:
            raise ValueError(f"Target operation {target_ops[0]} not found")

        # Get model from original reduce operation or use global default
        default_model = target_op.get("model", global_default_model)

        # Create the code reduce operation
        code_reduce_op = {
            "name": rewrite.code_reduce_name,
            "type": "code_reduce",
            "code": rewrite.code,
            "reduce_key": target_op.get("reduce_key", "_all"),
        }

        # Start with just the code reduce operation
        replacement_ops = [code_reduce_op]

        # Add optional map operation if specified
        if rewrite.map_prompt is not None and rewrite.map_prompt.strip():
            map_op = {
                "name": target_op[
                    "name"
                ],  # Keep the original name for the final output
                "type": "map",
                "prompt": rewrite.map_prompt,
                "model": default_model,
                "output": target_op.get("output", {}),
            }
            replacement_ops.append(map_op)
        else:
            # If no map operation, rename the code reduce to match original name
            code_reduce_op["name"] = target_op["name"]
            # Add output schema if it exists in original
            if "output" in target_op:
                code_reduce_op["output"] = target_op["output"]

        # Replace the target operation with the replacement operations
        new_ops_list = (
            new_ops_list[:target_pos] + replacement_ops + new_ops_list[target_pos + 1 :]
        )

        return new_ops_list

    def instantiate(
        self,
        operators: List[Dict],
        target_ops: List[str],
        agent_llm: str,
        message_history: list = [],
        global_default_model: str = None,
        **kwargs,
    ):
        """
        Main method that orchestrates directive instantiation:
        1. Use agentic approach to analyze data and generate code reduce + optional map
        2. Apply the transformation using that configuration
        """
        assert (
            len(target_ops) == 1
        ), "SwapWithCode directive requires exactly one target operation"

        input_file_path = kwargs.get("input_file_path", None)
        pipeline_code = kwargs.get("pipeline_code", None)

        if not input_file_path:
            raise ValueError("input_file_path is required for SwapWithCode directive")

        # Get configuration for target operation
        target_ops_configs = [op for op in operators if op["name"] in target_ops]

        if not target_ops_configs:
            raise ValueError(f"Target operation {target_ops[0]} not found in operators")

        # Validate that target operation is a reduce operation
        target_op = target_ops_configs[0]
        if target_op.get("type") != "reduce":
            raise ValueError(
                f"SwapWithCode directive can only be applied to reduce operations, but {target_ops[0]} is of type {target_op.get('type')}"
            )

        # Step 1: Agent analyzes data and generates code reduce + optional map solution
        rewrite, message_history, call_cost = self.llm_instantiate(
            target_ops_configs,
            input_file_path,
            agent_llm,
            message_history,
            pipeline_code,
        )

        # Step 2: Apply transformation using the generated configuration
        return (
            self.apply(global_default_model, operators, target_ops, rewrite),
            message_history, call_cost
        )
