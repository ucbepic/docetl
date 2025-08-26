import json
from typing import Dict, List, Type

from pydantic import BaseModel, Field

from docetl.reasoning_optimizer.instantiate_schemas import (
    ArbitraryRewriteInstantiateSchema,
)
from docetl.reasoning_optimizer.op_descriptions import get_all_operator_descriptions

from .agent_utils import AgenticDirectiveRunner
from .base import Directive, DirectiveTestCase


class ArbitraryRewriteDirective(Directive):
    name: str = Field(
        default="arbitrary_rewrite", description="The name of the directive"
    )
    formal_description: str = Field(
        default="Pipeline => Modified Pipeline (search/replace edits)"
    )
    nl_description: str = Field(
        default="Allows the agent to make arbitrary edits to the pipeline using search/replace operations on the JSON representation. The agent can add, modify, remove, or replace operations to optimize for cost, accuracy, or both. This is a catch-all directive for optimizations that don't fit into other specific directive patterns."
    )
    when_to_use: str = Field(
        default="When you identify obvious optimizations that can make the pipeline cheaper or more accurate, but they don't fit into existing directive patterns. Use this for complex multi-operation changes, reordering operations, consolidating redundant operations, or making systematic improvements across the pipeline."
    )

    instantiate_schema_type: Type[BaseModel] = Field(
        default=ArbitraryRewriteInstantiateSchema
    )

    example: str = Field(
        default="""
        # Example 1: Consolidating Redundant Operations
        Original Pipeline JSON (formatted for readability):
        [
            {
                "name": "extract_names",
                "type": "map",
                "model": "gpt-4o",
                "prompt": "Extract all person names from: {{ input.text }}",
                "output": {"schema": {"names": "list[string]"}}
            },
            {
                "name": "extract_locations",
                "type": "map",
                "model": "gpt-4o",
                "prompt": "Extract all locations from: {{ input.text }}",
                "output": {"schema": {"locations": "list[string]"}}
            },
            {
                "name": "extract_dates",
                "type": "map",
                "model": "gpt-4o",
                "prompt": "Extract all dates from: {{ input.text }}",
                "output": {"schema": {"dates": "list[string]"}}
            }
        ]

        Example SearchReplaceEdits (agent recognizes redundant LLM calls):
        search_replace_edits=[
            SearchReplaceEdit(
                search='    {\\n        "name": "extract_names",\\n        "type": "map",\\n        "model": "gpt-4o",\\n        "prompt": "Extract all person names from: {{ input.text }}",\\n        "output": {"schema": {"names": "list[string]"}}\\n    },\\n    {\\n        "name": "extract_locations",\\n        "type": "map",\\n        "model": "gpt-4o",\\n        "prompt": "Extract all locations from: {{ input.text }}",\\n        "output": {"schema": {"locations": "list[string]"}}\\n    },\\n    {\\n        "name": "extract_dates",\\n        "type": "map",\\n        "model": "gpt-4o",\\n        "prompt": "Extract all dates from: {{ input.text }}",\\n        "output": {"schema": {"dates": "list[string]"}}\\n    }',
                replace='    {\\n        "name": "extract_all_entities",\\n        "type": "map",\\n        "model": "gpt-4o-mini",\\n        "prompt": "Extract all entities from the text:\\\\n{{ input.text }}\\\\n\\\\nReturn names, locations, and dates.",\\n        "output": {\\n            "schema": {\\n                "names": "list[string]",\\n                "locations": "list[string]",\\n                "dates": "list[string]"\\n            }\\n        }\\n    }',
                reasoning="Consolidate three separate GPT-4o calls into one GPT-4o-mini call"
            )
        ]

        # Example 2: Reordering for Efficiency
        Original Pipeline JSON:
        [
            {
                "name": "summarize_documents",
                "type": "map",
                "model": "gpt-4o",
                "prompt": "Summarize this document: {{ input.full_text }}",
                "output": {"schema": {"summary": "string"}}
            },
            {
                "name": "filter_relevant",
                "type": "filter",
                "model": "gpt-4o-mini",
                "prompt": "Is this document about technology? {{ input.full_text }}",
                "output": {"schema": {"is_relevant": "boolean"}}
            }
        ]

        Example SearchReplaceEdits (agent recognizes inefficient ordering):
        search_replace_edits=[
            SearchReplaceEdit(
                search='{\\n        "name": "summarize_documents",\\n        "type": "map",\\n        "model": "gpt-4o",\\n        "prompt": "Summarize this document: {{ input.full_text }}",\\n        "output": {"schema": {"summary": "string"}}\\n    },\\n    {\\n        "name": "filter_relevant",',
                replace='{\\n        "name": "filter_relevant",',
                reasoning="Remove summarize operation from before filter"
            ),
            SearchReplaceEdit(
                search='"output": {"schema": {"is_relevant": "boolean"}}\\n    }',
                replace='"output": {"schema": {"is_relevant": "boolean"}}\\n    },\\n    {\\n        "name": "summarize_documents",\\n        "type": "map",\\n        "model": "gpt-4o",\\n        "prompt": "Summarize this technology document: {{ input.full_text }}",\\n        "output": {"schema": {"summary": "string"}}\\n    }',
                reasoning="Add summarization after filter to only process relevant documents"
            )
        ]

        Note: The agent must provide exact string matches including whitespace for the search strings.
        """
    )

    test_cases: List[DirectiveTestCase] = Field(
        default_factory=lambda: [
            DirectiveTestCase(
                name="consolidate_operations",
                description="Should identify and consolidate redundant operations",
                input_config=[
                    {
                        "name": "extract_a",
                        "type": "map",
                        "model": "gpt-4o",
                        "prompt": "Extract A from: {{ input.text }}",
                        "output": {"schema": {"a": "string"}},
                    },
                    {
                        "name": "extract_b",
                        "type": "map",
                        "model": "gpt-4o",
                        "prompt": "Extract B from: {{ input.text }}",
                        "output": {"schema": {"b": "string"}},
                    },
                ],
                target_ops=[],  # Analyzes entire pipeline
                expected_behavior="Should propose consolidating the two extraction operations into one",
                should_pass=True,
            )
        ]
    )

    def __eq__(self, other):
        return isinstance(other, ArbitraryRewriteDirective)

    def __hash__(self):
        return hash("ArbitraryRewriteDirective")

    def _order_ops_by_pipeline_steps(
        self, ops_list: List[Dict], pipeline_code: Dict = None
    ) -> List[Dict]:
        """Order operations according to pipeline steps if pipeline_code is provided."""
        if not (
            pipeline_code
            and "pipeline" in pipeline_code
            and "steps" in pipeline_code["pipeline"]
        ):
            return ops_list

        # Get the order from pipeline steps
        steps = pipeline_code["pipeline"]["steps"]
        step_order = []
        for step in steps:
            if isinstance(step, dict) and "operations" in step:
                step_order.extend(step["operations"])
            elif isinstance(step, str):
                step_order.append(step)

        # Create a mapping of op names to ops
        ops_by_name = {op["name"]: op for op in ops_list}

        # Reorder ops according to pipeline steps
        ordered_ops = []
        for op_name in step_order:
            if op_name in ops_by_name:
                ordered_ops.append(ops_by_name[op_name])

        # Add any ops that weren't in steps (shouldn't happen, but just in case)
        for op in ops_list:
            if op not in ordered_ops:
                ordered_ops.append(op)

        return ordered_ops

    def to_string_for_instantiate(
        self, pipeline_ops: List[Dict], pipeline_code: Dict = None
    ) -> str:
        """Generate prompt for agent to analyze pipeline and propose edits."""

        # Convert ops to pretty JSON string for display (already ordered by instantiate)
        pipeline_json = json.dumps(pipeline_ops, indent=4)

        return (
            f"You are an expert at optimizing data processing pipelines for cost and accuracy.\n\n"
            f"Current Pipeline Operations (as JSON array):\n"
            f"{pipeline_json}\n\n"
            f"Your task is to analyze this pipeline and propose search/replace edits to optimize it.\n\n"
            f"IMPORTANT: The above JSON is ONLY the operations array. When creating search/replace edits, "
            f"work with this exact JSON structure. Do not include pipeline wrapper or other fields.\n\n"
            f"You have access to:\n"
            f"1. Sample input data through read_next_docs() - to understand data patterns and flow\n"
            f"2. Operator documentation through read_operator_doc(operator_name) - to learn about available operators\n\n"
            f"IMPORTANT: Read documentation for no more than 2 operators before analyzing sample data to get a sense for how best to rewrite the pipeline.\n\n"
            f"Use these tools to:\n"
            f"1. Understand the data flow through the pipeline\n"
            f"2. Identify inefficiencies or redundancies\n"
            f"3. Find opportunities for consolidation or reordering\n"
            f"4. Determine if cheaper models could be used\n"
            f"5. Discover new operators that might be more efficient\n\n"
            f"IMPORTANT: You will provide search/replace edits that work on the JSON string representation.\n"
            f"Each edit consists of:\n"
            f"- search: An exact string to find in the JSON (including whitespace)\n"
            f"- replace: The string to replace it with (can be empty to delete)\n"
            f"- reasoning: Why this edit improves the pipeline\n\n"
            f"The edits will be applied sequentially to the JSON string, then parsed back to operations.\n\n"
            f"Guidelines for search/replace:\n"
            f"- The search string must match EXACTLY including all whitespace, quotes, brackets, etc.\n"
            f"- You can delete operations by replacing them with empty string (but be careful with commas)\n"
            f"- You can add operations by replacing closing brackets with new operations\n"
            f"- You can reorder by using multiple search/replace operations\n"
            f"- Each edit operates on the result of the previous edit\n\n"
            f"Types of optimizations to look for:\n"
            f"- Redundant operations that could be consolidated\n"
            f"- Operations in suboptimal order (e.g., expensive operations before filters)\n"
            f"- Opportunities to use cheaper models\n"
            f"- Complex operations that could be broken into simpler steps\n"
            f"- Independent operations that could be parallelized\n\n"
            f"Examples:\n"
            f"{self.example}\n\n"
            f"Analyze the pipeline and sample data strategically. When you have identified optimizations, "
            f"output your proposed edits as an ArbitraryRewriteInstantiateSchema.\n\n"
            f"Remember: Your search strings must match the JSON exactly as it appears above."
        )

    def llm_instantiate(
        self,
        pipeline_ops: List[Dict],
        input_file_path: str,
        agent_llm: str,
        message_history: list = [],
        pipeline_code: Dict = None,
    ):
        """Use agentic approach to analyze pipeline and generate edits."""
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

        # Set up agentic runner with operator doc reading enabled
        runner = AgenticDirectiveRunner(
            input_data=input_data,
            agent_llm=agent_llm,
            enable_operator_docs=True,  # Enable reading operator documentation
        )

        # Create system prompt with operator descriptions
        operator_descriptions = get_all_operator_descriptions()
        system_prompt = (
            "You are an expert at optimizing data processing pipelines. "
            "You analyze pipeline structures and data flow to identify opportunities "
            "for cost reduction and accuracy improvement through strategic search/replace edits on the JSON representation.\n\n"
            f"{operator_descriptions}"
        )

        # Create initial user message
        initial_message = self.to_string_for_instantiate(pipeline_ops, pipeline_code)

        # Run the agentic loop
        try:
            schema, updated_message_history, call_cost = runner.run_agentic_loop(
                system_prompt=system_prompt,
                initial_user_message=initial_message,
                response_schema=ArbitraryRewriteInstantiateSchema,
            )

            # Update message history
            message_history.extend(updated_message_history)

            return schema, message_history, call_cost

        except Exception as e:
            raise Exception(
                f"Failed to instantiate arbitrary_rewrite directive: {str(e)}"
            )

    def apply(
        self,
        ops_list: List[Dict],
        rewrite: ArbitraryRewriteInstantiateSchema,
    ) -> List[Dict]:
        """Apply the search/replace edits to the pipeline."""
        # Convert operations list to JSON string with consistent formatting
        pipeline_json = json.dumps(ops_list, indent=4)

        # Apply each search/replace edit in sequence
        for i, edit in enumerate(rewrite.search_replace_edits):
            if edit.search in pipeline_json:
                pipeline_json = pipeline_json.replace(
                    edit.search, edit.replace, 1
                )  # Replace first occurrence only
            else:
                # Log warning but continue with other edits
                print(
                    f"Warning: Search string not found in edit {i+1}: {edit.search[:50]}..."
                )

        # Parse the modified JSON back to operations list
        try:
            new_ops_list = json.loads(pipeline_json)
            if not isinstance(new_ops_list, list):
                raise ValueError("Pipeline must be a list of operations")

            # Get rid of any empty operations in new_ops_list
            new_ops_list = [op for op in new_ops_list if op]
            return new_ops_list
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse pipeline after edits. The search/replace operations resulted in invalid JSON: {e}\n"
                f"Modified JSON:\n{pipeline_json[:500]}..."
            )

    def instantiate(
        self,
        operators: List[Dict],
        target_ops: List[str] = None,
        agent_llm: str = "gpt-4o-mini",
        message_history: list = [],
        **kwargs,
    ):
        """
        Main method that orchestrates directive instantiation.
        For ArbitraryRewrite, we analyze the entire pipeline rather than specific target ops.
        """
        input_file_path = kwargs.get("input_file_path", None)
        pipeline_code = kwargs.get("pipeline_code", None)

        if not input_file_path:
            raise ValueError(
                "input_file_path is required for ArbitraryRewrite directive"
            )

        # Order ops according to pipeline steps before everything else
        ordered_ops = self._order_ops_by_pipeline_steps(operators, pipeline_code)

        # Step 1: Agent analyzes pipeline and generates edits
        rewrite, message_history, call_cost = self.llm_instantiate(
            ordered_ops,
            input_file_path,
            agent_llm,
            message_history,
            pipeline_code,
        )

        # Step 2: Apply the edits
        return (
            self.apply(ordered_ops, rewrite),
            message_history,
            call_cost,
        )
