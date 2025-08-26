import json
from copy import deepcopy
from typing import Dict, List, Type

from pydantic import BaseModel, Field

from docetl.reasoning_optimizer.instantiate_schemas import (
    ClarifyInstructionsInstantiateSchema,
)

from .agent_utils import AgenticDirectiveRunner
from .base import Directive, DirectiveTestCase


class ClarifyInstructionsDirective(Directive):
    name: str = Field(
        default="clarify_instructions", description="The name of the directive"
    )
    formal_description: str = Field(default="Single-op => Op")
    nl_description: str = Field(
        default="Improves a single operation's prompt clarity and specificity by analyzing sample input data to identify patterns, resolve ambiguities, and create more precise instructions that reduce LLM confusion and improve output consistency"
    )
    when_to_use: str = Field(
        default="When a single operation has a vague or ambiguous prompt that could benefit from more specific instructions based on actual data patterns. Particularly useful when you have multiple samples of input data and want to create a prompt for one specific operation that handles the patterns and edge cases present in your dataset"
    )

    instantiate_schema_type: Type[BaseModel] = Field(
        default=ClarifyInstructionsInstantiateSchema
    )

    example: str = Field(
        default="""
        Target Operation:
        - name: extract_key_findings
          type: map
          prompt: |
            Extract the key findings from: {{ input.research_paper }}
          output:
            schema:
              findings: "list[str]"

        After analyzing sample research papers, the agent might discover papers contain
        abstracts, conclusions, and results sections with different formats.

        Example InstantiateSchema (what the agent should output):
        ClarifyInstructionsInstantiateSchema(
            clarified_prompt="Extract the key findings from the research paper: {{ input.research_paper }}. Focus on: 1) Main experimental results and statistical significance from Results sections, 2) Primary conclusions from Abstract and Conclusion sections, 3) Novel contributions explicitly stated by authors. Ignore methodological details, related work summaries, and future work suggestions. Return 3-7 concise findings as bullet points."
        )
        """
    )

    test_cases: List[DirectiveTestCase] = Field(
        default_factory=lambda: [
            DirectiveTestCase(
                name="single_target_clarification",
                description="Should create clarified prompt for single target operation",
                input_config={
                    "name": "analyze_feedback",
                    "type": "map",
                    "prompt": "Analyze the feedback: {{ input.feedback }}",
                    "output": {
                        "schema": {"sentiment": "string", "issues": "list[str]"}
                    },
                },
                target_ops=["analyze_feedback"],
                expected_behavior="Should replace the original prompt with a more specific version based on analysis of sample feedback data. The clarified prompt should reference {{ input.feedback }} and provide specific guidance on what to analyze.",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="preserve_input_variables",
                description="Should preserve all input variable references from original prompt",
                input_config={
                    "name": "compare_documents",
                    "type": "map",
                    "prompt": "Compare {{ input.doc1 }} with {{ input.doc2 }} and identify differences",
                    "output": {"schema": {"differences": "list[str]"}},
                },
                target_ops=["compare_documents"],
                expected_behavior="Should create clarified prompt that still references both {{ input.doc1 }} and {{ input.doc2 }} while providing more specific comparison instructions.",
                should_pass=True,
            ),
        ]
    )

    def __eq__(self, other):
        return isinstance(other, ClarifyInstructionsDirective)

    def __hash__(self):
        return hash("ClarifyInstructionsDirective")

    def to_string_for_instantiate(
        self, target_ops_configs: List[Dict], pipeline_code: Dict = None
    ) -> str:
        """
        Generate a prompt that asks the agent to analyze sample data and create an improved prompt.
        """
        assert (
            len(target_ops_configs) == 1
        ), "ClarifyInstructions directive only supports single target operation"

        op = target_ops_configs[0]
        original_prompt = op.get("prompt", "")

        # Build pipeline context
        pipeline_context = ""
        if pipeline_code:
            pipeline_context = f"""
Pipeline Context:
{json.dumps(pipeline_code, indent=2)}

The target operation '{op['name']}' fits into this broader pipeline. Consider:
- What data flows into this operation from previous steps
- How this operation's output will be used by subsequent operations
- The overall goal of the pipeline when creating your improved prompt
"""

        return (
            f"You are an expert at analyzing data patterns and creating precise, effective prompts.\n\n"
            f"Target Operation:\n"
            f"{json.dumps(op, indent=2)}\n\n"
            f"Original Prompt: {original_prompt}\n\n"
            f"{pipeline_context}\n"
            f"Your task is to analyze sample input data and create a significantly improved version of this prompt.\n\n"
            f"You will be given access to sample input data through a read_next_doc() function. Use this to:\n"
            f"1. Understand the actual structure and patterns in the input data\n"
            f"2. Identify ambiguities in the original prompt that could be clarified\n"
            f"3. Discover specific patterns, formats, or edge cases that should be addressed\n"
            f"4. Consider how this operation fits into the broader pipeline context\n"
            f"5. Create a more specific, actionable prompt based on these insights\n\n"
            f"Guidelines for the improved prompt:\n"
            f"- Must preserve ALL original input variable references (like {{{{ input.fieldname }}}})\n"
            f"- Should be significantly more specific than the original\n"
            f"- Include concrete instructions based on patterns observed in the data\n"
            f"- Address potential ambiguities or edge cases discovered in samples\n"
            f"- Consider the pipeline context and how this operation contributes to the overall goal\n"
            f"- Maintain the same general task but with much clearer execution details\n\n"
            f"Example transformation:\n"
            f"{self.example}\n\n"
            f"Analyze samples strategically - focus on diversity and understanding patterns rather than reading every document.\n"
            f"When you have enough information to create a substantially improved prompt, output your result.\n\n"
            f"Remember: Your goal is to make the prompt so clear and specific that it produces more consistent, higher-quality results."
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
        Use agentic approach to analyze sample data and generate improved prompt.
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

        # Create validation function for input variable preservation
        original_prompt = target_ops_configs[0].get("prompt", "")

        def validate_input_variables(schema_instance):
            ClarifyInstructionsInstantiateSchema.validate_input_variables_preserved(
                schema_instance.clarified_prompt, original_prompt
            )

        # Set up agentic runner with validation
        runner = AgenticDirectiveRunner(
            input_data=input_data,
            agent_llm=agent_llm,
            validation_func=validate_input_variables,
        )

        # Create system prompt for the agentic runner
        system_prompt = (
            "You are an expert prompt engineer who analyzes data to create better, more specific prompts. "
            "Your goal is to examine input samples to understand patterns, identify ambiguities, and create "
            "significantly improved prompts that are more specific and actionable than the original. "
            "You also consider the broader pipeline context to ensure the improved prompt serves the overall data processing goal."
        )

        # Create initial user message
        initial_message = self.to_string_for_instantiate(
            target_ops_configs, pipeline_code
        )

        # Run the agentic loop (validation is handled internally)
        try:
            schema, updated_message_history, call_cost = runner.run_agentic_loop(
                system_prompt=system_prompt,
                initial_user_message=initial_message,
                response_schema=ClarifyInstructionsInstantiateSchema,
            )

            # Update message history
            message_history.extend(updated_message_history)

            return schema, message_history, call_cost

        except Exception as e:
            raise Exception(
                f"Failed to instantiate clarify_instructions directive: {str(e)}"
            )

    def apply(
        self,
        global_default_model: str,
        ops_list: List[Dict],
        target_ops: List[str],
        rewrite: ClarifyInstructionsInstantiateSchema,
    ) -> List[Dict]:
        """
        Apply the directive by replacing the target operation's prompt with the clarified version.
        """
        new_ops_list = deepcopy(ops_list)

        # Find and update the target operation
        for i, op in enumerate(new_ops_list):
            if op["name"] in target_ops:
                # Update the prompt with the clarified version
                new_ops_list[i]["prompt"] = rewrite.clarified_prompt
                break

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
        1. Use agentic approach to analyze data and generate improved prompt
        2. Apply the transformation using that improved prompt
        """
        assert (
            len(target_ops) == 1
        ), "ClarifyInstructions directive requires exactly one target operation"
        input_file_path = kwargs.get("input_file_path", None)
        pipeline_code = kwargs.get("pipeline_code", None)

        if not input_file_path:
            raise ValueError(
                "input_file_path is required for ClarifyInstructions directive"
            )

        # Get configuration for target operation
        target_ops_configs = [op for op in operators if op["name"] in target_ops]

        if not target_ops_configs:
            raise ValueError(f"Target operation {target_ops[0]} not found in operators")

        # Step 1: Agent analyzes data and generates improved prompt
        rewrite, message_history, call_cost = self.llm_instantiate(
            target_ops_configs,
            input_file_path,
            agent_llm,
            message_history,
            pipeline_code,
        )

        # Step 2: Apply transformation using the improved prompt
        return (
            self.apply(global_default_model, operators, target_ops, rewrite),
            message_history,
            call_cost,
        )
