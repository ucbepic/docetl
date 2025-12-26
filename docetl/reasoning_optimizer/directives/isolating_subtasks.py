import json
import os
import re
from copy import deepcopy
from typing import Dict, List, Type

from litellm import completion
from pydantic import BaseModel, Field

from docetl.reasoning_optimizer.instantiate_schemas import (
    IsolatingSubtasksInstantiateSchema,
)

from .base import MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS, Directive, DirectiveTestCase


class IsolatingSubtasksDirective(Directive):
    name: str = Field(
        default="isolating_subtasks", description="The name of the directive"
    )
    formal_description: str = Field(default="Map => Parallel Map -> Map")
    nl_description: str = Field(
        default="Rewrites a single Map into a Parallel Map that isolates subtasks and generates separate outputs for each, followed by a Map that aggregates or synthesizes the results."
    )
    when_to_use: str = Field(
        default="When the original Map is overloaded—either the prompt asks for many different things OR the output schema has many fields—and subtasks are better handled independently (e.g., extract each attribute in parallel, then combine into a unified output)."
    )

    instantiate_schema_type: Type[BaseModel] = Field(
        default=IsolatingSubtasksInstantiateSchema
    )

    example: str = Field(
        default="""
        Original Op (MapOpConfig):
        - name: extract_contract_info
          type: map
          prompt: |
            Extract the following from this contract: {{ input.document }}
            - party names
            - agreement date
            - governing law
            - termination clauses
          output:
            schema:
              parties: "string"
              agreement_date: "string"
              governing_law: "string"
              termination_clauses: "string"

        Example InstantiateSchema (what the agent should output):
        IsolatingSubtasksConfig(
            subtasks=[
                SubtaskConfig(
                    name="Extract Basic Contract Info",
                    prompt="Extract party names and agreement date from: {{ input.document }}",
                    output_keys=["parties", "agreement_date"]
                ),
                SubtaskConfig(
                    name="Extract Legal Terms",
                    prompt="Extract governing law and termination clauses from: {{ input.document }}",
                    output_keys=["governing_law", "termination_clauses"]
                )
            ],
            aggregation_prompt="Combine the basic info {{ input.subtask_1_output }} with legal terms {{ input.subtask_2_output }} into the final contract summary."
        )
        """
    )

    test_cases: List[DirectiveTestCase] = Field(
        default_factory=lambda: [
            DirectiveTestCase(
                name="complex_prompt_simple_output",
                description="Complex multi-task prompt but simple list output - should isolate by prompt complexity",
                input_config={
                    "name": "analyze_document",
                    "type": "map",
                    "prompt": "Analyze this document for: 1) sentiment and emotional tone, 2) key topics and themes, 3) factual accuracy and bias, 4) writing quality and readability, 5) actionable insights and recommendations. Document: {{ input.text }}",
                    "output": {"schema": {"results": "list[string]"}},
                },
                target_ops=["analyze_document"],
                expected_behavior="Should create >1 parallel map prompts covering all analysis aspects (sentiment, topics, bias, quality, insights) and aggregation prompt referencing all subtask outputs",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="contract_analysis_many_fields",
                description="Legal contract extraction with many specific fields",
                input_config={
                    "name": "extract_contract_terms",
                    "type": "map",
                    "prompt": "Extract contract information from: {{ input.contract_text }}",
                    "output": {
                        "schema": {
                            "parties": "string",
                            "agreement_date": "string",
                            "governing_law": "string",
                            "termination_clause": "string",
                            "payment_terms": "string",
                            "liability_cap": "string",
                        }
                    },
                },
                target_ops=["extract_contract_terms"],
                expected_behavior="Should create >1 parallel map prompts covering all 6 fields (parties, agreement_date, governing_law, termination_clause, payment_terms, liability_cap) and aggregation prompt referencing all subtask outputs",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="medical_transcript_processing",
                description="Medical data extraction - different subtasks for different medical info types",
                input_config={
                    "name": "process_medical_record",
                    "type": "map",
                    "prompt": "Extract patient demographics, symptoms, diagnosis, and treatment plan from: {{ input.transcript }}",
                    "output": {
                        "schema": {
                            "patient_info": "string",
                            "symptoms": "string",
                            "diagnosis": "string",
                            "treatment": "string",
                        }
                    },
                },
                target_ops=["process_medical_record"],
                expected_behavior="Should create >1 parallel map prompts covering all medical aspects (demographics, symptoms, diagnosis, treatment) and aggregation prompt referencing all subtask outputs",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="research_paper_summary",
                description="Academic paper analysis with focus on different aspects",
                input_config={
                    "name": "summarize_research",
                    "type": "map",
                    "prompt": "Analyze this research paper for methodology, key findings, limitations, and practical applications: {{ input.paper_text }}",
                    "output": {
                        "schema": {"summary": "string", "key_points": "list[string]"}
                    },
                },
                target_ops=["summarize_research"],
                expected_behavior="Should create >1 parallel map prompts covering all research aspects (methodology, findings, limitations, applications) and aggregation prompt referencing all subtask outputs",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="customer_feedback_analysis",
                description="Multi-aspect customer feedback analysis with simple output",
                input_config={
                    "name": "analyze_feedback",
                    "type": "map",
                    "prompt": "Analyze customer feedback for: product quality issues, service experience problems, pricing concerns, feature requests, and overall satisfaction. Feedback: {{ input.feedback_text }}",
                    "output": {
                        "schema": {
                            "insights": "list[string]",
                            "priority_score": "string",
                        }
                    },
                },
                target_ops=["analyze_feedback"],
                expected_behavior="Should create >1 parallel map prompts covering all feedback aspects (quality, service, pricing, features, satisfaction) and aggregation prompt referencing all subtask outputs",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="financial_report_extraction",
                description="Financial document with many specific metrics to extract",
                input_config={
                    "name": "extract_financial_data",
                    "type": "map",
                    "prompt": "Extract financial metrics from earnings report: {{ input.report }}",
                    "output": {
                        "schema": {
                            "revenue": "string",
                            "profit_margin": "string",
                            "cash_flow": "string",
                            "debt_ratio": "string",
                            "growth_rate": "string",
                            "market_share": "string",
                        }
                    },
                },
                target_ops=["extract_financial_data"],
                expected_behavior="Should create >1 parallel map prompts covering all 6 financial metrics (revenue, profit_margin, cash_flow, debt_ratio, growth_rate, market_share) and aggregation prompt referencing all subtask outputs",
                should_pass=True,
            ),
        ]
    )

    def __eq__(self, other):
        return isinstance(other, IsolatingSubtasksDirective)

    def __hash__(self):
        return hash("IsolatingSubtasksDirective")

    def to_string_for_instantiate(self, original_op: Dict) -> str:
        """
        Generate a prompt that asks the agent to output the instantiate schema.
        """
        # Extract original operation details
        original_name = original_op.get("name", "unknown")
        original_prompt = original_op.get("prompt", "")
        original_output_schema = original_op.get("output", {}).get("schema", {})
        original_output_keys = (
            list(original_output_schema.keys()) if original_output_schema else []
        )

        # Find the input key from the original prompt (look for {{ input.XXX }} pattern)

        input_matches = re.findall(r"\{\{\s*input\.([^}\s]+)\s*\}\}", original_prompt)
        input_key = input_matches[0] if input_matches else "document"

        return (
            f"You are an expert at analyzing overloaded map operations and breaking them into focused subtasks.\n\n"
            f"Original Operation:\n"
            f"Name: {original_name}\n"
            f"Prompt: {original_prompt}\n"
            f"Output Schema: {original_output_schema}\n"
            f"Output Keys: {original_output_keys}\n\n"
            f"This map operation is overloaded - either the prompt asks for many different things "
            f"OR it has {len(original_output_keys)} output fields to generate. "
            f"Your task is to create an IsolatingSubtasksConfig with:\n\n"
            f"1. **SUBTASKS**: Group the {len(original_output_keys)} output fields into 2-4 logical subtasks "
            f"where each subtask handles related fields that can be processed independently:\n"
            f"   - Each subtask needs a descriptive 'name'\n"
            f"   - Each subtask needs a focused Jinja 'prompt' that uses {{{{ input.{input_key} }}}} (same as original)\n"
            f"   - Each subtask needs 'output_keys' listing which fields it extracts\n"
            f"   - Every original output key must appear in exactly one subtask's output_keys\n\n"
            f"2. **AGGREGATION_PROMPT**: A Jinja template that combines all subtask results:\n"
            f"   - Must reference ALL subtask outputs as {{{{ input.subtask_1_output }}}}, {{{{ input.subtask_2_output }}}}, etc.\n"
            f"   - Should synthesize/combine the subtask results into the final output\n"
            f"   - Final result must match the original output schema exactly\n"
            f"   - Example: 'Combine basic info {{{{ input.subtask_1_output }}}} with details {{{{ input.subtask_2_output }}}} into complete result.'\n\n"
            f"CRITICAL REQUIREMENTS:\n"
            f"- All {len(original_output_keys)} original output keys must be covered by subtasks\n"
            f"- Subtask prompts must use {{{{ input.{input_key} }}}} (same input as original)\n"
            f"- Aggregation prompt must reference {{{{ input.subtask_N_output }}}} for each subtask\n"
            f"- No information should be lost in the transformation\n\n"
            f"Example logical grouping for contract extraction:\n"
            f"- Subtask 1: Basic info (parties, dates) \n"
            f"- Subtask 2: Legal terms (governing law, clauses)\n"
            f"- Subtask 3: Commercial terms (pricing, commitments)\n\n"
            f"Please output the IsolatingSubtasksConfig that transforms this overloaded operation."
        )

    def llm_instantiate(
        self,
        original_op: Dict,
        agent_llm: str,
        message_history: list = [],
    ):
        """
        Call the LLM to generate the instantiate schema with validation.
        """
        message_history.extend(
            [
                {
                    "role": "user",
                    "content": self.to_string_for_instantiate(original_op),
                },
            ]
        )

        original_output_keys = list(
            original_op.get("output", {}).get("schema", {}).keys()
        )

        for _ in range(MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS):
            resp = completion(
                model=agent_llm,
                messages=message_history,
                api_key=os.environ.get("AZURE_API_KEY"),
                api_base=os.environ.get("AZURE_API_BASE"),
                api_version=os.environ.get("AZURE_API_VERSION"),
                azure=True,
                response_format=IsolatingSubtasksInstantiateSchema,
            )
            call_cost = resp._hidden_params["response_cost"]
            try:
                parsed_res = json.loads(resp.choices[0].message.content)
                schema = IsolatingSubtasksInstantiateSchema(**parsed_res)

                # Use the schema's validation methods
                schema.validate_subtasks_coverage(original_output_keys)
                schema.validate_aggregation_references_all_subtasks()

                message_history.append(
                    {"role": "assistant", "content": resp.choices[0].message.content}
                )
                return schema, message_history, call_cost

            except Exception as err:
                error_message = f"Validation error: {err}\nPlease ensure all original output keys are covered by subtasks and try again."
                message_history.append({"role": "user", "content": error_message})

        raise Exception(
            f"Failed to instantiate directive after {MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS} attempts."
        )

    def apply(
        self,
        global_default_model,
        ops_list: List[Dict],
        target_op: str,
        rewrite: IsolatingSubtasksInstantiateSchema,
    ) -> List[Dict]:
        """
        Apply the directive by replacing the target map with parallel map + aggregation map.
        """
        new_ops_list = deepcopy(ops_list)

        # Find the target operation
        pos_to_replace = None
        original_op = None
        for i, op in enumerate(ops_list):
            if op["name"] == target_op:
                pos_to_replace = i
                original_op = op
                break

        if pos_to_replace is None:
            raise ValueError(f"Target operation '{target_op}' not found")

        # Create the parallel map operation
        parallel_map_op = {
            "name": f"{target_op}_parallel",
            "type": "parallel_map",
            "litellm_completion_kwargs": {"temperature": 0},
            "prompts": [],
        }

        assert original_op
        # Copy over other fields from original operation (sample, random_sample, etc.)
        for key, value in original_op.items():
            if key not in ["name", "type", "prompt", "output"]:
                parallel_map_op[key] = value

        # Set up output schema for parallel map
        parallel_output_schema = {}

        # Add each subtask as a prompt in the parallel map
        for i, subtask in enumerate(rewrite.subtasks, 1):
            subtask_output_key = f"subtask_{i}_output"

            prompt_config = {
                "name": subtask.name,
                "output_keys": [subtask_output_key],
                "prompt": subtask.prompt,
            }

            parallel_map_op["prompts"].append(prompt_config)
            parallel_output_schema[subtask_output_key] = "string"

        # Set the output schema for parallel map
        parallel_map_op["output"] = {"schema": parallel_output_schema}

        # Use the same model as the original operation
        default_model = original_op.get("model", global_default_model)
        parallel_map_op["model"] = default_model

        # Check if aggregation is needed by comparing subtask output keys with original keys
        subtask_output_keys = set()
        for subtask in rewrite.subtasks:
            subtask_output_keys.update(subtask.output_keys)

        # Check if aggregation is needed: aggregation_prompt is empty
        if not rewrite.aggregation_prompt.strip():
            # Just return the parallel map - it already produces the right output
            parallel_map_op["output"] = original_op.get("output", {})
            new_ops_list[pos_to_replace : pos_to_replace + 1] = [parallel_map_op]
        else:
            # Need aggregation step
            aggregation_map_op = {
                "name": f"{target_op}_aggregate",
                "type": "map",
                "prompt": rewrite.aggregation_prompt,
                "litellm_completion_kwargs": {"temperature": 0},
                "output": original_op.get(
                    "output", {}
                ),  # Same output schema as original
            }

            # Use the same model as the original operation
            aggregation_map_op["model"] = default_model

            # Replace the original operation with both operations
            new_ops_list[pos_to_replace : pos_to_replace + 1] = [
                parallel_map_op,
                aggregation_map_op,
            ]

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
        Main method that orchestrates directive instantiation.
        """
        assert len(target_ops) == 1, "This directive requires exactly one target op"

        target_op_config = [op for op in operators if op["name"] == target_ops[0]][0]

        # Step 1: Agent generates the instantiate schema
        rewrite, message_history, call_cost = self.llm_instantiate(
            target_op_config, agent_llm, message_history
        )

        # Step 2: Apply transformation using the schema
        return (
            self.apply(global_default_model, operators, target_ops[0], rewrite),
            message_history,
            call_cost,
        )
