import json
import os
from copy import deepcopy
from typing import Dict, List, Type

from litellm import completion
from pydantic import BaseModel, Field

from docetl.reasoning_optimizer.instantiate_schemas import (
    OperatorFusionInstantiateSchema,
)

from .base import MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS, Directive, DirectiveTestCase


class OperatorFusionDirective(Directive):
    name: str = Field(
        default="operator_fusion", description="The name of the directive"
    )
    formal_description: str = Field(default="Op1 -> Op2 => Op2")
    nl_description: str = Field(
        default="Combines two sequential operations into a single operation to reduce LLM processing costs by avoiding duplicate document reads and API calls"
    )
    when_to_use: str = Field(
        default="When you have two sequential LLM operations processing the same documents keys and want to optimize cost by combining them into one operation that performs both tasks. The target operators should be two consecutive operations.  Make sure you specify two operators when choosing this directive."
    )

    instantiate_schema_type: Type[BaseModel] = Field(
        default=OperatorFusionInstantiateSchema
    )

    example: str = Field(
        default="""
        Example 1 - Map + Filter Fusion:
        Original: extract_sentiment (map) → filter_positive (filter)
        Agent output: {"fused_prompt": "Extract sentiment from {{ input.review }} AND determine if it's positive (true/false)"}

        Example 2 - Map + Map Fusion:
        Original: extract_entities (map) → classify_urgency (map)
        Agent output: {"fused_prompt": "Extract entities from {{ input.text }} AND classify urgency level"}

        Example 3 - Map + Reduce Fusion:
        Original: extract_themes (map) → summarize_themes (reduce)
        Agent output: {"fused_prompt": "For each group of feedback, extract themes and summarize them: {% for item in inputs %}{{ item.feedback }}{% endfor %}"}
        """
    )

    test_cases: List[DirectiveTestCase] = Field(
        default_factory=lambda: [
            DirectiveTestCase(
                name="sentiment_analysis_fusion",
                description="Fuse sentiment extraction + quality filter",
                input_config=[
                    {
                        "name": "extract_sentiment",
                        "type": "map",
                        "prompt": "What is the sentiment of this review? {{ input.review_text }}",
                        "output": {
                            "schema": {"sentiment": "string", "confidence": "float"}
                        },
                        "model": "gpt-4o-mini",
                    },
                    {
                        "name": "filter_confident",
                        "type": "filter",
                        "prompt": "Is this sentiment confident (>0.8)? Confidence: {{ input.confidence }}",
                        "output": {"schema": {"is_confident": "boolean"}},
                        "model": "gpt-4o-mini",
                    },
                ],
                target_ops=["extract_sentiment", "filter_confident"],
                expected_behavior="Should create map + code_filter for sentiment extraction with confidence filtering",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="document_processing_fusion",
                description="Fuse document classification + summarization",
                input_config=[
                    {
                        "name": "classify_doc",
                        "type": "map",
                        "prompt": "Classify this document: {{ input.content }}",
                        "output": {"schema": {"category": "string"}},
                        "model": "gpt-4o-mini",
                    },
                    {
                        "name": "summarize_doc",
                        "type": "map",
                        "prompt": "Summarize this document: {{ input.content }}",
                        "output": {"schema": {"summary": "string"}},
                        "model": "gpt-4o-mini",
                    },
                ],
                target_ops=["classify_doc", "summarize_doc"],
                expected_behavior="Should create single map combining classification and summarization",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="extract_and_aggregate_fusion",
                description="Fuse entity extraction + aggregation",
                input_config=[
                    {
                        "name": "extract_mentions",
                        "type": "map",
                        "prompt": "Extract company mentions from: {{ input.article }}",
                        "output": {"schema": {"companies": "list[str]"}},
                        "model": "gpt-4o-mini",
                    },
                    {
                        "name": "count_mentions",
                        "type": "reduce",
                        "reduce_key": "topic",
                        "prompt": "Count company mentions: {% for item in inputs %}{{ item.companies }}{% endfor %}",
                        "output": {"schema": {"mention_counts": "str"}},
                        "model": "gpt-4o-mini",
                    },
                ],
                target_ops=["extract_mentions", "count_mentions"],
                expected_behavior="Should create single reduce combining extraction and counting",
                should_pass=True,
            ),
        ]
    )

    def __eq__(self, other):
        return isinstance(other, OperatorFusionDirective)

    def __hash__(self):
        return hash("OperatorFusionDirective")

    def to_string_for_instantiate(self, original_ops: List[Dict]) -> str:
        """
        Generate a prompt that asks the agent to output the instantiate schema.
        """
        op1, op2 = original_ops[0], original_ops[1]

        return (
            f"You are an expert at optimizing data processing pipelines for cost efficiency.\n\n"
            f"Two Sequential Operations:\n"
            f"Operation 1: {op1}\n"
            f"Operation 2: {op2}\n\n"
            f"Your task is to fuse these two operations into a single operation that performs both tasks, "
            f"reducing LLM API calls and processing costs.\n\n"
            f"Create a combined prompt that:\n"
            f"1. Performs the logic of both operations in one LLM call\n"
            f"2. Uses the same input references as the original operations\n"
            f"3. If either operation is a filter, include boolean logic for filtering\n"
            f"4. Maintains the same output schema requirements\n\n"
            f"IMPORTANT: If either operation is a filter, your fused prompt MUST include logic that "
            f"outputs a boolean field for filtering purposes. A code_filter will be automatically added.\n\n"
            f"Example outputs:\n"
            f"{self.example}\n\n"
            f"Please output only the OperatorFusionInstantiateSchema with 'fused_prompt' field "
            f"that specifies how to combine these operations efficiently."
        )

    def llm_instantiate(
        self,
        original_ops: List[Dict],
        agent_llm: str,
        message_history: list = [],
    ) -> tuple:
        """
        Call the LLM to generate the instantiate schema.
        """
        message_history.extend(
            [
                {
                    "role": "user",
                    "content": self.to_string_for_instantiate(original_ops),
                },
            ]
        )

        for _ in range(MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS):
            resp = completion(
                model=agent_llm,
                messages=message_history,
                api_key=os.environ.get("AZURE_API_KEY"),
                api_base=os.environ.get("AZURE_API_BASE"),
                api_version=os.environ.get("AZURE_API_VERSION"),
                azure=True,
                response_format=OperatorFusionInstantiateSchema,
            )
            call_cost = resp._hidden_params["response_cost"]
            try:
                parsed_res = json.loads(resp.choices[0].message.content)
                schema = OperatorFusionInstantiateSchema(**parsed_res)
                message_history.append(
                    {"role": "assistant", "content": resp.choices[0].message.content}
                )
                return schema, message_history, call_cost
            except Exception as err:
                error_message = f"Validation error: {err}\nPlease try again."
                message_history.append({"role": "user", "content": error_message})

        raise Exception(
            f"Failed to instantiate directive after {MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS} attempts."
        )

    def apply(
        self,
        global_default_model: str,
        ops_list: List[Dict],
        target_ops: List[str],
        rewrite: OperatorFusionInstantiateSchema,
    ) -> List[Dict]:
        """
        Apply the fusion directive by combining two operations and optionally adding code_filter.
        """
        assert (
            len(target_ops) == 2
        ), "Operator fusion requires exactly two target operations"

        new_ops_list = deepcopy(ops_list)
        op1_name, op2_name = target_ops[0], target_ops[1]


        # Find the operations
        op1_idx = next(i for i, op in enumerate(ops_list) if op["name"] == op1_name)
        op2_idx = next(i for i, op in enumerate(ops_list) if op["name"] == op2_name)
        op1, op2 = ops_list[op1_idx], ops_list[op2_idx]

        # Determine fused operation type and schema based on the combination
        op1_type, op2_type = op1.get("type"), op2.get("type")

        assert (
            op1_type != "reduce" and op2_type != "reduce"
        ), "Cannot apply fusion on reduce"

        default_model = op1.get("model", global_default_model)

        # Create base fused operation
        fused_op = {
            "name": f"fused_{op1_name}_{op2_name}",
            "prompt": rewrite.fused_prompt,
            "model": default_model,
            "litellm_completion_kwargs": {"temperature": 0},
        }

        needs_code_filter = False

        # Determine type, schema, and code_filter need based on combination
        if op1_type == "map" and op2_type == "map":
            # map + map => fuse into one map
            fused_op["type"] = "map"
            fused_op["output"] = {
                "schema": {**op1["output"]["schema"], **op2["output"]["schema"]}
            }
            needs_code_filter = False
        elif (op1_type == "map" and op2_type == "filter") or (
            op1_type == "filter" and op2_type == "map"
        ):
            # map + filter OR filter + map => fuse into map (with union of schemas) + code filter
            fused_op["type"] = "map"
            fused_op["output"] = {
                "schema": {**op1["output"]["schema"], **op2["output"]["schema"]}
            }
            needs_code_filter = True

        elif op1_type == "filter" and op2_type == "filter":
            # filter + filter => fuse into one filter with bool output
            fused_op["type"] = "filter"
            fused_op["output"] = {"schema": {"_bool": "bool"}}
            needs_code_filter = False

        # Replace the original operations
        if op1_idx < op2_idx:
            # Remove in reverse order to maintain indices
            new_ops_list.pop(op2_idx)
            new_ops_list.pop(op1_idx)
            new_ops_list.insert(op1_idx, fused_op)
        else:
            new_ops_list.pop(op1_idx)
            new_ops_list.pop(op2_idx)
            new_ops_list.insert(op2_idx, fused_op)

        # Add code_filter if needed
        if needs_code_filter:
            # Get the filter field name from the filter operation
            filter_op = op1 if op1.get("type") == "filter" else op2
            filter_field = list(filter_op["output"]["schema"].keys())[0]

            code_filter_op = {
                "name": f"filter_{fused_op['name']}",
                "type": "code_filter",
                "code": f"""
def transform(input_doc):
    return input_doc.get('{filter_field}', False)
""",
            }

            # Insert code_filter after the fused operation
            fused_idx = next(
                i for i, op in enumerate(new_ops_list) if op["name"] == fused_op["name"]
            )
            new_ops_list.insert(fused_idx + 1, code_filter_op)

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
        assert (
            len(target_ops) == 2
        ), "Operator fusion requires exactly two target operations"

        # Get the two operations to fuse
        target_op_configs = [op for op in operators if op["name"] in target_ops]

        # Ensure they are in the correct order
        target_op_configs.sort(key=lambda op: target_ops.index(op["name"]))

        # Step 1: Agent generates the instantiate schema
        rewrite, message_history, call_cost = self.llm_instantiate(
            target_op_configs, agent_llm, message_history
        )

        # Step 2: Apply transformation using the schema
        return (
            self.apply(global_default_model, operators, target_ops, rewrite),
            message_history,
            call_cost
        )
