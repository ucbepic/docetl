import json
import os
from copy import deepcopy
from typing import Dict, List, Type

from litellm import completion
from pydantic import BaseModel, Field

from docetl.reasoning_optimizer.instantiate_schemas import (
    HierarchicalReduceInstantiateSchema,
)

from .base import MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS, Directive, DirectiveTestCase


class HierarchicalReduceDirective(Directive):
    name: str = Field(
        default="hierarchical_reduce", description="The name of the directive"
    )
    formal_description: str = Field(
        default="Reduce => (Map* ->) Reduce -> Reduce (optionally with Map before first Reduce)"
    )
    nl_description: str = Field(
        default="Transform a reduce operation that aggregates large groups of documents by first aggregating at a finer granularity (reduce_key + additional_key), then rolling up to the desired level (reduce_key only). This hierarchical approach can capture nuances that might be lost in a single large-scale aggregation and allows for intermediate validation."
    )
    when_to_use: str = Field(
        default="When a reduce operation processes many documents per group and it would be beneficial to first aggregate at a finer granularity before rolling up. Useful when there's a semantic hierarchy in the data (e.g., aggregate by state+city first, then by state only) or when you want to prevent information loss in large-scale aggregations. The target operator must be a reduce operator."
    )
    instantiate_schema_type: Type[BaseModel] = HierarchicalReduceInstantiateSchema

    example: str = Field(
        default=(
            "Original Reduce Op:\n"
            "- name: summarize_by_state\n"
            "  type: reduce\n"
            "  reduce_key: state\n"
            "  prompt: |\n"
            "    Summarize voting patterns from these social media posts:\n"
            "    {% for input in inputs %}\n"
            "    Post: {{ input.content }}\n"
            "    {% endfor %}\n"
            "    Return a summary of voting patterns.\n"
            "  output:\n"
            "    schema:\n"
            "      summary: string\n"
            "\n"
            "Example InstantiateSchema (with Map for synthetic key):\n"
            "HierarchicalReduceInstantiateSchema(\n"
            "  map_config=MapOpConfig(\n"
            "    name='extract_city',\n"
            "    prompt='Extract the city mentioned in this post:\\n{{ input.content }} made in this state:\\n{{ input.state }}\\nReturn the city name or \"Unknown\" if not found.',\n"
            "    output_keys=['city'],\n"
            "  ),\n"
            "  additional_key='city',\n"
            "  reduce_1_name='summarize_by_state_city',\n"
            "  # First reduce: Process raw posts at city level\n"
            "  reduce_1_prompt='Goal: Summarize voting patterns from social media posts to understand public sentiment.\\n\\nFor this state and city, analyze these posts:\\n{% for input in inputs %}\\nPost: {{ input.content }}\\n{% endfor %}\\nReturn a summary of voting patterns and key themes.',\n"
            "  # Second reduce: Explicitly work with summaries from first reduce\n"
            "  reduce_2_prompt='Goal: Summarize voting patterns from social media posts to understand public sentiment.\\n\\nWe have already summarized voting patterns at the city level. Your task is to combine these city-level summaries into a comprehensive state-level analysis:\\n{% for input in inputs %}\\nCity: {{ input.city }}\\nCity-Level Summary: {{ input.summary }}\\n{% endfor %}\\nSynthesize these city summaries into a unified state-level summary of voting patterns.',\n"
            ")\n"
            "\n"
            "Example InstantiateSchema (using existing key):\n"
            "HierarchicalReduceInstantiateSchema(\n"
            "  map_config=None,  # No Map needed when using existing key\n"
            "  additional_key='county',  # Assuming 'county' already exists in the data\n"
            "  reduce_1_name='summarize_by_state_county',\n"
            "  # First reduce: Process raw posts at county level\n"
            "  reduce_1_prompt='Goal: Summarize voting patterns from social media posts.\\n\\nAnalyze posts for this state and county:\\n{% for input in inputs %}\\nPost: {{ input.content }}\\n{% endfor %}\\nReturn voting pattern summary for this county.',\n"
            "  # Second reduce: Explicitly work with county summaries\n"
            "  reduce_2_prompt='Goal: Summarize voting patterns from social media posts.\\n\\nWe have already analyzed voting patterns at the county level. Your task is to synthesize these county-level summaries into a state-level overview:\\n{% for input in inputs %}\\nCounty: {{ input.county }}\\nCounty Analysis: {{ input.summary }}\\n{% endfor %}\\nCombine these county analyses into a comprehensive state voting pattern summary.',\n"
            ")"
        ),
    )

    test_cases: List[DirectiveTestCase] = Field(
        default_factory=lambda: [
            DirectiveTestCase(
                name="voting_pattern_aggregation",
                description="Should create hierarchical aggregation for voting patterns",
                input_config={
                    "name": "analyze_voting",
                    "type": "reduce",
                    "reduce_key": "state",
                    "prompt": "Analyze voting sentiments from these posts:\n{% for input in inputs %}\nPost: {{ input.post }}\n{% endfor %}\nReturn voting sentiment analysis.",
                    "output": {"schema": {"analysis": "string"}},
                },
                target_ops=["analyze_voting"],
                expected_behavior="Should create two reduce operations: first by state+additional_key, then by state only, optionally with a Map to create synthetic keys",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="sales_hierarchical_aggregation",
                description="Should create hierarchical aggregation for sales data",
                input_config={
                    "name": "aggregate_sales",
                    "type": "reduce",
                    "reduce_key": "region",
                    "prompt": "Aggregate sales data from these records:\n{% for input in inputs %}\nSales: {{ input.sales_data }}\n{% endfor %}\nReturn total sales metrics.",
                    "output": {"schema": {"metrics": "object"}},
                },
                target_ops=["aggregate_sales"],
                expected_behavior="Should create hierarchical reduce pattern for sales aggregation",
                should_pass=True,
            ),
        ]
    )

    def __eq__(self, other):
        return isinstance(other, HierarchicalReduceDirective)

    def __hash__(self):
        return hash("HierarchicalReduceDirective")

    def to_string_for_instantiate(self, original_op: Dict) -> str:
        """
        Generate a prompt for an agent to instantiate this directive.

        Args:
            original_op (Dict): The original reduce operation.

        Returns:
            str: The agent prompt for instantiating the directive.
        """
        return (
            f"You are an expert at optimizing data processing operations using hierarchical aggregation patterns.\n\n"
            f"Original Operation:\n"
            f"{str(original_op)}\n"
            f"Directive: {self.name}\n"
            f"Your task is to instantiate this directive by creating a hierarchical reduce pattern that:\n"
            f"1. First aggregates data at a finer granularity (original reduce_key + additional_key)\n"
            f"2. Then rolls up these finer-grained aggregations to the desired level (original reduce_key only)\n\n"
            f"Key Requirements:\n"
            f"1. Identify or create an appropriate additional key for finer granularity:\n"
            f"   - Check if there's an existing key in the data that provides meaningful sub-grouping\n"
            f"   - If no suitable key exists, create a Map operation to synthesize one (e.g., extract city from text)\n"
            f"2. Adapt the original reduce prompt for both aggregation levels:\n"
            f"   - reduce_1_prompt: Should aggregate at the finer granularity (both keys) from raw data\n"
            f"   - reduce_2_prompt: Should combine the outputs from reduce_1 to the target granularity\n"
            f"3. IMPORTANT: Both reduce prompts should understand the overall goal from the original operation:\n"
            f"   - Each reduce operation runs independently without access to the original prompt\n"
            f"   - Both prompts should share a common context/prefix that explains the overall goal\n"
            f"   - The second reduce MUST explicitly acknowledge it's working with summaries/aggregations from the first reduce\n"
            f"   - Example: 'We have already summarized X at the Y level. Your task is to combine these Y-level summaries...'\n"
            f"4. Both reduce operations should produce the same output schema as the original\n"
            f"5. The reduce_2_prompt must reference the outputs from reduce_1, not the original documents\n\n"
            f"This hierarchical approach is especially useful when:\n"
            f"- There are many documents per group in the original reduce\n"
            f"- There's a natural hierarchy in the data (geographic, temporal, categorical)\n"
            f"- You want to prevent information loss in large-scale aggregations\n\n"
            f"Example:\n"
            f"{self.example}\n\n"
            f"Please output the HierarchicalReduceInstantiateSchema with the hierarchical aggregation details."
        )

    def llm_instantiate(
        self, original_op: Dict, agent_llm: str, message_history: list = []
    ):
        """
        Use LLM to instantiate this directive by creating a hierarchical reduce pattern.

        Args:
            original_op (Dict): The original reduce operation.
            agent_llm (str): The LLM model to use.
            message_history (List, optional): Conversation history for context.

        Returns:
            HierarchicalReduceInstantiateSchema: The structured output from the LLM.
        """

        message_history.extend(
            [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant for document processing pipelines specializing in hierarchical aggregation patterns.",
                },
                {
                    "role": "user",
                    "content": self.to_string_for_instantiate(original_op),
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
                response_format=HierarchicalReduceInstantiateSchema,
            )

            call_cost = resp._hidden_params["response_cost"]

            try:
                parsed_res = json.loads(resp.choices[0].message.content)
                schema = HierarchicalReduceInstantiateSchema(**parsed_res)

                # Validate that if map_config is provided, additional_key should match one of the output keys
                if schema.map_config:
                    if len(schema.map_config.output_keys) != 1:
                        raise ValueError(
                            "Map config must have exactly one output key for hierarchical reduce"
                        )
                    if schema.additional_key != schema.map_config.output_keys[0]:
                        raise ValueError(
                            f"When creating a synthetic key with Map, additional_key must match the map output key '{schema.map_config.output_keys[0]}'"
                        )

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
        global_default_model,
        ops_list: List[Dict],
        target_op: str,
        rewrite: HierarchicalReduceInstantiateSchema,
    ) -> List[Dict]:
        """
        Apply the directive to the pipeline config.
        """
        # Create a copy of the pipeline config
        new_ops_list = deepcopy(ops_list)

        # Find position of the target reduce op to modify
        pos_to_modify = None
        orig_op = None
        for i, op in enumerate(ops_list):
            if op["name"] == target_op:
                pos_to_modify = i
                orig_op = op
                break

        if pos_to_modify is None:
            raise ValueError(
                f"Target operation '{target_op}' not found in operations list"
            )

        # Determine the model to use
        default_model = orig_op.get("model", global_default_model)

        operations_to_insert = []

        # Create the optional Map operation if specified
        if rewrite.map_config:
            new_map_op = {
                "name": rewrite.map_config.name,
                "type": "map",
                "prompt": rewrite.map_config.prompt,
                "model":  default_model,
                "litellm_completion_kwargs": {"temperature": 0},
                "output": {"schema": {rewrite.map_config.output_keys[0]: "string"}},
            }
            operations_to_insert.append(new_map_op)

        # Create the first reduce operation (finer granularity)
        first_reduce_op = deepcopy(orig_op)
        first_reduce_op["name"] = rewrite.reduce_1_name
        first_reduce_op["reduce_key"] = (
            [orig_op["reduce_key"], rewrite.additional_key]
            if isinstance(orig_op["reduce_key"], str)
            else orig_op["reduce_key"] + [rewrite.additional_key]
        )
        first_reduce_op["prompt"] = rewrite.reduce_1_prompt
        first_reduce_op["model"] = default_model
        operations_to_insert.append(first_reduce_op)

        # Create the second reduce operation (target granularity)
        second_reduce_op = deepcopy(orig_op)
        second_reduce_op["prompt"] = rewrite.reduce_2_prompt
        second_reduce_op["model"] = default_model
        operations_to_insert.append(second_reduce_op)

        # Replace the original operation with the new operations
        new_ops_list[pos_to_modify : pos_to_modify + 1] = operations_to_insert

        return new_ops_list

    def instantiate(
        self,
        operators: List[Dict],
        target_ops: List[str],
        agent_llm: str,
        message_history: list = [],
        optimize_goal="acc",
        global_default_model: str = None,
        **kwargs,
    ):
        """
        Instantiate the directive for a list of operators.
        """
        # Assert that there is only one target op
        assert (
            len(target_ops) == 1
        ), "There must be exactly one target op to instantiate this hierarchical reduce directive"
        target_op_config = [op for op in operators if op["name"] == target_ops[0]][0]

        # Ensure it's a reduce operation
        if target_op_config.get("type") != "reduce":
            raise ValueError(
                f"Target operation '{target_ops[0]}' must be a reduce operation"
            )

        # Instantiate the directive
        rewrite, message_history, call_cost = self.llm_instantiate(
            target_op_config,
            agent_llm,
            message_history,
        )

        # Apply the rewrite to the operators
        new_ops_plan = self.apply(
            global_default_model, operators, target_ops[0], rewrite
        )
        return new_ops_plan, message_history, call_cost 
