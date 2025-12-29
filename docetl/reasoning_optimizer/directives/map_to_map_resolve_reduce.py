import json
from copy import deepcopy
from typing import Dict, List, Type

from litellm import completion
from pydantic import BaseModel, Field

from docetl.reasoning_optimizer.instantiate_schemas import (
    MapToMapResolveReduceInstantiateSchema,
)

from .base import MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS, Directive, DirectiveTestCase


class MapToMapResolveReduceDirective(Directive):
    name: str = Field(
        default="map_to_map_resolve_reduce", description="The name of the directive"
    )
    formal_description: str = Field(default="Map -> Reduce => Map -> Resolve -> Reduce")
    nl_description: str = Field(
        default="Insert a Resolve operation between Map and Reduce to deduplicate or normalize entities before aggregation. The Resolve operation uses code-powered blocking conditions to efficiently identify which pairs to compare, avoiding O(n^2) comparisons. This is useful when the Map output contains duplicate or near-duplicate entities that should be merged before the Reduce step."
    )
    when_to_use: str = Field(
        default="When a Map operation produces outputs that may contain duplicates, variations, or near-duplicates (e.g., different spellings of names, similar categories), and these should be normalized before the Reduce aggregation step. The target must be a Map operation followed by a Reduce operation."
    )
    instantiate_schema_type: Type[BaseModel] = MapToMapResolveReduceInstantiateSchema

    example: str = Field(
        default=(
            "Original Pipeline:\n"
            "- name: extract_companies\n"
            "  type: map\n"
            "  prompt: |\n"
            "    Extract company names from this news article:\n"
            "    {{ input.article }}\n"
            "  output:\n"
            "    schema:\n"
            "      company_name: string\n"
            "\n"
            "- name: aggregate_companies\n"
            "  type: reduce\n"
            "  reduce_key: sector\n"
            "  prompt: |\n"
            "    List all unique companies in this sector:\n"
            "    {% for input in inputs %}\n"
            "    - {{ input.company_name }}\n"
            "    {% endfor %}\n"
            "  output:\n"
            "    schema:\n"
            "      companies: list[str]\n"
            "\n"
            "Example InstantiateSchema:\n"
            "MapToMapResolveReduceInstantiateSchema(\n"
            "  resolve_name='normalize_company_names',\n"
            "  comparison_prompt='''Are these two company names referring to the same company?\n"
            "Company 1: {{ input1.company_name }}\n"
            "Company 2: {{ input2.company_name }}\n"
            "Consider variations like abbreviations (IBM vs International Business Machines), \n"
            "different legal suffixes (Inc, Corp, LLC), and common misspellings.''',\n"
            "  resolution_prompt='''Given these variations of a company name:\n"
            "{% for input in inputs %}\n"
            "- {{ input.company_name }}\n"
            "{% endfor %}\n"
            "Return the canonical/official company name.''',\n"
            "  blocking_conditions=[\n"
            "    \"input1['company_name'][:3].lower() == input2['company_name'][:3].lower()\",\n"
            "    \"input1['company_name'].split()[0].lower() == input2['company_name'].split()[0].lower()\",\n"
            "  ],\n"
            "  blocking_keys=['company_name'],\n"
            "  limit_comparisons=1000,\n"
            "  output_schema={'company_name': 'string'},\n"
            ")"
        ),
    )

    test_cases: List[DirectiveTestCase] = Field(
        default_factory=lambda: [
            DirectiveTestCase(
                name="company_name_normalization",
                description="Should insert resolve between map and reduce for company names",
                input_config=[
                    {
                        "name": "extract_companies",
                        "type": "map",
                        "prompt": "Extract company name from: {{ input.text }}",
                        "output": {"schema": {"company_name": "string"}},
                    },
                    {
                        "name": "aggregate_by_sector",
                        "type": "reduce",
                        "reduce_key": "sector",
                        "prompt": "List companies:\n{% for input in inputs %}\n- {{ input.company_name }}\n{% endfor %}",
                        "output": {"schema": {"companies": "list[str]"}},
                    },
                ],
                target_ops=["extract_companies", "aggregate_by_sector"],
                expected_behavior="Should create a resolve operation between the map and reduce with appropriate blocking conditions",
                should_pass=True,
            ),
        ]
    )

    def __eq__(self, other):
        return isinstance(other, MapToMapResolveReduceDirective)

    def __hash__(self):
        return hash("MapToMapResolveReduceDirective")

    def to_string_for_instantiate(self, map_op: Dict, reduce_op: Dict) -> str:
        """
        Generate a prompt for an agent to instantiate this directive.

        Args:
            map_op (Dict): The map operation configuration.
            reduce_op (Dict): The reduce operation configuration.

        Returns:
            str: The agent prompt for instantiating the directive.
        """
        return (
            f"You are an expert at optimizing data processing pipelines by inserting entity resolution steps.\n\n"
            f"Map Operation:\n"
            f"{str(map_op)}\n\n"
            f"Reduce Operation:\n"
            f"{str(reduce_op)}\n\n"
            f"Directive: {self.name}\n"
            f"Your task is to insert a Resolve operation between the Map and Reduce to deduplicate/normalize entities.\n\n"
            f"Key Requirements:\n"
            f"1. Create a comparison_prompt that determines if two items from the Map output are duplicates/variations:\n"
            f"   - Must reference {{ input1.key }} and {{ input2.key }} for comparing fields\n"
            f"   - Should handle common variations (abbreviations, misspellings, formatting differences)\n\n"
            f"2. Create a resolution_prompt that merges matched items into a canonical form:\n"
            f"   - Must use {{% for input in inputs %}} to iterate over matched items\n"
            f"   - Should produce the most authoritative/complete version\n\n"
            f"3. Create blocking_conditions to avoid O(n^2) comparisons:\n"
            f"   - These are Python expressions with access to 'input1' and 'input2' dicts\n"
            f"   - They should filter pairs to only those likely to match\n"
            f"   - Examples:\n"
            f"     * \"input1['name'][:3].lower() == input2['name'][:3].lower()\" (first 3 chars match)\n"
            f"     * \"input1['name'].split()[0].lower() == input2['name'].split()[0].lower()\" (first word matches)\n"
            f"     * \"abs(len(input1['name']) - len(input2['name'])) < 10\" (similar length)\n"
            f"   - Multiple conditions are OR'd together\n\n"
            f"4. Set limit_comparisons to cap the number of pairs (recommended: 500-2000)\n\n"
            f"5. The output_schema should match what the Reduce operation expects from each input\n\n"
            f"Example:\n"
            f"{self.example}\n\n"
            f"Please output the MapToMapResolveReduceInstantiateSchema."
        )

    def llm_instantiate(
        self, map_op: Dict, reduce_op: Dict, agent_llm: str, message_history: list = []
    ):
        """
        Use LLM to instantiate this directive.

        Args:
            map_op (Dict): The map operation configuration.
            reduce_op (Dict): The reduce operation configuration.
            agent_llm (str): The LLM model to use.
            message_history (List, optional): Conversation history for context.

        Returns:
            MapToMapResolveReduceInstantiateSchema: The structured output from the LLM.
        """

        message_history.extend(
            [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant for document processing pipelines specializing in entity resolution.",
                },
                {
                    "role": "user",
                    "content": self.to_string_for_instantiate(map_op, reduce_op),
                },
            ]
        )

        for _ in range(MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS):

            resp = completion(
                model=agent_llm,
                messages=message_history,
                response_format=MapToMapResolveReduceInstantiateSchema,
            )

            call_cost = resp._hidden_params.get("response_cost", 0)

            try:
                parsed_res = json.loads(resp.choices[0].message.content)
                schema = MapToMapResolveReduceInstantiateSchema(**parsed_res)

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
        map_op_name: str,
        reduce_op_name: str,
        rewrite: MapToMapResolveReduceInstantiateSchema,
    ) -> List[Dict]:
        """
        Apply the directive to the pipeline config.
        """
        new_ops_list = deepcopy(ops_list)

        # Find position of the map and reduce ops
        map_pos = None
        reduce_pos = None
        map_op = None

        for i, op in enumerate(ops_list):
            if op["name"] == map_op_name:
                map_pos = i
                map_op = op
            elif op["name"] == reduce_op_name:
                reduce_pos = i

        if map_pos is None or reduce_pos is None:
            raise ValueError(
                f"Could not find map '{map_op_name}' and reduce '{reduce_op_name}' operations"
            )

        # Determine the model to use
        default_model = map_op.get("model", global_default_model)

        # Find the reduce operation to get the reduce_key
        reduce_op = None
        for op in ops_list:
            if op["name"] == reduce_op_name:
                reduce_op = op
                break

        # Derive output schema from reduce_key - that's what's being grouped/resolved
        reduce_key = reduce_op.get("reduce_key", []) if reduce_op else []
        if isinstance(reduce_key, str):
            reduce_key = [reduce_key]

        # Build output schema from reduce_key fields
        output_schema = {key: "string" for key in reduce_key}

        # Create the resolve operation
        resolve_op = {
            "name": rewrite.resolve_name,
            "type": "resolve",
            "comparison_prompt": rewrite.comparison_prompt,
            "resolution_prompt": rewrite.resolution_prompt,
            "blocking_conditions": rewrite.blocking_conditions,
            "blocking_keys": rewrite.blocking_keys,
            "limit_comparisons": rewrite.limit_comparisons,
            "model": default_model,
            "output": {"schema": output_schema},
        }

        # Insert resolve operation after the map operation
        new_ops_list.insert(map_pos + 1, resolve_op)

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
        # Assert that there are exactly two target ops (map and reduce)
        assert (
            len(target_ops) == 2
        ), "There must be exactly two target ops (map and reduce) to instantiate this directive"

        # Find the map and reduce operations
        map_op = None
        reduce_op = None

        for op in operators:
            if op["name"] == target_ops[0]:
                if op.get("type") == "map":
                    map_op = op
                elif op.get("type") == "reduce":
                    reduce_op = op
            elif op["name"] == target_ops[1]:
                if op.get("type") == "map":
                    map_op = op
                elif op.get("type") == "reduce":
                    reduce_op = op

        if map_op is None or reduce_op is None:
            raise ValueError(
                f"Could not find both a map and reduce operation in target_ops: {target_ops}"
            )

        # Verify the map comes before reduce
        map_idx = next(
            i for i, op in enumerate(operators) if op["name"] == map_op["name"]
        )
        reduce_idx = next(
            i for i, op in enumerate(operators) if op["name"] == reduce_op["name"]
        )

        if map_idx >= reduce_idx:
            raise ValueError(
                f"Map operation '{map_op['name']}' must come before reduce operation '{reduce_op['name']}'"
            )

        # Instantiate the directive
        rewrite, message_history, call_cost = self.llm_instantiate(
            map_op,
            reduce_op,
            agent_llm,
            message_history,
        )

        # Apply the rewrite to the operators
        new_ops_plan = self.apply(
            global_default_model, operators, map_op["name"], reduce_op["name"], rewrite
        )
        return new_ops_plan, message_history, call_cost
