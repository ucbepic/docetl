import json
from copy import deepcopy
from typing import Dict, List, Type

from litellm import completion
from pydantic import BaseModel, Field

from docetl.reasoning_optimizer.instantiate_schemas import (
    MapResolveToMapWithCategoriesInstantiateSchema,
)

from .base import MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS, Directive, DirectiveTestCase


class MapResolveToMapWithCategoriesDirective(Directive):
    name: str = Field(
        default="map_resolve_to_map_with_categories",
        description="The name of the directive",
    )
    formal_description: str = Field(default="Map -> Resolve => Map (with categories)")
    nl_description: str = Field(
        default="Replace a Map -> Resolve pattern with a single Map operation that has predefined categories. The agent analyzes the data and task to propose a set of canonical categories, and the new Map forces outputs into one of these categories (or 'none of the above'). This effectively performs entity resolution deterministically by standardizing outputs upfront, avoiding the need for pairwise comparisons."
    )
    when_to_use: str = Field(
        default="When a Map operation produces outputs that are then resolved/deduplicated, and the set of valid output categories can be enumerated upfront. This is more efficient than Resolve when the category space is small and well-defined (e.g., standardizing company types, product categories, sentiment labels). The target must be a Map operation followed by a Resolve operation."
    )
    instantiate_schema_type: Type[BaseModel] = (
        MapResolveToMapWithCategoriesInstantiateSchema
    )

    example: str = Field(
        default=(
            "Original Pipeline:\n"
            "- name: extract_sentiment\n"
            "  type: map\n"
            "  prompt: |\n"
            "    What is the sentiment of this review?\n"
            "    {{ input.review }}\n"
            "  output:\n"
            "    schema:\n"
            "      sentiment: string\n"
            "\n"
            "- name: normalize_sentiment\n"
            "  type: resolve\n"
            "  comparison_prompt: |\n"
            "    Are these sentiments equivalent?\n"
            "    Sentiment 1: {{ input1.sentiment }}\n"
            "    Sentiment 2: {{ input2.sentiment }}\n"
            "  resolution_prompt: |\n"
            "    Normalize these sentiments:\n"
            "    {% for input in inputs %}\n"
            "    - {{ input.sentiment }}\n"
            "    {% endfor %}\n"
            "  output:\n"
            "    schema:\n"
            "      sentiment: string\n"
            "\n"
            "Example InstantiateSchema:\n"
            "MapResolveToMapWithCategoriesInstantiateSchema(\n"
            "  categories=['Positive', 'Negative', 'Neutral', 'Mixed'],\n"
            "  category_key='sentiment',\n"
            "  new_prompt='''Analyze the sentiment of this review and classify it into one of the following categories:\n"
            "\n"
            "Categories:\n"
            "- Positive: Clearly positive sentiment, satisfaction, praise\n"
            "- Negative: Clearly negative sentiment, complaints, criticism\n"
            "- Neutral: No strong sentiment, factual statements\n"
            "- Mixed: Contains both positive and negative elements\n"
            "- None of the above: If the review doesn't fit any category\n"
            "\n"
            "Review: {{ input.review }}\n"
            "\n"
            "Return exactly one of: Positive, Negative, Neutral, Mixed, or None of the above.''',\n"
            "  include_none_of_above=True,\n"
            ")"
        ),
    )

    test_cases: List[DirectiveTestCase] = Field(
        default_factory=lambda: [
            DirectiveTestCase(
                name="sentiment_categorization",
                description="Should replace map+resolve with categorized map for sentiment",
                input_config=[
                    {
                        "name": "extract_sentiment",
                        "type": "map",
                        "prompt": "What is the sentiment? {{ input.text }}",
                        "output": {"schema": {"sentiment": "string"}},
                    },
                    {
                        "name": "normalize_sentiment",
                        "type": "resolve",
                        "comparison_prompt": "Same sentiment? {{ input1.sentiment }} vs {{ input2.sentiment }}",
                        "resolution_prompt": "Normalize: {% for input in inputs %}{{ input.sentiment }}{% endfor %}",
                        "output": {"schema": {"sentiment": "string"}},
                    },
                ],
                target_ops=["extract_sentiment", "normalize_sentiment"],
                expected_behavior="Should create a single map with predefined sentiment categories",
                should_pass=True,
            ),
        ]
    )

    def __eq__(self, other):
        return isinstance(other, MapResolveToMapWithCategoriesDirective)

    def __hash__(self):
        return hash("MapResolveToMapWithCategoriesDirective")

    def to_string_for_instantiate(
        self, map_op: Dict, resolve_op: Dict, sample_data: List[Dict] = None
    ) -> str:
        """
        Generate a prompt for an agent to instantiate this directive.

        Args:
            map_op (Dict): The map operation configuration.
            resolve_op (Dict): The resolve operation configuration.
            sample_data (List[Dict], optional): Sample data to help identify categories.

        Returns:
            str: The agent prompt for instantiating the directive.
        """
        sample_str = ""
        if sample_data:
            sample_str = f"\n\nSample Input Data (first 5 items):\n{json.dumps(sample_data[:5], indent=2)}"

        return (
            f"You are an expert at optimizing data processing pipelines by replacing entity resolution with categorical constraints.\n\n"
            f"Map Operation:\n"
            f"{str(map_op)}\n\n"
            f"Resolve Operation:\n"
            f"{str(resolve_op)}\n\n"
            f"Directive: {self.name}\n"
            f"Your task is to replace the Map -> Resolve pattern with a single Map that uses predefined categories.\n\n"
            f"Key Requirements:\n"
            f"1. Analyze the map's output field and the resolve operation to understand what values are being normalized:\n"
            f"   - Look at the comparison_prompt to understand what variations are being matched\n"
            f"   - Look at the resolution_prompt to understand the canonical form\n\n"
            f"2. Propose a set of categories that cover all expected outputs:\n"
            f"   - Categories should be mutually exclusive\n"
            f"   - Categories should be exhaustive (cover all realistic cases)\n"
            f"   - Consider including 'None of the above' for edge cases\n\n"
            f"3. Optionally provide descriptions for each category to help the LLM classify correctly\n\n"
            f"4. Create a new_prompt that:\n"
            f"   - Lists all valid categories with their descriptions\n"
            f"   - Instructs the LLM to output exactly one category\n"
            f"   - References the input using {{{{ input.key }}}} syntax\n"
            f"   - Includes any context from the original map prompt\n\n"
            f"5. Identify the category_key (the output field that will contain the category)\n\n"
            f"Benefits of this approach:\n"
            f"- Eliminates O(n^2) pairwise comparisons from Resolve\n"
            f"- Produces consistent, standardized outputs\n"
            f"- Reduces cost by removing the Resolve operation entirely\n"
            f"{sample_str}\n\n"
            f"Example:\n"
            f"{self.example}\n\n"
            f"Please analyze the operations and propose appropriate categories. Output the MapResolveToMapWithCategoriesInstantiateSchema."
        )

    def llm_instantiate(
        self,
        map_op: Dict,
        resolve_op: Dict,
        agent_llm: str,
        message_history: list = [],
        sample_data: List[Dict] = None,
    ):
        """
        Use LLM to instantiate this directive.

        Args:
            map_op (Dict): The map operation configuration.
            resolve_op (Dict): The resolve operation configuration.
            agent_llm (str): The LLM model to use.
            message_history (List, optional): Conversation history for context.
            sample_data (List[Dict], optional): Sample data to help identify categories.

        Returns:
            MapResolveToMapWithCategoriesInstantiateSchema: The structured output from the LLM.
        """

        message_history.extend(
            [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant for document processing pipelines specializing in categorical classification.",
                },
                {
                    "role": "user",
                    "content": self.to_string_for_instantiate(
                        map_op, resolve_op, sample_data
                    ),
                },
            ]
        )

        for _ in range(MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS):

            resp = completion(
                model=agent_llm,
                messages=message_history,
                response_format=MapResolveToMapWithCategoriesInstantiateSchema,
            )

            call_cost = resp._hidden_params.get("response_cost", 0)

            try:
                parsed_res = json.loads(resp.choices[0].message.content)
                schema = MapResolveToMapWithCategoriesInstantiateSchema(**parsed_res)

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
        resolve_op_name: str,
        rewrite: MapResolveToMapWithCategoriesInstantiateSchema,
    ) -> List[Dict]:
        """
        Apply the directive to the pipeline config.
        """
        new_ops_list = deepcopy(ops_list)

        # Find position of the map and resolve ops
        map_pos = None
        resolve_pos = None
        map_op = None

        for i, op in enumerate(new_ops_list):
            if op["name"] == map_op_name:
                map_pos = i
                map_op = op
            elif op["name"] == resolve_op_name:
                resolve_pos = i

        if map_pos is None or resolve_pos is None:
            raise ValueError(
                f"Could not find map '{map_op_name}' and resolve '{resolve_op_name}' operations"
            )

        # Determine the model to use
        default_model = map_op.get("model", global_default_model)

        # Build the list of valid values for validation
        valid_values = list(rewrite.categories)
        if rewrite.include_none_of_above:
            valid_values.append("None of the above")

        # Modify the map operation with the new prompt and add validation
        new_ops_list[map_pos]["prompt"] = rewrite.new_prompt
        new_ops_list[map_pos]["model"] = default_model

        # Add validation to ensure output is one of the categories
        if "validate" not in new_ops_list[map_pos]:
            new_ops_list[map_pos]["validate"] = []

        # Add validation rule for the category key
        validation_rule = f"output['{rewrite.category_key}'] in {valid_values}"
        new_ops_list[map_pos]["validate"].append(validation_rule)

        # Update the output schema to reflect the category key
        if "output" not in new_ops_list[map_pos]:
            new_ops_list[map_pos]["output"] = {"schema": {}}
        new_ops_list[map_pos]["output"]["schema"][rewrite.category_key] = "string"

        # Remove the resolve operation
        new_ops_list.pop(resolve_pos)

        return new_ops_list

    def instantiate(
        self,
        operators: List[Dict],
        target_ops: List[str],
        agent_llm: str,
        message_history: list = [],
        optimize_goal="acc",
        global_default_model: str = None,
        dataset: str = None,
        **kwargs,
    ):
        """
        Instantiate the directive for a list of operators.
        """
        # Assert that there are exactly two target ops (map and resolve)
        assert (
            len(target_ops) == 2
        ), "There must be exactly two target ops (map and resolve) to instantiate this directive"

        # Find the map and resolve operations
        map_op = None
        resolve_op = None

        for op in operators:
            if op["name"] == target_ops[0]:
                if op.get("type") == "map":
                    map_op = op
                elif op.get("type") == "resolve":
                    resolve_op = op
            elif op["name"] == target_ops[1]:
                if op.get("type") == "map":
                    map_op = op
                elif op.get("type") == "resolve":
                    resolve_op = op

        if map_op is None or resolve_op is None:
            raise ValueError(
                f"Could not find both a map and resolve operation in target_ops: {target_ops}"
            )

        # Verify the map comes before resolve
        map_idx = next(
            i for i, op in enumerate(operators) if op["name"] == map_op["name"]
        )
        resolve_idx = next(
            i for i, op in enumerate(operators) if op["name"] == resolve_op["name"]
        )

        if map_idx >= resolve_idx:
            raise ValueError(
                f"Map operation '{map_op['name']}' must come before resolve operation '{resolve_op['name']}'"
            )

        # Load sample data if available
        sample_data = None
        if dataset:
            try:
                with open(dataset, "r") as f:
                    sample_data = json.load(f)
            except Exception:
                pass  # Ignore if we can't load sample data

        # Instantiate the directive
        rewrite, message_history, call_cost = self.llm_instantiate(
            map_op,
            resolve_op,
            agent_llm,
            message_history,
            sample_data,
        )

        # Apply the rewrite to the operators
        new_ops_plan = self.apply(
            global_default_model, operators, map_op["name"], resolve_op["name"], rewrite
        )
        return new_ops_plan, message_history, call_cost
