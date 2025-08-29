import json
import os
import re
from copy import deepcopy
from typing import Dict, List, Type

from litellm import completion
from pydantic import BaseModel, Field

from docetl.reasoning_optimizer.instantiate_schemas import (
    ReduceChainingInstantiateSchema,
)

from .base import MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS, Directive, DirectiveTestCase


class ReduceChainingDirective(Directive):
    name: str = Field(
        default="reduce_chaining", description="The name of the directive"
    )
    formal_description: str = Field(default="Reduce => Map -> Reduce")
    nl_description: str = Field(
        default="Transform a reduce operation that processes long documents by inserting a Map step that extracts/processes relevant information from each document first, then modifying the reduce prompt to work with the processed results instead of full document content."
    )
    when_to_use: str = Field(
        default="When a reduce operation iterates through long documents to extract specific information (e.g., locations, entities, themes) that could be pre-extracted per document to make the reduce step more efficient and focused. The target operator must be a reduce operator. You should specify a reduce operator as the target operator when choosing this directive."
    )
    instantiate_schema_type: Type[BaseModel] = ReduceChainingInstantiateSchema

    example: str = Field(
        default=(
            "Original Reduce Op:\n"
            "- name: extract_all_locations\n"
            "  type: reduce\n"
            "  reduce_key: document_collection\n"
            "  prompt: |\n"
            "    Extract all distinct locations mentioned across these documents:\n"
            "    {% for input in inputs %}\n"
            "    Document: {{ input.document }}\n"
            "    {% endfor %}\n"
            "    Return a list of unique location names.\n"
            "  output:\n"
            "    schema:\n"
            "      locations: list[str]\n"
            "\n"
            "Example InstantiateSchema:\n"
            "ReduceChainingInstantiateSchema(\n"
            "  map_name='extract_document_locations',\n"
            "  map_prompt='Extract all location names mentioned in this document:\\n{{ input.document }}\\nReturn a list of locations.',\n"
            "  new_key='locations',\n"
            "  modified_reduce_prompt='Combine and deduplicate all locations from these documents:\\n{% for input in inputs %}\\nLocations from document: {{ input.locations }}\\n{% endfor %}\\nReturn a list of unique location names.',\n"
            ")"
        ),
    )

    test_cases: List[DirectiveTestCase] = Field(
        default_factory=lambda: [
            DirectiveTestCase(
                name="extract_distinct_entities",
                description="Should decompose entity extraction into map-reduce pattern",
                input_config={
                    "name": "extract_all_people",
                    "type": "reduce",
                    "reduce_key": "doc_id",
                    "prompt": "Extract all distinct person names mentioned across these documents:\n{% for input in inputs %}\nDocument: {{ input.text }}\n{% endfor %}\nReturn a list of unique person names.",
                    "output": {"schema": {"people": "list[str]"}},
                },
                target_ops=["extract_all_people"],
                expected_behavior="Should create a map op to extract people from each document, then modify reduce to work with extracted lists",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="theme_analysis",
                description="Should decompose theme analysis into map-reduce pattern",
                input_config={
                    "name": "analyze_themes",
                    "type": "reduce",
                    "reduce_key": "category",
                    "prompt": "Identify common themes across these research papers:\n{% for input in inputs %}\nPaper: {{ input.content }}\n{% endfor %}\nReturn the main themes.",
                    "output": {"schema": {"themes": "list[str]"}},
                },
                target_ops=["analyze_themes"],
                expected_behavior="Should create a map op to extract themes from each paper, then reduce to identify common ones",
                should_pass=True,
            ),
        ]
    )

    def __eq__(self, other):
        return isinstance(other, ReduceChainingDirective)

    def __hash__(self):
        return hash("ReduceChainingDirective")

    def to_string_for_instantiate(self, original_op: Dict) -> str:
        """
        Generate a prompt for an agent to instantiate this directive.

        Args:
            original_op (str): The YAML or string representation of the original operation.

        Returns:
            str: The agent prompt for instantiating the directive.
        """
        return (
            f"You are an expert at optimizing data processing operations by decomposing complex reduce operations.\n\n"
            f"Original Operation:\n"
            f"{str(original_op)}\n"
            f"Directive: {self.name}\n"
            f"Your task is to instantiate this directive by creating a Map operation that preprocesses individual documents, "
            f"and then modifying the Reduce operation to work with the preprocessed results instead of raw document content.\n\n"
            f"The goal is to make the reduce operation more efficient by having the map operation extract or process "
            f"the specific information needed from each document, rather than having the reduce operation process the full document content.\n\n"
            f"Key Requirements:\n"
            f"1. Create a Map operation that processes individual documents and extracts the relevant information\n"
            f"2. Choose an appropriate new key name for the Map operation's output\n"
            f"3. Modify the original reduce prompt to work with the processed results instead of the original document content\n"
            f"4. Ensure the final output schema and semantics remain the same\n"
            f"5. The modified reduce prompt should reference the new key, not the original document key\n\n"
            f"Example:\n"
            f"{self.example}\n\n"
            f"Please output the ReduceChainingInstantiateSchema with the map operation details and modified reduce prompt."
        )

    def llm_instantiate(
        self,
        original_op: Dict,
        expected_document_key: str,
        agent_llm: str,
        message_history: list = []
    ):
        """
        Use LLM to instantiate this directive by decomposing the reduce operation.

        Args:
            original_op (Dict): The original reduce operation.
            expected_document_key (str): The key that contains the document content to be processed.
            agent_llm (str): The LLM model to use.
            message_history (List, optional): Conversation history for context.

        Returns:
            ReduceChainingInstantiateSchema: The structured output from the LLM.
        """

        message_history.extend(
            [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant for document processing pipelines.",
                },
                {
                    "role": "user",
                    "content": self.to_string_for_instantiate(original_op),
                },
            ]
        )

        for _ in range(MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS):

            call_cost = 0
            resp = completion(
                model=agent_llm,
                messages=message_history,
                api_key=os.environ.get("AZURE_API_KEY"),
                api_base=os.environ.get("AZURE_API_BASE"),
                api_version=os.environ.get("AZURE_API_VERSION"),
                azure=True,
                response_format=ReduceChainingInstantiateSchema
            )
            call_cost = resp._hidden_params["response_cost"]
            try:
                parsed_res = json.loads(resp.choices[0].message.content)
                schema = ReduceChainingInstantiateSchema(**parsed_res)

                # Validate the schema
                ReduceChainingInstantiateSchema.validate_reduce_prompt_references_new_key(
                    schema.modified_reduce_prompt, schema.new_key, expected_document_key
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
        rewrite: ReduceChainingInstantiateSchema,
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

        # Create the new map operation
        new_map_op = {
            "name": rewrite.map_name,
            "type": "map",
            "prompt": rewrite.map_prompt,
            "model": default_model,
            "litellm_completion_kwargs": {"temperature": 0},
            "output": {"schema": {rewrite.new_key: "string"}},
        }

        # Modify the reduce operation
        modified_reduce_op = deepcopy(orig_op)
        modified_reduce_op["prompt"] = rewrite.modified_reduce_prompt

        # Insert the map operation before the reduce operation
        new_ops_list.insert(pos_to_modify, new_map_op)
        # Update the reduce operation (now at pos_to_modify + 1)
        new_ops_list[pos_to_modify + 1] = modified_reduce_op

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
        ), "There must be exactly one target op to instantiate this reduce chaining directive"
        target_op_config = [op for op in operators if op["name"] == target_ops[0]][0]

        # Ensure it's a reduce operation
        if target_op_config.get("type") != "reduce":
            raise ValueError(
                f"Target operation '{target_ops[0]}' must be a reduce operation"
            )

        # Extract expected document key from the reduce prompt template
        prompt_template = target_op_config["prompt"]
        # Find all occurrences of {{ input.key }} in the prompt
        input_key_pattern = r"\{\{\s*([^\}\s]+)\s*\}\}"
        input_keys = list(set(re.findall(input_key_pattern, prompt_template)))
        print("input_keys: ", input_keys)
        # Heuristic: pick the key that's most likely to contain document content
        # Look for common document field names
        document_key_candidates = [
            key
            for key in input_keys
            if any(
                doc_word in key.lower()
                for doc_word in ["document", "text", "content", "body", "description"]
            )
        ]

        if document_key_candidates:
            expected_document_key = document_key_candidates[0]  # Pick the first candidate
        elif input_keys:
            expected_document_key = input_keys[0]  # Fall back to the first input key
        else:
            raise ValueError("No input keys found in the reduce operation prompt")

        print(f"Detected document key: {expected_document_key}")

        # Instantiate the directive
        rewrite, message_history, call_cost = self.llm_instantiate(
            target_op_config,
            expected_document_key,
            agent_llm,
            message_history,
        )

        # Apply the rewrite to the operators
        new_ops_plan = self.apply(
            global_default_model, operators, target_ops[0], rewrite
        )
        return new_ops_plan, message_history, call_cost
