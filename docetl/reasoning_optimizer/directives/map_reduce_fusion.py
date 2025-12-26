import json
import os
import re
from copy import deepcopy
from typing import Dict, List, Type

from litellm import completion
from pydantic import BaseModel, Field

from docetl.reasoning_optimizer.instantiate_schemas import MapReduceFusionInstantiateSchema

from .base import MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS, Directive, DirectiveTestCase


class MapReduceFusionDirective(Directive):
    name: str = Field(default="map_reduce_fusion", description="The name of the directive")
    formal_description: str = Field(default="Map -> Reduce => Map (with new prompt) -> Reduce (with new propmt)")
    nl_description: str = Field(
        default="Transform a Map operation followed by a Reduce operation over long documents by updating the Map prompt to first extract or process only the relevant information from each document. Then, modify the Reduce prompt to operate on these processed outputs instead of the full document content."
    )
    when_to_use: str = Field(
        default="There is a Map operation followed by a Reduce operation, and the Reduce step iterates over long documents to extract specific information (e.g., locations, entities, themes) that could be pre-extracted per document in the Map step. This makes the Reduce operation more efficient and focused. The target operators must be a Map operator followed by a Reduce operator. When selecting this directive, specify the Map and Reduce operators as the target operators."
    )
    instantiate_schema_type: Type[BaseModel] = MapReduceFusionInstantiateSchema

    example: str = Field(
        default=(
            "Original Pipeline:\\n"
            "Map Op (classify_document):\\n"
            "- name: classify_document\\n"
            "  type: map\\n"
            "  prompt: |\\n"
            "    Classify the following document into a category:\\n"
            "    {{ input.content }}\\n"
            "    Choose from: news, research, legal, business\\n"
            "  output:\\n"
            "    schema:\\n"
            "      category: str\\n"
            "\\n"
            "Reduce Op (extract_organizations):\\n"
            "- name: extract_organizations\\n"
            "  type: reduce\\n"
            "  reduce_key: category\\n"
            "  prompt: |\\n"
            "    For each category \\\"{{ reduce_key }}\\\", extract all organization names from these documents:\\n"
            "    {% for input in inputs %}\\n"
            "    Document {{ loop.index }}: {{ input.content }}\\n"
            "    {% endfor %}\\n"
            "    Return a list of unique organization names.\\n"
            "  output:\\n"
            "    schema:\\n"
            "      organizations: list[str]\\n"
            "\\n"
            "Example InstantiateSchema Output:\\n"
            "MapReduceFusionInstantiateSchema(\\n"
            "  new_map_name='fused_classify_extract_organizations',\\n"
            "  new_map_prompt='Analyze the following document:\\n{{ input.content }}\\n\\n1. Classify the document into a category (news, research, legal, business)\\n\\n2. Extract all organization names mentioned in the document\\n\\nProvide both the category and list of organizations.',\\n"
            "  new_key='organizations',\\n"
            "  new_reduce_prompt='For each category \\\"{{ reduce_key }}\\\", combine all organization names from these pre-extracted lists:\\n{% for input in inputs %}\\nOrganizations from document {{ loop.index }}: {{ input.organizations }}\\n{% endfor %}\\nReturn a single deduplicated list of all unique organization names.'\\n"
            ")"
        ),
    )

    test_cases: List[DirectiveTestCase] = Field(
        default_factory=lambda: [
            DirectiveTestCase(
                name="classification_organization_fusion",
                description="Should fuse document classification with organization extraction",
                input_config=[
                    {
                        "name": "classify_document",
                        "type": "map",
                        "prompt": "Classify this document: {{ input.content }}",
                        "output": {"schema": {"category": "str"}},
                    },
                    {
                        "name": "extract_organizations",
                        "type": "reduce",
                        "reduce_key": "category",
                        "prompt": "For each category '{{ reduce_key }}', extract organizations from: {% for input in inputs %}{{ input.content }}{% endfor %}",
                        "output": {"schema": {"organizations": "list[str]"}},
                    }
                ],
                target_ops=["classify_document", "extract_organizations"],
                expected_behavior="Should modify map to classify AND extract organizations per document, then reduce to aggregate pre-extracted organizations",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="analysis_entity_fusion",
                description="Should fuse document analysis with entity extraction per category",
                input_config=[
                    {
                        "name": "analyze_document",
                        "type": "map",
                        "prompt": "Analyze document type: {{ input.text }}",
                        "output": {"schema": {"doc_type": "str"}},
                    },
                    {
                        "name": "find_people",
                        "type": "reduce",
                        "reduce_key": "doc_type",
                        "prompt": "For each document type '{{ reduce_key }}', find all people mentioned: {% for input in inputs %}Document: {{ input.text }}{% endfor %}",
                        "output": {"schema": {"people": "list[str]"}},
                    }
                ],
                target_ops=["analyze_document", "find_people"],
                expected_behavior="Should modify map to analyze type AND extract people per document, then reduce to combine pre-extracted people lists",
                should_pass=True,
            ),
        ]
    )

    def __eq__(self, other):
        return isinstance(other, MapReduceFusionDirective)

    def __hash__(self):
        return hash("MapReduceFusionDirective")

    def to_string_for_instantiate(self, original_ops: List[Dict]) -> str:
        """
        Generate a prompt for an agent to instantiate this directive.

        Args:
            original_ops (List[Dict]): List containing the map and reduce operations.

        Returns:
            str: The agent prompt for instantiating the directive.
        """
        map_op, reduce_op = original_ops[0], original_ops[1]
        return (
            f"You are an expert at optimizing data processing pipelines for efficiency.\\n\\n"
            f"Original Map Operation:\\n"
            f"{str(map_op)}\\n\\n"
            f"Original Reduce Operation:\\n"
            f"{str(reduce_op)}\\n\\n"
            f"Directive: {self.name}\\n"
            f"Your task is to apply map-reduce fusion: modify the Map operation to pre-extract the information that the Reduce operation needs, "
            f"then update the Reduce operation to work with these pre-extracted results instead of processing full documents.\\n\\n"
            f"Key Requirements:\\n"
            f"1. Analyze what specific information the Reduce operation extracts from documents\\n"
            f"2. Create a new Map prompt that does BOTH the original Map task AND extracts the information needed by Reduce\\n"
            f"3. Create a new key name for the pre-extracted information that the Reduce will reference\\n"
            f"4. Create a new Reduce prompt that aggregates the pre-extracted information instead of processing full documents\\n"
            f"5. The Reduce operation should reference the new key (e.g., {{ input.new_key }}) instead of full document content\\n\\n"
            f"Output Format:\\n"
            f"Return a MapReduceFusionInstantiateSchema with:\\n"
            f"- new_map_name: Combined name for the fused map operation\\n"
            f"- new_map_prompt: Prompt that does both original map task + extraction\\n"
            f"- new_key: Key name for the extracted information\\n"
            f"- new_reduce_prompt: Prompt that works with pre-extracted data\\n\\n"
            f"Example:\\n"
            f"{self.example}\\n\\n"
            f"Please output only the MapReduceFusionInstantiateSchema with the four required fields."
        )

    def llm_instantiate(
        self,
        map_op: Dict,
        reduce_op: Dict,
        expected_document_key,
        agent_llm: str,
        message_history: list = []
    ):
        """
        Use LLM to instantiate this directive by transforming the map and reduce operations.

        Args:
            map_op (Dict): The original map operation.
            reduce_op (Dict): The original reduce operation.
            agent_llm (str): The LLM model to use.
            message_history (List, optional): Conversation history for context.

        Returns:
            MapReduceFusionInstantiateSchema: The structured output from the LLM.
        """
        
        message_history.extend([
            {"role": "system", "content": "You are a helpful AI assistant for optimizing document processing pipelines."},
            {"role": "user", "content": self.to_string_for_instantiate([map_op, reduce_op])},
        ])

        for _ in range(MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS):

            resp = completion(
                model=agent_llm,
                messages=message_history,
                api_key=os.environ.get("AZURE_API_KEY"),
                api_base=os.environ.get("AZURE_API_BASE"),
                api_version=os.environ.get("AZURE_API_VERSION"),
                azure=True,
                response_format=MapReduceFusionInstantiateSchema
            )
            call_cost = resp._hidden_params["response_cost"]
            try:
                parsed_res = json.loads(resp.choices[0].message.content)
                if "new_map_name" not in parsed_res or "new_map_prompt" not in parsed_res or "new_key" not in parsed_res or "new_reduce_prompt" not in parsed_res:
                    raise ValueError(
                        "Response from LLM is missing required keys: 'new_map_name', 'new_map_prompt', 'new_key', or 'new_reduce_prompt'"
                    )
                
                schema = MapReduceFusionInstantiateSchema(
                    new_map_name=parsed_res["new_map_name"],
                    new_map_prompt=parsed_res["new_map_prompt"],
                    new_key=parsed_res["new_key"],
                    new_reduce_prompt=parsed_res["new_reduce_prompt"]
                )

                # Validate the schema
                MapReduceFusionInstantiateSchema.validate_reduce_prompt_references_new_key(
                    schema.new_reduce_prompt, schema.new_key, expected_document_key
                )
                
                message_history.append(
                    {"role": "assistant", "content": resp.choices[0].message.content}
                )
                return schema, message_history, call_cost       
            except Exception as err:
                error_message = f"Validation error: {err}\\nPlease try again."
                message_history.append({"role": "user", "content": error_message})
        
        raise Exception(f"Failed to instantiate directive after {MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS} attempts.")
    
    def apply(self, global_default_model, ops_list: List[Dict], map_target: str, reduce_target: str, rewrite: MapReduceFusionInstantiateSchema) -> List[Dict]:
        """
        Apply the directive to the pipeline config.
        """
        # Create a copy of the pipeline config
        new_ops_list = deepcopy(ops_list)
        
        # Find positions of the target ops
        map_pos = None
        reduce_pos = None
        orig_map_op = None
        orig_reduce_op = None
        
        for i, op in enumerate(ops_list):
            if op["name"] == map_target:
                map_pos = i
                orig_map_op = op
            elif op["name"] == reduce_target:
                reduce_pos = i
                orig_reduce_op = op
        
        if map_pos is None or reduce_pos is None:
            raise ValueError(f"Could not find target operations: {map_target}, {reduce_target}")
        
        # Get default model
        default_model = orig_map_op.get("model", global_default_model)
        
        # Create the new map operation with fused functionality
        new_map_op = {
            "name": rewrite.new_map_name,
            "type": "map",
            "prompt": rewrite.new_map_prompt,
            "model": default_model,
            "litellm_completion_kwargs": {"temperature": 0},
            "output": {
                "schema": {
                    **orig_map_op.get("output", {}).get("schema", {}),
                    rewrite.new_key: "list[str]"
                }
            }
        }
        
        # Create the new reduce operation that works with pre-extracted data
        new_reduce_op = {
            "name": orig_reduce_op["name"],
            "type": "reduce",
            "reduce_key": orig_reduce_op.get("reduce_key"),
            "prompt": rewrite.new_reduce_prompt,
            "model": default_model,
            "litellm_completion_kwargs": {"temperature": 0},
            "output": orig_reduce_op.get("output", {})
        }
        
        # Replace the operations
        new_ops_list[map_pos] = new_map_op
        new_ops_list[reduce_pos] = new_reduce_op
        
        return new_ops_list
    
    def instantiate(
        self, 
        operators: List[Dict], 
        target_ops: List[str], 
        agent_llm: str, 
        message_history: list = [], 
        optimize_goal="acc", 
        global_default_model: str = None, 
        **kwargs
    ):
        """
        Instantiate the directive for a list of operators.
        """
        # Assert that there are exactly two target ops (map and reduce)
        if len(target_ops) != 2:
            raise ValueError("map_reduce_fusion directive requires exactly two target operators: a map and a reduce operation")
        
        # Find the map and reduce operations
        first_op = None
        second_op = None
        
        for op in operators:
            if op["name"] == target_ops[0]:
                first_op = op
            elif op["name"] == target_ops[1]:
                second_op = op
        
        if first_op is None or second_op is None:
            raise ValueError(f"Could not find target operations: {target_ops}")
        
        if first_op.get("type") == "map" and second_op.get("type") == "reduce":
            map_op = first_op
            reduce_op = second_op
            map_target = target_ops[0]
            reduce_target = target_ops[1]
        else:
            raise ValueError("Target operators must be one map operation followed by one reduce operation!")
        
         # Extract expected document key from the reduce prompt template
        prompt_template = reduce_op["prompt"]
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
        rewrite, message_history, call_cost = self.llm_instantiate(map_op, reduce_op, expected_document_key, agent_llm, message_history)
        
        # Apply the rewrite to the operators
        new_ops_plan = self.apply(global_default_model, operators, map_target, reduce_target, rewrite)
        return new_ops_plan, message_history, call_cost