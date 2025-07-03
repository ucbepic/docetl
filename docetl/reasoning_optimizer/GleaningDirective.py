from copy import deepcopy
from types import NoneType
from pydantic import BaseModel, Field
from typing import Type, Dict, List
import os
from litellm import completion
from docetl.reasoning_optimizer.directive import Directive
from instantiate_schemas import MapOpConfig, GleaningInstantiateSchema
import re
import json

MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS = 3


class GleaningDirective(Directive):
    name: str = Field(default="gleaning", description="The name of the directive")
    formal_description: str = Field(default="Map => Map_m (with gleaning config)")
    nl_description: str = Field(default="""Adds a validation loop to Map: after each LLM generation, a separate "judge" LLM evaluates the output using a yes/no validation prompt. If the output fails, the original LLM refines its answer and repeats until the output passes or the max number of rounds is reached.""")
    when_to_use: str = Field(default="When initial Map outputs may not meet quality criteria and must be checked or improved automatically (e.g., too short, missing required info).")
    
    # Remove from Pydantic fields, make it a plain class variable
    instantiate_schema_type: Type[BaseModel] = Field(default=GleaningInstantiateSchema)
    
    example: str = Field(
        default="""
            "Original Op (MapOpConfig):\n"
            "- name: extract_insights\n"
            "  type: map\n"
            "  prompt: |\n"
            "    From the user log below, list 2-3 concise insights (1-2 words each) and 1-2 supporting actions per insight.\n"
            "    Return as a list of dictionaries with 'insight' and 'supporting_actions'.\n"
            "    Log: {{ input.log }}\n"
            "  output:\n"
            "    schema:\n"
            "      insights_summary: "string"\n"
            "\n"
            "Example InstantiateSchema:\n"
            "[\n"
            "  GleaningConfig(\n"
            "    validation_prompt='''There should be at least 2 insights, and each insight should have at least 1 supporting action.''',\n"
            "    num_rounds = 2,\n"
                 model='gpt-4o-mini',\n"
            "  ),\n"
            "]"
        """,
    )
    
    def to_string_for_instantiate(self, original_op: Dict) -> str:
        """
        Generate a prompt for an agent to instantiate this directive.

        Args:
            original_op (str): The YAML or string representation of the original operation.

        Returns:
            str: The agent prompt for instantiating the directive.
        """
        return (
            f"You are an expert at adding validation and refinement loops to data processing operations.\n\n"
            f"Original Operation:\n"
            f"{str(original_op)}\n\n"
            f"Directive: {self.name}\n"
            f"Your task is to instantiate this directive by generating a GleaningConfig that adds validation loops to the original operation. "
            f"The gleaning configuration should include a validation prompt that evaluates the output quality and provides feedback for improvement, "
            f"along with the number of refinement rounds to attempt.\n\n"
            f"Example:\n"
            f"{self.example}\n\n"
            f"Please output only the InstantiateSchema (a GleaningConfig object) that specifies how to validate and refine the output of the original operation."
        )

    def llm_instantiate(
        self,
        original_op: Dict,
        agent_llm: str,
        message_history: list = [],
    ) -> GleaningInstantiateSchema:
        """
        Use LLM to instantiate this directive by decomposing the original operation.

        Args:
            original_op (Dict): The original operation.
            agent_llm (str): The LLM model to use.
            message_history (List, optional): Conversation history for context.

        Returns:
            GleaningInstantiateSchema: The structured output from the LLM.
        """
        
        message_history.extend([
            {"role": "system", "content": "You are a helpful AI assistant for document processing pipelines."},
            {"role": "user", "content": self.to_string_for_instantiate(original_op)},
        ])

        for _ in range(MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS):
            resp = completion(
                model=agent_llm,
                messages=message_history,
                api_key=os.environ.get("AZURE_API_KEY"),
                api_base=os.environ.get("AZURE_API_BASE"),
                api_version=os.environ.get("AZURE_API_VERSION"),
                azure=True,
                response_format=GleaningInstantiateSchema
            )

            try:
                parsed_res = json.loads(resp.choices[0].message.content)
                print("*************** parsed res:")
                print(parsed_res)
                print("***************")
                if "gleaning_config" not in parsed_res:
                    raise ValueError("Response from LLM is missing required key 'gleaning_config'")
                gleaning_config = parsed_res["gleaning_config"]
                schema = GleaningInstantiateSchema(gleaning_config = gleaning_config)
                print("schema:")
                print(schema)
                return schema
            except Exception as err:
                error_message = f"Validation error: {err}\nPlease try again."
                message_history.append({"role": "user", "content": error_message})
        
        raise Exception(f"Failed to instantiate directive after {MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS} attempts.")
    
    def apply(self, ops_list: List[Dict], target_op: str, rewrite: GleaningInstantiateSchema) -> List[Dict]:
        """
        Apply the directive to the pipeline config by adding gleaning configuration to the target operator.
        """
        # Create a copy of the pipeline config
        new_ops_list = deepcopy(ops_list)
        
        # Find position of the target op to modify
        pos_to_replace = [i for i, op in enumerate(ops_list) if op["name"] == target_op][0]
        
        # Add gleaning configuration to the target operator
        target_operator = new_ops_list[pos_to_replace]
        target_operator["gleaning"] = {
            "validation_prompt": rewrite.gleaning_config.validation_prompt,
            "num_rounds": rewrite.gleaning_config.num_rounds,
            "model": rewrite.gleaning_config.model
        }
        
        return new_ops_list
    
    def instantiate(self, operators: List[Dict], target_ops: List[str], agent_llm: str, message_history: list = []) -> List[Dict]:
        """
        Instantiate the directive for a list of operators.
        """
        # Assert that there is only one target op
        assert len(target_ops) == 1, "There must be exactly one target op to instantiate this chaining directive"
        target_op_config = [op for op in operators if op["name"] == target_ops[0]][0]

        # Instantiate the directive
        rewrite = self.llm_instantiate(target_op_config, agent_llm, message_history)
        
        # Apply the rewrite to the operators
        return self.apply(operators, target_ops[0], rewrite)