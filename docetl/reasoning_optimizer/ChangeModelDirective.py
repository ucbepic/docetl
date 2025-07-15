from copy import deepcopy
from types import NoneType
from pydantic import BaseModel, Field
from typing import Type, Dict, List
import os
from litellm import completion
from docetl.reasoning_optimizer.directive import Directive
from docetl.reasoning_optimizer.instantiate_schemas import ChangeModelConfig, ChangeModelInstantiateSchema
import re
import json

MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS = 3

class ChangeModelDirective(Directive):
    name: str = Field(default="change model", description="The name of the directive")
    formal_description: str = Field(default="Op => Op* (same operation with a different model choice)")
    nl_description: str = Field(default="Rewrites an operator to use a different LLM model, changing the underlying engine while keeping all logic and prompts the same..")
    when_to_use: str = Field(default="When a specific step in the pipeline would benefit from a different model (e.g., for cost, speed, or accuracy reasons), but all other config stays the same.")
    instantiate_schema_type: Type[BaseModel] = ChangeModelConfig
    
    example: str = Field(
        default=(
            "Original Op (MapOpConfig):\n"
            "- name: extract_insights\n"
            "  type: map\n"
            "  prompt: |\n"
            "    From the user log below, list 2-3 concise insights (1-2 words each) and 1-2 supporting actions per insight.\n"
            "    Return as a list of dictionaries with 'insight' and 'supporting_actions'.\n"
            "    Log: {{ input.log }}\n"
            "  output:\n"
            "    schema:\n"
            "      insights_summary: \"string\"\n"
            "  model: gpt-4o\n"
            "\n"
            "Example InstantiateSchema:\n"
            "[\n"
            "  ChangeModelConfig(\n"
            "    model='gpt-4o-mini'\n"
            "  ),\n"
            "]"
        ),
    )

    allowed_model_list: List[str] = Field(default=["gpt-4.1-nano", "gpt-4o-mini", "gpt-4o", "gpt-4.1"], description="The allowed list of models to choose from")

    model_info: str = Field(
        default=("""
        OpenAI-MRCR evaluates a model’s ability to locate and disambiguate multiple well-hidden “needles” within a large context.
            Below are the actual performance scores for the 8-needle retrieval task at various context lengths. Use these results to compare the retrieval capabilities of each model.

            Input Tokens (1000s) | GPT-4.1 | GPT-4.1 nano | GPT-4o (2024-11-20) | GPT-4o mini
            ---------------------|---------|--------------|---------------------|-------------
            8                    | 32.5%   | 21.0%        | 17.5%               | 20.5%
            16                   | 26.0%   | 18.5%        | 17.5%               | 15.5%
            32                   | 16.5%   | 13.5%        | 17.0%               | 12.0%
            64                   | 21.5%   | 16.0%        | 19.5%               | 13.5%
            128                  | 18.5%   | 14.5%        | 18.0%               | 11.0%
            256                  | 17.0%   | 14.5%        | -                   | -
            512                  | 16.5%   | 9.0%         | -                   | -
            1024                 | 16.0%   | 5.0%         | -                   | -

            The context window and pricing details for each model are shown below (token prices are per 1 million tokens):
            Model              | GPT-4.1-nano | GPT-4o-mini | GPT-4o     | GPT-4.1
            -------------------|--------------|-------------|------------|---------
            Context Window     | 1,047,576    | 128,000     | 128,000    | 1,047,576
            Max Output Tokens  | 32,768       | 16,384      | 16,384     | 32,768
            Input Token Price  | $0.10        | $0.15       | $2.50      | $2.00
            Output Token Price | $0.40        | $0.60       | $1.25      | $8.00
        """),
    )
    
    def __eq__(self, other):
        return isinstance(other, ChangeModelDirective)  

    def __hash__(self):
        return hash('ChangeModelDirective')  

    def to_string_for_instantiate(self, original_op: Dict) -> str:
        """
        Generate a prompt for an agent to instantiate this directive.

        Args:
            original_op (str): The YAML or string representation of the original operation.

        Returns:
            str: The agent prompt for instantiating the directive.
        """
        return (
            f"You are an expert at choosing the most suitable model for a given task.\n\n"
            f"Original Operation:\n"
            f"{str(original_op)}\n\n"
            f"Directive: {self.name}\n"
            f"Your task is to instantiate this directive by generating a ChangeModelConfig that suggests a better model for executing the original operation."
            f"You have a list of allowed models to choose from: {str(self.allowed_model_list)}.\n\n"
            f"Consider the information about the allowed models: \n {self.model_info}\n"
            f"The ChangeModelConfig should include the new model choice for the operation."
            f"Ensure that your chosen model is in the list of allowed models."
            f"Example:\n"
            f"{self.example}\n\n"
            f"Please output only the InstantiateSchema (a list of ChangeModelConfig object)."
        )

    def llm_instantiate(
        self,
        original_op: Dict,
        agent_llm: str,
        message_history: list = [],
    ) -> tuple:
        """
        Use LLM to instantiate this directive.

        Args:
            original_op (Dict): The original operation.
            agent_llm (str): The LLM model to use.
            message_history (List, optional): Conversation history for context.

        Returns:
            ChangeModelInstantiateSchema: The structured output from the LLM.
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
                # api_key=os.environ["GEMINI_API_KEY"],
                azure=True,
                response_format=ChangeModelInstantiateSchema
            )
            try:
                parsed_res = json.loads(resp.choices[0].message.content)
                if "change_model_config" not in parsed_res:
                    raise ValueError("Response from LLM is missing required key 'change_model_config'")
                change_model_config = parsed_res["change_model_config"]
                schema = ChangeModelInstantiateSchema(change_model_config = change_model_config)
                
                # Validate the model is in the allowed model list
                ChangeModelInstantiateSchema.validate_model_in_list(
                    change_model_config=schema.change_model_config,
                    list_of_model=self.allowed_model_list
                )
                message_history.append({"role": "assistant", "content": resp.choices[0].message.content})
                return schema, message_history
            except Exception as err:
                error_message = f"Validation error: {err}\nPlease try again."
                message_history.append({"role": "user", "content": error_message})
        
        raise Exception(f"Failed to instantiate directive after {MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS} attempts.")
    
    def apply(self, ops_list: List[Dict], target_op: str, rewrite: ChangeModelInstantiateSchema) -> List[Dict]:
        """
        Apply the directive to the pipeline config by adding gleaning configuration to the target operator.
        """
        # Create a copy of the pipeline config
        new_ops_list = deepcopy(ops_list)
        
        # Find position of the target op to modify
        pos_to_replace = [i for i, op in enumerate(ops_list) if op["name"] == target_op][0]
        
        # Add gleaning configuration to the target operator
        target_operator = new_ops_list[pos_to_replace]
        target_operator["model"] = rewrite.change_model_config.model
        
        return new_ops_list
    
    def instantiate(self, operators: List[Dict], target_ops: List[str], agent_llm: str, message_history: list = []) -> tuple:
        """
        Instantiate the directive for a list of operators.
        """
        # Assert that there is only one target op
        assert len(target_ops) == 1, "There must be exactly one target op to instantiate this chaining directive"
        target_op_config = [op for op in operators if op["name"] == target_ops[0]][0]

        # Instantiate the directive
        rewrite, message_history = self.llm_instantiate(target_op_config, agent_llm, message_history)
        
        # Apply the rewrite to the operators
        return self.apply(operators, target_ops[0], rewrite), message_history