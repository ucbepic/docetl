import json
import os
from copy import deepcopy
from typing import Dict, List, Type

from litellm import completion
from pydantic import BaseModel, Field

from docetl.reasoning_optimizer.instantiate_schemas import ChangeModelInstantiateSchema

from .base import (
    AVAILABLE_MODELS,
    MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS,
    Directive,
    DirectiveTestCase,
)


class ChangeModelAccDirective(Directive):
    name: str = Field(default="change model acc", description="The name of the directive")
    formal_description: str = Field(
        default="Op => Op* (same operation with a more accurate model choice)"
    )
    nl_description: str = Field(
        default="Rewrites an operator to use a more powerful LLM model to optimize accuracy. Prioritizes model performance and quality over cost considerations, typically suggesting more capable models like gpt-5 for complex reasoning tasks."
    )
    when_to_use: str = Field(
        default="When accuracy and quality are the primary concerns, and cost is secondary. Suitable for complex reasoning tasks, critical analysis, or when maximum model performance is needed. Usually changing to a more expensive model will improve accuracy, so you should try this directive if it has not been used in the past iterations."
    )
    instantiate_schema_type: Type[BaseModel] = ChangeModelInstantiateSchema

    example: str = Field(
        default=(
            "Original Op (MapOpConfig):\n"
            "- name: analyze_complex_data\n"
            "  type: map\n"
            "  prompt: |\n"
            "    Analyze this complex financial data and provide detailed insights with risk assessment.\n"
            "    Data: {{ input.financial_data }}\n"
            "  output:\n"
            "    schema:\n"
            '      analysis: "string"\n'
            "  model: gpt-4o-mini\n"
            "\n"
            "Example InstantiateSchema:\n"
            "{\n"
            '  "model": "gpt-5"\n'
            "}"
        ),
    )

    allowed_model_list: List[str] = Field(
        default=AVAILABLE_MODELS,
        description="The allowed list of models to choose from",
    )

    model_info: str = Field(
        default=(
            """
            OpenAI-MRCR evaluates a model's ability to locate and disambiguate multiple well-hidden "needles" within a large context.
            Below are the actual performance scores for the 8-needle retrieval task at various context lengths. Use these results to compare the retrieval capabilities of each model.
            The below results are the mean match ratio.

            Input Tokens (1000s) | GPT-5   | GPT-5 nano   | GPT-4o mini
            ---------------------|---------|--------------|-------------------|
            8                    | 99%     | 69%          | 32%               | 
            16                   | 100%    | 64%          | 30%               | 
            32                   | 96%     | 55%          | 27%               |
            64                   | 98%     | 45%          | 25%               | 
            128                  | 97%     | 44%          | 25%               | 
            256                  | 92%     | 40%          | -                 | 
              

            The context window and pricing details for each model are shown below (token prices are per 1 million tokens):
            Model              | GPT-5-nano   | GPT-4o-mini | GPT-5     
            -------------------|--------------|-------------|----------
            Context Window     | 400,000      | 128,000     | 400,000    
            Max Output Tokens  | 128,000      | 16,384      | 128,000   
            Input Token Price  | $0.05        | $0.15       | $1.25    
            Output Token Price | $0.40        | $0.60       | $10    
        """
        ),
    )

    test_cases: List[DirectiveTestCase] = Field(
        default_factory=lambda: [
            DirectiveTestCase(
                name="complex_reasoning_needs_powerful_model",
                description="Should recommend most powerful model for complex reasoning task",
                input_config={
                    "name": "analyze_legal_implications",
                    "type": "map",
                    "prompt": "Analyze the legal implications and potential risks in this complex contract: {{ input.contract }}",
                    "output": {"schema": {"analysis": "string"}},
                    "model": "gpt-4o-mini",
                },
                target_ops=["analyze_legal_implications"],
                expected_behavior="Should recommend gpt-5 for complex legal analysis requiring strong reasoning",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="accuracy_over_cost_scientific_analysis",
                description="Should prioritize accuracy for scientific analysis task",
                input_config={
                    "name": "analyze_research_data",
                    "type": "map",
                    "prompt": "Perform detailed statistical analysis and provide research insights: {{ input.data }}",
                    "output": {"schema": {"insights": "string"}},
                    "model": "gpt-5-nano",
                },
                target_ops=["analyze_research_data"],
                expected_behavior="Should recommend gpt-5 for accurate scientific analysis despite higher cost",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="critical_medical_analysis_high_accuracy",
                description="Should recommend most accurate model for critical medical analysis",
                input_config={
                    "name": "medical_diagnosis_support",
                    "type": "map",
                    "prompt": "Analyze medical symptoms and provide diagnostic insights: {{ input.symptoms }}",
                    "output": {"schema": {"diagnosis": "string"}},
                    "model": "gpt-4o-mini",
                },
                target_ops=["medical_diagnosis_support"],
                expected_behavior="Should recommend gpt-5 for critical medical analysis requiring highest accuracy",
                should_pass=True,
            ),
        ]
    )

    def __eq__(self, other):
        return isinstance(other, ChangeModelAccDirective)

    def __hash__(self):
        return hash("ChangeModelAccDirective")

    def to_string_for_instantiate(self, original_op: Dict) -> str:
        """
        Generate a prompt for an agent to instantiate this directive for accuracy optimization.

        Args:
            original_op (str): The YAML or string representation of the original operation.

        Returns:
            str: The agent prompt for instantiating the directive.
        """
        return (
            f"You are an expert at choosing the most accurate and powerful model for a given task, prioritizing quality over cost.\n\n"
            f"Original Operation:\n"
            f"{str(original_op)}\n\n"
            f"Directive: {self.name}\n"
            f"Your task is to instantiate this directive by suggesting the most accurate model for executing the original operation.\n\n"
            f"ACCURACY OPTIMIZATION CONSIDERATIONS:\n"
            f"• Prioritize model performance and accuracy over cost considerations\n"
            f"• For complex reasoning tasks (analysis, interpretation, multi-step thinking, legal/medical analysis), strongly prefer gpt-5\n"
            f"• For tasks requiring high precision, detailed analysis, or critical decisions, use the most capable model available\n"
            f"• Consider using gpt-5 even for seemingly simple tasks if accuracy is paramount\n"
            f"• Only consider cheaper models if the task is genuinely simple and accuracy requirements are low\n\n"
            f"You have a list of allowed models to choose from: {str(self.allowed_model_list)}.\n\n"
            f"Consider the information about the allowed models: \n {self.model_info}\n"
            f"Your response should include the new model choice for the operation that maximizes accuracy."
            f"Ensure that your chosen model is in the list of allowed models."
            f"Example:\n"
            f"{self.example}\n\n"
            f"Please output only the ChangeModelInstantiateSchema as JSON."
        )

    def llm_instantiate(
        self,
        global_default_model: str,
        original_op: Dict,
        agent_llm: str,
        message_history: list = [],
    ) -> tuple:
        """
        Use LLM to instantiate this directive for accuracy optimization.

        Args:
            original_op (Dict): The original operation.
            agent_llm (str): The LLM model to use.
            message_history (List, optional): Conversation history for context.

        Returns:
            ChangeModelInstantiateSchema: The structured output from the LLM.
        """

        message_history.extend(
            [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant for document processing pipelines focused on accuracy optimization.",
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
                response_format=ChangeModelInstantiateSchema,
            )
            try:
                parsed_res = json.loads(resp.choices[0].message.content)
                schema = ChangeModelInstantiateSchema(**parsed_res)
                orig_model = global_default_model
                if "model" in original_op:
                    orig_model = original_op.get("model")
                # Validate the model is in the allowed model list
                ChangeModelInstantiateSchema.validate_diff_model_in_list(
                    orig_model=orig_model,
                    model=schema.model,
                    list_of_model=self.allowed_model_list,
                )
                message_history.append(
                    {"role": "assistant", "content": resp.choices[0].message.content}
                )
                return schema, message_history
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
        target_op: str,
        rewrite: ChangeModelInstantiateSchema,
    ) -> List[Dict]:
        """
        Apply the directive to the pipeline config by changing the model of the target operator.
        """
        # Create a copy of the pipeline config
        new_ops_list = deepcopy(ops_list)

        # Find position of the target op to modify
        pos_to_replace = [
            i for i, op in enumerate(ops_list) if op["name"] == target_op
        ][0]

        # Add change model configuration to the target operator
        target_operator = new_ops_list[pos_to_replace]
        target_operator["model"] = rewrite.model

        return new_ops_list

    def instantiate(
        self,
        operators: List[Dict],
        target_ops: List[str],
        agent_llm: str,
        message_history: list = [],
        global_default_model: str = None,
        **kwargs
    ) -> tuple:
        """
        Instantiate the directive for a list of operators.
        """
        new_ops_list = deepcopy(operators)
        inst_error = 0
        for target_op in target_ops:
            target_op_config = [op for op in operators if op["name"] == target_op][0]
            # Instantiate the directive
            try:
                rewrite, message_history = self.llm_instantiate(
                    global_default_model,
                    target_op_config,
                    agent_llm,
                    message_history,
                )
                print(rewrite)
            except Exception as e:
                inst_error += 1
            new_ops_list = self.apply(
                global_default_model, new_ops_list, target_op, rewrite
            )

        if inst_error == len(target_ops):
            print("CHANGE MODEL ACC ERROR")
            return None, message_history
        return new_ops_list, message_history