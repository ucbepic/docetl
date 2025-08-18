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


class ChangeModelCostDirective(Directive):
    name: str = Field(default="change model cost", description="The name of the directive")
    formal_description: str = Field(
        default="Op => Op* (same operation with a more cost-effective model choice)"
    )
    nl_description: str = Field(
        default="Rewrites an operator to use a more cost-effective LLM model to optimize expenses. Prioritizes cost savings while maintaining adequate performance, typically suggesting cheaper models like gpt-4o-mini or gpt-5-nano for simpler tasks."
    )
    when_to_use: str = Field(
        default="When cost optimization is the primary concern and the task can be performed adequately by a less expensive model. Suitable for simple extraction, basic classification, or high-volume processing where budget constraints are important."
    )
    instantiate_schema_type: Type[BaseModel] = ChangeModelInstantiateSchema

    example: str = Field(
        default=(
            "Original Op (MapOpConfig):\n"
            "- name: extract_names\n"
            "  type: map\n"
            "  prompt: |\n"
            "    Extract person names from the following text.\n"
            "    Text: {{ input.text }}\n"
            "  output:\n"
            "    schema:\n"
            '      names: "list[str]"\n'
            "  model: gpt-5\n"
            "\n"
            "Example InstantiateSchema:\n"
            "{\n"
            '  "model": "gpt-4o-mini"\n'
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
                name="cost_optimization_simple_extraction",
                description="Should suggest cheaper model for simple extraction task",
                input_config={
                    "name": "extract_names",
                    "type": "map",
                    "prompt": "Extract person names from: {{ input.text }}",
                    "output": {"schema": {"names": "list[str]"}},
                    "model": "gpt-5",
                },
                target_ops=["extract_names"],
                expected_behavior="Should recommend gpt-4o-mini or gpt-5-nano for cost savings on simple task",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="high_volume_processing_cost_optimization",
                description="Should optimize model for high-volume document processing to reduce costs",
                input_config={
                    "name": "batch_document_summary",
                    "type": "map",
                    "prompt": "Summarize this document in 2-3 sentences: {{ input.document }}",
                    "output": {"schema": {"summary": "str"}},
                    "model": "gpt-5",
                },
                target_ops=["batch_document_summary"],
                expected_behavior="Should recommend cheaper model like gpt-4o-mini or gpt-5-nano for high-volume simple summarization",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="basic_classification_cost_savings",
                description="Should recommend cost-effective model for basic classification",
                input_config={
                    "name": "classify_sentiment",
                    "type": "map",
                    "prompt": "Classify the sentiment of this text as positive, negative, or neutral: {{ input.text }}",
                    "output": {"schema": {"sentiment": "string"}},
                    "model": "gpt-5",
                },
                target_ops=["classify_sentiment"],
                expected_behavior="Should recommend cheaper model for basic sentiment classification to reduce costs",
                should_pass=True,
            ),
        ]
    )

    def __eq__(self, other):
        return isinstance(other, ChangeModelCostDirective)

    def __hash__(self):
        return hash("ChangeModelCostDirective")

    def to_string_for_instantiate(self, original_op: Dict) -> str:
        """
        Generate a prompt for an agent to instantiate this directive for cost optimization.

        Args:
            original_op (str): The YAML or string representation of the original operation.

        Returns:
            str: The agent prompt for instantiating the directive.
        """
        return (
            f"You are an expert at choosing the most cost-effective model for a given task while maintaining adequate performance.\n\n"
            f"Original Operation:\n"
            f"{str(original_op)}\n\n"
            f"Directive: {self.name}\n"
            f"Your task is to instantiate this directive by suggesting a more cost-effective model for executing the original operation.\n\n"
            f"COST OPTIMIZATION CONSIDERATIONS:\n"
            f"• Prioritize cost savings while maintaining adequate performance for the task\n"
            f"• For simple tasks (extraction, basic classification, straightforward summarization), strongly prefer cheaper models like gpt-4o-mini or gpt-5-nano\n"
            f"• For high-volume processing where cost accumulates quickly, favor the most economical option\n"
            f"• Only use expensive models like gpt-5 if the task absolutely requires the additional capability\n"
            f"• Consider the trade-off between model performance and cost efficiency\n"
            f"• Remember that cheaper models can often handle simpler tasks adequately\n\n"
            f"You have a list of allowed models to choose from: {str(self.allowed_model_list)}.\n\n"
            f"Consider the information about the allowed models: \n {self.model_info}\n"
            f"Your response should include the new model choice for the operation that reduces costs while maintaining adequate performance."
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
        Use LLM to instantiate this directive for cost optimization.

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
                    "content": "You are a helpful AI assistant for document processing pipelines focused on cost optimization.",
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
            print("CHANGE MODEL COST ERROR")
            return None, message_history
        return new_ops_list, message_history