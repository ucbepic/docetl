import json
import os
from copy import deepcopy
from typing import Dict, List, Type

from litellm import completion
from pydantic import BaseModel, Field

from docetl.reasoning_optimizer.instantiate_schemas import ChangeModelInstantiateSchema

from .base import MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS, Directive, DirectiveTestCase


class ChangeModelAccDirective(Directive):
    name: str = Field(
        default="change model acc", description="The name of the directive"
    )
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
        default_factory=list,
        description="The allowed list of models to choose from",
    )

    model_info: str = Field(
        default=(
            """
            OpenAI-MRCR evaluates a model's ability to locate and disambiguate multiple well-hidden "needles" within a large context.
            Below are the actual performance scores for the 2-needle retrieval task at various context lengths. Use these results to compare the retrieval capabilities of each model.
            The below results are the mean match ratio.

            Context Length | gpt-5 | gpt-5-nano | gpt-4o-mini | gemini-2.5-pro | gemini-2.5-flash | gpt-4.1 | gpt-4.1-mini | gpt-4.1-nano | gemini-2.5-flash-lite
            ---------------|-------|------------|-------------|----------------|------------------|---------|--------------|--------------|----------------------
            128k           | 97%   | 44%        | 25%         | 83.6%          | 86.2%            | 61.3%   | 47.1%        | 36.7%        | 39.9%
            1M             | -     | -          | -           | 62.8%          | 60.0%            | 45.9%   | 34.6%        | 14.2%        | 18.1%

            The context window and pricing details for each model are shown below (token prices are per 1 million tokens):
            | Family         | Model                      | Input Price /1M                 | Output Price /1M                  | Context Window (API)       |
            |----------------|----------------------------|---------------------------------|-----------------------------------|-----------------------------|
            | **GPT-5**      | azure/gpt-5                | $1.25                           | $10.00                            | 400K (272K in + 128K out) |
            |                | azure/gpt-5-mini           | $0.25                           | $2.00                             | 400K                       |
            |                | azure/gpt-5-nano           | $0.05                           | $0.40                             | 400K                       |
            | **GPT-4.1**    | azure/gpt-4.1              | $2.00                           | $8.00                             | 1M                         |
            |                | azure/gpt-4.1-mini         | $0.40                           | $1.60                             | 1M                         |
            |                | azure/gpt-4.1-nano         | $0.10                           | $0.40                             | 1M                         |
            | **GPT-4o**     | azure/gpt-4o               | $2.50                           | $10.00                            | 128K                       |
            |                | azure/gpt-4o-mini          | $0.15                           | $0.60                             | 128K (≈16K output cap)     |
            | **Gemini 2.5** | gemini/gemini-2.5-pro      | $1.25 (≤200K) / $2.50 (>200K)  | $10.00 (≤200K) / $15.00 (>200K)  | 1M (2M soon)               |
            |                | gemini/gemini-2.5-flash    | $0.30                           | $2.50                             | 1M                         |
            |                | gemini/gemini-2.5-flash-lite | $0.10                         | $0.40                             | 1M                         |
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
            f"TASK COMPLEXITY ANALYSIS AND MODEL SELECTION:\n"
            f"First, carefully analyze the original operation to understand:\n"
            f"• What specific task is being performed (extraction, analysis, transformation, reasoning, etc.)\n"
            f"• How much cognitive complexity and intelligence is required\n"
            f"• Whether the task involves simple pattern matching or sophisticated reasoning\n"
            f"• If the task requires domain expertise, multi-step thinking, or nuanced understanding\n"
            f"• The criticality and precision requirements of the output\n"
            f"• If the task requires processing very long context (1M+ tokens)\n\n"
            f"Based on your analysis of task complexity, select the model that will provide the most accurate response:\n"
            f"• For simple extraction or formatting tasks: Consider efficient models from the available options\n"
            f"• For moderate complexity tasks requiring some reasoning: Use capable models from gpt-5 or gemini series\n"
            f"• For complex reasoning, analysis, interpretation, legal/medical tasks, or critical decisions: Strongly prefer the most advanced models from gpt-5 or gemini series\n"
            f"• For tasks requiring very long context (1M+ tokens): Consider models with extended context like gpt-4.1 series or gemini models\n"
            f"• For highly specialized or extremely complex cognitive tasks: Use the most powerful model available\n\n"
            f"Remember: The goal is maximum accuracy given the intelligence and context requirements of the specific task.\n"
            f"You have a list of allowed models to choose from: {str(self.allowed_model_list)}.\n\n"
            f"Consider the information about the allowed models: \n {self.model_info}\n"
            f"Your response should include the new model choice for the operation that maximizes accuracy given the task complexity and context requirements."
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
    ):
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
            call_cost = resp._hidden_params["response_cost"]
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
        allowed_model_list: List[str] = None,
        **kwargs,
    ):
        """
        Instantiate the directive for a list of operators.
        """
        # Update allowed_model_list if provided
        if allowed_model_list is not None:
            self.allowed_model_list = allowed_model_list

        new_ops_list = deepcopy(operators)
        inst_error = 0
        for target_op in target_ops:
            target_op_config = [op for op in operators if op["name"] == target_op][0]
            # Instantiate the directive
            try:
                rewrite, message_history, call_cost = self.llm_instantiate(
                    global_default_model,
                    target_op_config,
                    agent_llm,
                    message_history,
                )
                print(rewrite)
            except Exception:
                inst_error += 1
            new_ops_list = self.apply(
                global_default_model, new_ops_list, target_op, rewrite
            )

        if inst_error == len(target_ops):
            print("CHANGE MODEL ACC ERROR")
            return None, message_history
        return new_ops_list, message_history, call_cost
