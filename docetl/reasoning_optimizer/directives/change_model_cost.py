import json
import os
from copy import deepcopy
from typing import Dict, List, Type

from litellm import completion
from pydantic import BaseModel, Field

from docetl.reasoning_optimizer.instantiate_schemas import ChangeModelInstantiateSchema


def get_cheaper_models(current_model: str, model_stats: Dict = None) -> List[str]:
    """
    Get list of models that are cheaper than the current model.

    Args:
        current_model: The current model name (may include "azure/" or "gemini/" prefix)
        model_stats: Dictionary of model statistics from MOARSearch

    Returns:
        List of model names that are cheaper than current_model, sorted by cost (cheapest first)
    """
    if model_stats is None or not model_stats:
        return []

    current_model_stats = model_stats.get(current_model)
    if current_model_stats is None:
        return []

    current_cost = current_model_stats.get("cost")
    if current_cost is None:
        return []

    cheaper_models = []
    for model, stats in model_stats.items():
        if not isinstance(stats, dict):
            continue
        model_cost = stats.get("cost")
        if model_cost is not None and model_cost < current_cost:
            cheaper_models.append(model)

    return cheaper_models


from .base import (  # noqa: E402
    MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS,
    Directive,
    DirectiveTestCase,
)


def create_model_specific_directives(
    current_model: str, allowed_model_list: List[str] = None
):
    """Create model-specific directives for the current model."""
    directive = ChangeModelCostDirective(target_model=current_model)
    directive.name = f"change to {current_model}"
    directive.nl_description = f"Rewrites an operator to use the {current_model} model to optimize expenses while maintaining adequate performance."
    if allowed_model_list is not None:
        directive.allowed_model_list = allowed_model_list
    else:
        directive.allowed_model_list = [
            current_model
        ]  # Only allow this specific model if no list provided

    return directive


class ChangeModelCostDirective(Directive):
    name: str = Field(
        default="change model cost", description="The name of the directive"
    )
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
        default_factory=list,
        description="The allowed list of models to choose from",
    )

    target_model: str = Field(
        default="",
        description="The specific target model for this directive instance",
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
        return (
            isinstance(other, ChangeModelCostDirective)
            and self.target_model == other.target_model
        )

    def __hash__(self):
        return hash(f"ChangeModelCostDirective_{self.target_model}")

    def to_string_for_instantiate(
        self, original_op: Dict, dataset: str, model_stats: Dict = None
    ) -> str:
        """
        Generate a prompt for an agent to instantiate this directive for cost optimization.

        Args:
            original_op (str): The YAML or string representation of the original operation.
            dataset: The dataset name
            model_stats: Dictionary of model statistics from MOARSearch (optional)

        Returns:
            str: The agent prompt for instantiating the directive.
        """
        model_stats_str = ""
        if model_stats:
            model_stats_str = f"You have a list of model statistics on the task with the original query pipeline: \n {str(model_stats)}\n"

        return (
            f"You are an expert at choosing the most cost-effective model for a given task while maintaining adequate performance.\n\n"
            f"Original Operation:\n"
            f"{str(original_op)}\n\n"
            f"Directive: {self.name}\n"
            f"Your task is to instantiate this directive by suggesting the cheapest model that meets the requirements for executing the original operation.\n\n"
            f"COST OPTIMIZATION STRATEGY:\n"
            f"• Choose the cheapest model that meets the task requirements\n"
            f"• For tasks requiring ultra-long context (1M+ context window), use gpt-4.1 series or gemini models\n"
            f"• For tasks that can work with document samples or fit within 272k context window, use gpt-5-nano\n"
            f"• For simple tasks (extraction, basic classification, straightforward summarization), use the cheapest available model like gpt-4o-mini or gpt-5-nano\n"
            f"• For high-volume processing where cost accumulates quickly, prioritize the most economical option\n"
            f"• Only use expensive models if the task absolutely requires capabilities not available in cheaper alternatives\n"
            f"• Consider document length and context requirements when selecting models\n\n"
            f"You have a list of allowed models to choose from: {str(self.allowed_model_list)}.\n\n"
            f"Consider the information about the allowed models: \n {self.model_info}\n"
            f"{model_stats_str}"
            f"Your response should include the cheapest model choice that meets the operation requirements."
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
        dataset: str,
        message_history: list = [],
        model_stats: Dict = None,
    ):
        """
        Use LLM to instantiate this directive for cost optimization.

        Args:
            original_op (Dict): The original operation.
            agent_llm (str): The LLM model to use.
            dataset: The dataset name
            message_history (List, optional): Conversation history for context.
            model_stats: Dictionary of model statistics from MOARSearch (optional)

        Returns:
            ChangeModelInstantiateSchema: The structured output from the LLM.
        """

        # If target_model is specified, use it directly without LLM call
        if self.target_model:
            schema = ChangeModelInstantiateSchema(model=self.target_model)
            return schema, message_history, 0.0

        # Otherwise, use LLM to choose the model
        message_history.extend(
            [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant for document processing pipelines focused on cost optimization.",
                },
                {
                    "role": "user",
                    "content": self.to_string_for_instantiate(
                        original_op, dataset, model_stats
                    ),
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
        dataset: str = None,
        model_stats: Dict = None,
        allowed_model_list: List[str] = None,
        **kwargs,
    ):
        """
        Instantiate the directive for a list of operators.

        Args:
            operators: List of operator configurations
            target_ops: List of target operation names
            agent_llm: LLM model to use for instantiation
            message_history: Conversation history
            global_default_model: Default model for the pipeline
            dataset: Dataset name
            model_stats: Dictionary of model statistics from MOARSearch (optional)
            allowed_model_list: List of allowed models (optional)
            **kwargs: Additional keyword arguments
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
                    dataset,
                    message_history,
                    model_stats,
                )
                print(rewrite)
            except Exception:
                inst_error += 1
            new_ops_list = self.apply(
                global_default_model, new_ops_list, target_op, rewrite
            )

        if inst_error == len(target_ops):
            print("CHANGE MODEL COST ERROR")
            return None, message_history
        return new_ops_list, message_history, call_cost
