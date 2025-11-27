import json
import os
from copy import deepcopy
from typing import Dict, List, Type

from litellm import completion
from pydantic import BaseModel, Field

from docetl.reasoning_optimizer.instantiate_schemas import ChangeModelInstantiateSchema

from .base import MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS, Directive, DirectiveTestCase


class ChangeModelDirective(Directive):
    name: str = Field(default="change model", description="The name of the directive")
    formal_description: str = Field(
        default="Op => Op* (same operation with a different model choice)"
    )
    nl_description: str = Field(
        default="Rewrites an operator to use a different LLM model based on task requirements. Generally, simpler tasks like extraction or classification may work well with cheaper models (gpt-4o-mini, gpt-5-nano), while complex reasoning tasks often benefit from more powerful models (gpt-5), though actual performance and constraints should guide the choice."
    )
    when_to_use: str = Field(
        default="When the current model choice may not be optimal for the task requirements, considering factors like task complexity, performance needs, cost constraints, and quality requirements."
    )
    instantiate_schema_type: Type[BaseModel] = ChangeModelInstantiateSchema

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
            '      insights_summary: "string"\n'
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
                name="cost_optimization_simple_task",
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
                name="complex_analysis_needs_powerful_model",
                description="Should recommend powerful model for complex reasoning task",
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
                name="high_volume_processing_optimization",
                description="Should optimize model for high-volume document processing",
                input_config={
                    "name": "batch_document_summary",
                    "type": "map",
                    "prompt": "Summarize this document in 2-3 sentences: {{ input.document }}",
                    "output": {"schema": {"summary": "str"}},
                    "model": "gpt-5",
                },
                target_ops=["batch_document_summary"],
                expected_behavior="Should recommend faster/cheaper model like gpt-4o-mini or gpt-5-nano for high-volume simple summarization",
                should_pass=True,
            ),
        ]
    )

    def __eq__(self, other):
        return isinstance(other, ChangeModelDirective)

    def __hash__(self):
        return hash("ChangeModelDirective")

    def to_string_for_instantiate(self, original_op: Dict, optimize_goal) -> str:
        """
        Generate a prompt for an agent to instantiate this directive.

        Args:
            original_op (str): The YAML or string representation of the original operation.

        Returns:
            str: The agent prompt for instantiating the directive.
        """
        if optimize_goal == "acc":
            return (
                f"You are an expert at choosing the most suitable model for a given task based on complexity and cost considerations.\n\n"
                f"Original Operation:\n"
                f"{str(original_op)}\n\n"
                f"Directive: {self.name}\n"
                f"Your task is to instantiate this directive by suggesting a better model for executing the original operation.\n\n"
                f"MODEL SELECTION CONSIDERATIONS:\n"
                f"• Generally, simpler tasks (extraction, basic classification, straightforward summarization) may work well with cheaper models like gpt-4o-mini or gpt-5-nano\n"
                f"• Complex reasoning tasks (analysis, interpretation, multi-step thinking, legal/medical analysis) often benefit from more powerful models like gpt-5\n"
                f"• However, consider actual performance needs, quality requirements, and cost constraints when making the choice\n"
                f"• Sometimes a powerful model may be needed for seemingly simple tasks if quality is critical, or a cheaper model may suffice for complex tasks if budget is constrained\n\n"
                f"You have a list of allowed models to choose from: {str(self.allowed_model_list)}.\n\n"
                f"Consider the information about the allowed models: \n {self.model_info}\n"
                f"Your response should include the new model choice for the operation."
                f"Ensure that your chosen model is in the list of allowed models."
                f"Example:\n"
                f"{self.example}\n\n"
                f"Please output only the ChangeModelInstantiateSchema as JSON."
            )
        else:
            return (
                f"You are an expert at choosing the most suitable model for a given task to optimize cost.\n\n"
                f"Original Operation:\n"
                f"{str(original_op)}\n\n"
                f"Directive: {self.name}\n"
                f"Your task is to instantiate this directive by generating a ChangeModelConfig that suggests a more cost-effective model for executing the original operation."
                f"You have a list of allowed models to choose from: {str(self.allowed_model_list)}.\n\n"
                f"Consider the information about the allowed models: \n {self.model_info}\n"
                f"The ChangeModelConfig should include the new model choice for the operation that reduces costs while maintaining adequate performance."
                f"Ensure that your chosen model is in the list of allowed models."
                f"Example:\n"
                f"{self.example}\n\n"
                f"Please output only the InstantiateSchema (a ChangeModelConfig object)."
            )

    def llm_instantiate(
        self,
        global_default_model: str,
        original_op: Dict,
        agent_llm: str,
        message_history: list = [],
        optimize_goal="acc",
    ):
        """
        Use LLM to instantiate this directive.

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
                    "content": "You are a helpful AI assistant for document processing pipelines.",
                },
                {
                    "role": "user",
                    "content": self.to_string_for_instantiate(
                        original_op, optimize_goal
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
                # api_key=os.environ["GEMINI_API_KEY"],
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
        Apply the directive to the pipeline config by adding gleaning configuration to the target operator.
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
        optimize_goal="acc",
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
                    optimize_goal=optimize_goal,
                )
                print(rewrite)
            except Exception:
                inst_error += 1
            new_ops_list = self.apply(
                global_default_model, new_ops_list, target_op, rewrite
            )

        if inst_error == len(target_ops):
            print("CHANEG MODEL ERROR")
            return None, message_history
        return new_ops_list, message_history, call_cost
