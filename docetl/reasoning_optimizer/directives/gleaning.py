import json
import os
from copy import deepcopy
from typing import Dict, List, Type

from litellm import completion
from pydantic import BaseModel, Field

from docetl.reasoning_optimizer.instantiate_schemas import GleaningInstantiateSchema

from .base import MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS, Directive, DirectiveTestCase


class GleaningDirective(Directive):
    name: str = Field(default="gleaning", description="The name of the directive")
    formal_description: str = Field(default="Map => Map_m (with gleaning config)")
    nl_description: str = Field(
        default="""Adds a validation loop to Map: after each LLM generation, a separate "judge" LLM evaluates the output using a yes/no validation prompt. If the output fails, the original LLM refines its answer and repeats until the output passes or the max number of rounds is reached."""
    )
    when_to_use: str = Field(
        default="When initial Map outputs may not meet quality criteria and must be checked or improved automatically (e.g., too short, missing required info)."
    )

    # Remove from Pydantic fields, make it a plain class variable
    instantiate_schema_type: Type[BaseModel] = Field(default=GleaningInstantiateSchema)

    example: str = Field(
        default="""
            Original Op (MapOpConfig):
            - name: extract_insights
              type: map
              prompt: |
                From the user log below, list 2-3 concise insights (1-2 words each) and 1-2 supporting actions per insight.
                Return as a list of dictionaries with 'insight' and 'supporting_actions'.
                Log: {{ input.log }}
              output:
                schema:
                  insights_summary: "string"

            Example InstantiateSchema (what the agent should output):
            {
              "validation_prompt": "There should be at least 2 insights, and each insight should have at least 1 supporting action.",
              "num_rounds": 2,
              "model": "gpt-4o-mini"
            }
        """,
    )

    test_cases: List[DirectiveTestCase] = Field(
        default_factory=lambda: [
            DirectiveTestCase(
                name="quality_validation_insights",
                description="Should add validation for insight extraction quality",
                input_config={
                    "name": "extract_insights",
                    "type": "map",
                    "prompt": "From the user log below, list 2-3 concise insights and supporting actions: {{ input.log }}",
                    "output": {"schema": {"insights_summary": "string"}},
                },
                target_ops=["extract_insights"],
                expected_behavior="Should add gleaning config with validation prompt checking for minimum number of insights and supporting actions",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="contract_term_validation",
                description="Should add validation for contract term extraction completeness",
                input_config={
                    "name": "extract_contract_terms",
                    "type": "map",
                    "prompt": "Extract payment terms, deadlines, and penalties from: {{ input.contract }}",
                    "output": {"schema": {"terms": "list[str]"}},
                },
                target_ops=["extract_contract_terms"],
                expected_behavior="Should add gleaning validation to ensure all required term types are extracted",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="financial_report_analysis",
                description="Should add validation for financial analysis accuracy",
                input_config={
                    "name": "analyze_financial_report",
                    "type": "map",
                    "prompt": "Extract revenue, expenses, profit margins, and key financial ratios from: {{ input.report }}",
                    "output": {"schema": {"financial_analysis": "string"}},
                },
                target_ops=["analyze_financial_report"],
                expected_behavior="Should add gleaning validation to ensure all financial metrics are accurately extracted and calculated",
                should_pass=True,
            ),
        ]
    )

    def __eq__(self, other):
        return isinstance(other, GleaningDirective)

    def __hash__(self):
        return hash("GleaningDirective")

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
            f"Your task is to instantiate this directive by generating a configuration that adds validation loops to the original operation. "
            f"The gleaning configuration should include a validation prompt that evaluates the output quality and provides feedback for improvement, "
            f"along with the number of refinement rounds to attempt. In the prompt, you shuold not use any Jinja variables. \n"
            f"Example:\n"
            f"{self.example}\n\n"
            f"Please output only the GleaningInstantiateSchema object that specifies how to validate and refine the output of the original operation."
        )

    def llm_instantiate(
        self,
        original_op: Dict,
        agent_llm: str,
        message_history: list = [],
    ):
        """
        Use LLM to instantiate this directive by decomposing the original operation.

        Args:
            original_op (Dict): The original operation.
            agent_llm (str): The LLM model to use.
            message_history (List, optional): Conversation history for context.

        Returns:
            GleaningInstantiateSchema: The structured output from the LLM.
        """

        message_history.extend(
            [
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
                # api_key=os.environ["GEMINI_API_KEY"],
                azure=True,
                response_format=GleaningInstantiateSchema,
            )
            call_cost = resp._hidden_params["response_cost"]
            try:
                parsed_res = json.loads(resp.choices[0].message.content)
                schema = GleaningInstantiateSchema(**parsed_res)
                GleaningInstantiateSchema.check_no_jinja_variables(schema.validation_prompt)
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
        default_model: str,
        ops_list: List[Dict],
        target_op: str,
        rewrite: GleaningInstantiateSchema,
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

        # Add gleaning configuration to the target operator
        target_operator = new_ops_list[pos_to_replace]
        target_operator["gleaning"] = {
            "validation_prompt": rewrite.validation_prompt,
            "num_rounds": rewrite.num_rounds,
            "model": rewrite.model,
        }

        return new_ops_list

    def instantiate(
        self,
        operators: List[Dict],
        target_ops: List[str],
        agent_llm: str,
        message_history: list = [],
        global_default_model: str = None,
        **kwargs,
    ):
        """
        Instantiate the directive for a list of operators.
        """
        # Assert that there is only one target op
        assert (
            len(target_ops) == 1
        ), "There must be exactly one target op to instantiate this chaining directive"
        target_op_config = [op for op in operators if op["name"] == target_ops[0]][0]

        # Instantiate the directive
        rewrite, message_history, call_cost = self.llm_instantiate(
            target_op_config, agent_llm, message_history
        )

        # Apply the rewrite to the operators
        return (
            self.apply(global_default_model, operators, target_ops[0], rewrite),
            message_history,
            call_cost,
        )
