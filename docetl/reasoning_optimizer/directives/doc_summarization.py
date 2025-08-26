import json
import os
from copy import deepcopy
from typing import Dict, List, Type

from litellm import completion
from pydantic import BaseModel, Field

from docetl.reasoning_optimizer.instantiate_schemas import (
    DocSummarizationInstantiateSchema,
)

from .base import MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS, Directive, DirectiveTestCase


class DocSummarizationDirective(Directive):
    name: str = Field(
        default="doc_summarization", description="The name of the directive"
    )
    formal_description: str = Field(default="Op => Map -> Op")
    nl_description: str = Field(
        default=(
            "Adds a Map summarization operator at the very beginning of the pipeline to shorten the document before any downstream operations. "
            "This reduces the number of tokens processed in later steps, saving cost and improving efficiency. "
            "The summary is constructed to include all information required by any downstream operator. "
            "Target operations should be all operators that reference the document key being summarized "
            "(e.g., all ops using {{ input.transcript }})."
        )
    )
    when_to_use: str = Field(
        default=(
            "Use when documents are too long or detailed for the downstream pipeline. "
            "Summarization should preserve essential information and make subsequent tasks more efficient. "
            "Target ops should include all operators that use the document key being summarized, and the summary model should be cheap."
        )
    )

    instantiate_schema_type: Type[BaseModel] = Field(
        default=DocSummarizationInstantiateSchema
    )

    example: str = Field(
        default="""
            "Original Pipeline - Complex medical reasoning across multiple operators:\n"
            "[\n"
            "  {\n"
            "    name: 'assess_drug_interactions',\n"
            "    type: 'map',\n"
            "    prompt: 'Analyze potential drug interactions from this consultation: {{ input.transcript }}. Consider contraindications.',\n"
            "    output: { schema: { interaction_risks: 'list[dict]' } }\n"
            "  },\n"
            "  {\n"
            "    name: 'predict_side_effects',\n"
            "    type: 'map', \n"
            "    prompt: 'Based on the transcript, predict likely side effects for this patient: {{ input.transcript }}. Consider patient demographics and medical history.',\n"
            "    output: { schema: { predicted_effects: 'list[dict]' } }\n"
            "  },\n"
            "  {\n"
            "    name: 'generate_monitoring_plan',\n"
            "    type: 'map',\n"
            "    prompt: 'Create a monitoring plan based on this consultation: {{ input.transcript }}. Focus on symptoms to watch for.',\n"
            "    output: { schema: { monitoring_plan: 'string' } }\n"
            "  }\n"
            "]\n"
            "\n"
            "Problem: 50-page transcript with scheduling, insurance, small talk distracts from medical reasoning.\n"
            "\n"
            "Example InstantiateSchema:\n"
            "[\n"
            "  DocSummarizationConfig(\n"
            "    name='extract_medical_essentials',\n"
            "    document_key='transcript',\n"
            "    prompt='Extract a summary of medical information from this consultation transcript for drug interaction analysis, side effect prediction, and monitoring plan creation. Include: all medications with dosages, patient complaints/symptoms, medical history, current conditions, patient demographics (age, weight), allergies, and any contraindications mentioned. Exclude scheduling, insurance, and casual conversation: {{ input.transcript }}',\n"
            "    model='gpt-4o-mini'\n"
            "  )\n"
            "]\n"
            "\n"
            "Result: All three reasoning operations work on focused medical facts, dramatically improving accuracy."
        """,
    )

    test_cases: List[DirectiveTestCase] = Field(
        default_factory=lambda: [
            DirectiveTestCase(
                name="medical_pipeline_analysis",
                description="Should add summarization for multi-step medical reasoning pipeline",
                input_config=[
                    {
                        "name": "assess_drug_interactions",
                        "type": "map",
                        "prompt": "Analyze potential drug interactions from this consultation: {{ input.transcript }}. Consider contraindications and patient allergies.",
                        "output": {"schema": {"interaction_risks": "list[str]"}},
                    },
                    {
                        "name": "predict_side_effects",
                        "type": "map",
                        "prompt": "Predict likely side effects for this patient based on: {{ input.transcript }}. Consider age, weight, and medical history.",
                        "output": {"schema": {"predicted_effects": "list[str]"}},
                    },
                    {
                        "name": "create_monitoring_plan",
                        "type": "map",
                        "prompt": "Create patient monitoring plan from: {{ input.transcript }}. Focus on symptoms to watch and lab work needed.",
                        "output": {"schema": {"monitoring_plan": "string"}},
                    },
                ],
                target_ops=[
                    "assess_drug_interactions",
                    "predict_side_effects",
                    "create_monitoring_plan",
                ],
                expected_behavior="Should add summarization that extracts comprehensive medical information (medications, dosages, allergies, demographics, symptoms, history) needed for all three downstream reasoning tasks",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="legal_contract_pipeline",
                description="Should add summarization for multi-step legal analysis pipeline",
                input_config=[
                    {
                        "name": "identify_liability_risks",
                        "type": "map",
                        "prompt": "Identify liability and indemnification risks in: {{ input.contract_document }}. Focus on limitation of liability clauses.",
                        "output": {"schema": {"liability_risks": "list[str]"}},
                    },
                    {
                        "name": "analyze_termination_terms",
                        "type": "map",
                        "prompt": "Analyze termination and breach conditions in: {{ input.contract_document }}. Include notice requirements and penalties.",
                        "output": {"schema": {"termination_analysis": "string"}},
                    },
                    {
                        "name": "assess_compliance_requirements",
                        "type": "map",
                        "prompt": "Assess regulatory compliance obligations from: {{ input.contract_document }}. Include data protection and industry standards.",
                        "output": {"schema": {"compliance_requirements": "list[str]"}},
                    },
                ],
                target_ops=[
                    "identify_liability_risks",
                    "analyze_termination_terms",
                    "assess_compliance_requirements",
                ],
                expected_behavior="Should add summarization that extracts all legally relevant clauses, terms, obligations, penalties, and compliance requirements needed across the legal analysis pipeline",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="financial_analysis_pipeline",
                description="Should add summarization for multi-step investment analysis pipeline",
                input_config=[
                    {
                        "name": "evaluate_financial_health",
                        "type": "map",
                        "prompt": "Evaluate company financial health from: {{ input.annual_report }}. Calculate key ratios and assess profitability trends.",
                        "output": {
                            "schema": {
                                "financial_metrics": "str",
                                "health_score": "float",
                            }
                        },
                    },
                    {
                        "name": "assess_market_position",
                        "type": "map",
                        "prompt": "Assess competitive market position using: {{ input.annual_report }}. Analyze market share and competitive advantages.",
                        "output": {
                            "schema": {
                                "market_analysis": "string",
                                "competitive_strengths": "list[str]",
                            }
                        },
                    },
                    {
                        "name": "identify_growth_risks",
                        "type": "map",
                        "prompt": "Identify growth opportunities and risks from: {{ input.annual_report }}. Include regulatory and market risks.",
                        "output": {
                            "schema": {
                                "growth_opportunities": "list[str]",
                                "risk_factors": "list[str]",
                            }
                        },
                    },
                ],
                target_ops=[
                    "evaluate_financial_health",
                    "assess_market_position",
                    "identify_growth_risks",
                ],
                expected_behavior="Should add summarization that extracts financial data, market information, competitive landscape, and risk factors needed for comprehensive investment analysis",
                should_pass=True,
            ),
        ]
    )

    def __eq__(self, other):
        return isinstance(other, DocSummarizationDirective)

    def __hash__(self):
        return hash("DocSummarizationDirective")

    def to_string_for_instantiate(
        self, operators: List[Dict], target_ops: List[str]
    ) -> str:
        """
        Generate a prompt for an agent to instantiate this directive.

        Args:
            operators: List of all operators in the pipeline
            target_ops: List of target operation names that need the summarized content

        Returns:
            str: The agent prompt for instantiating the directive.
        """
        return (
            f"You are an expert at adding document summarization to data processing pipelines.\n\n"
            f"Full Pipeline Context:\n"
            f"{str(operators)}\n\n"
            f"Target Operations (that will use summarized content): {target_ops}\n\n"
            f"Directive: {self.name}\n"
            f"Your task is to instantiate this directive by generating a DocSummarizationConfig that adds a Map summarization operator "
            f"at the start of the pipeline.\n\n"
            f"Analysis steps:\n"
            f"1. Identify which input field contains long documents that could benefit from summarization\n"
            f"2. Analyze ALL target operations' prompts to understand what information each needs\n"
            f"3. Create a comprehensive summarization prompt that preserves ALL information needed by ANY target operation\n"
            f"4. Ensure the summary contains sufficient detail for all downstream reasoning tasks\n\n"
            f"The document_key should be the field name containing the long content to summarize.\n"
            f"The prompt should instruct the LLM to extract and preserve ALL information types needed across ALL target operations.\n"
            f"The output will replace the original document field with the summarized version.\n\n"
            f"Example:\n"
            f"{self.example}\n\n"
            f"Please output only the InstantiateSchema (a DocSummarizationConfig object)."
        )

    def llm_instantiate(
        self,
        operators: List[Dict],
        target_ops: List[str],
        agent_llm: str,
        message_history: list = [],
    ):
        """
        Use LLM to instantiate this directive by creating a summarization operation.

        Args:
            operators (List[Dict]): All operators in the pipeline.
            target_ops (List[str]): Target operation names that need summarized content.
            agent_llm (str): The LLM model to use.
            message_history (List, optional): Conversation history for context.

        Returns:
            DocSummarizationInstantiateSchema: The structured output from the LLM.
        """

        message_history.extend(
            [
                {
                    "role": "user",
                    "content": self.to_string_for_instantiate(operators, target_ops),
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
                response_format=DocSummarizationInstantiateSchema,
            )
            call_cost = resp._hidden_params["response_cost"]
            try:
                parsed_res = json.loads(resp.choices[0].message.content)
                schema = DocSummarizationInstantiateSchema(**parsed_res)
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
        rewrite: DocSummarizationInstantiateSchema,
    ) -> List[Dict]:
        """
        Apply the directive to the pipeline config by adding a summarization Map operator at the start.
        """
        # Create a copy of the pipeline config
        new_ops_list = deepcopy(ops_list)

        # Create the summarization Map operator using the LLM-generated name
        summarization_op = {
            "name": rewrite.name,
            "type": "map",
            "prompt": rewrite.prompt,
            "model": rewrite.model,
            "litellm_completion_kwargs": {"temperature": 0},
            "output": {"schema": {rewrite.document_key: "string"}},
        }

        # Insert the summarization operator at the beginning of the pipeline
        new_ops_list.insert(0, summarization_op)

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
        # Validate that target ops exist in the pipeline
        operator_names = {op["name"] for op in operators}
        for target_op in target_ops:
            if target_op not in operator_names:
                raise ValueError(
                    f"Target operation '{target_op}' not found in pipeline"
                )

        # Instantiate the directive using full pipeline context
        rewrite, message_history, call_cost = self.llm_instantiate(
            operators, target_ops, agent_llm, message_history
        )

        # Apply the rewrite to the operators (use first target op for compatibility with apply method)
        return (
            self.apply(global_default_model, operators, target_ops[0], rewrite),
            message_history,
            call_cost,
        )
