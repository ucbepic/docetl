import json
import os
from copy import deepcopy
from typing import Dict, List, Type

from litellm import completion
from pydantic import BaseModel, Field

from docetl.reasoning_optimizer.instantiate_schemas import GleaningInstantiateSchema

from .base import MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS, Directive, DirectiveTestCase


class ReduceGleaningDirective(Directive):
    name: str = Field(
        default="reduce_gleaning", description="The name of the directive"
    )
    formal_description: str = Field(default="Reduce => Reduce_m (with gleaning config)")
    nl_description: str = Field(
        default="""Adds a validation loop to Reduce operations: after each LLM generation during the reduce process, a separate "judge" LLM evaluates the output using a yes/no validation prompt. If the output fails, the original LLM refines its answer and repeats until the output passes or the max number of rounds is reached."""
    )
    when_to_use: str = Field(
        default="When reduce operations process complex documents that require comprehensive analysis and synthesis (e.g., research paper analysis, customer feedback consolidation, legal document review, literature synthesis) where outputs must be validated for completeness, accuracy, and proper coverage of all input materials."
    )

    instantiate_schema_type: Type[BaseModel] = Field(default=GleaningInstantiateSchema)

    example: str = Field(
        default="""
            Original Op (ReduceOpConfig):
            - name: synthesize_research_findings
              type: reduce
              reduce_key: research_domain
              prompt: |
                You are analyzing research papers in the {{ research_domain }} domain.
                Synthesize the following research findings into a comprehensive domain overview:

                {% for paper in inputs %}
                **Paper {{ loop.index }}:**
                - Title: {{ paper.title }}
                - Key Findings: {{ paper.key_findings }}
                - Methodology: {{ paper.methodology }}
                - Limitations: {{ paper.limitations }}
                - Future Work: {{ paper.future_work }}

                {% endfor %}

                Generate a synthesis with the following structure:
                - **Domain Overview**: 2-3 sentences describing the field
                - **Major Findings**: List of 4-6 key insights across all papers
                - **Methodological Approaches**: Summary of research methods used
                - **Research Gaps**: Identified limitations and areas needing investigation
                - **Future Directions**: Consolidated recommendations for future research
              output:
                schema:
                  domain_overview: "string"
                  major_findings: "list[str]"
                  methodological_approaches: "string"
                  research_gaps: "list[str]"
                  future_directions: "list[str]"

            Example InstantiateSchema (what the agent should output):
            {
              "validation_prompt": "Verify that the synthesis includes all required sections (domain overview, major findings, methodological approaches, research gaps, future directions). Each major finding should be supported by evidence from multiple papers. Research gaps should be specific and actionable. The domain overview should accurately represent the scope covered by the input papers.",
              "num_rounds": 3,
              "model": "gpt-4o-mini"
            }
        """,
    )

    test_cases: List[DirectiveTestCase] = Field(
        default_factory=lambda: [
            DirectiveTestCase(
                name="customer_feedback_analysis",
                description="Should add validation for comprehensive customer feedback analysis by product category",
                input_config={
                    "name": "analyze_feedback_by_product",
                    "type": "reduce",
                    "reduce_key": "product_category",
                    "prompt": """Analyze customer feedback for {{ product_category }} products.
                    Create a comprehensive analysis from the following feedback:

                    {% for review in inputs %}
                    Review {{ loop.index }}:
                    - Rating: {{ review.rating }}/5
                    - Comment: {{ review.comment }}
                    - Customer Type: {{ review.customer_type }}
                    - Date: {{ review.date }}
                    {% endfor %}

                    Provide analysis with:
                    - Overall sentiment and satisfaction level
                    - Top 3 most praised features
                    - Top 3 most criticized issues
                    - Recommendations for product improvements
                    - Customer segment insights""",
                    "output": {
                        "schema": {
                            "overall_sentiment": "string",
                            "satisfaction_level": "string",
                            "top_praised_features": "list[str]",
                            "top_criticized_issues": "list[str]",
                            "improvement_recommendations": "list[str]",
                            "segment_insights": "string",
                        }
                    },
                },
                target_ops=["analyze_feedback_by_product"],
                expected_behavior="Should add gleaning validation to ensure all feedback is considered, sentiment analysis is accurate, and recommendations are actionable",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="legal_contract_analysis",
                description="Should add validation for thorough legal contract analysis by contract type",
                input_config={
                    "name": "analyze_contracts_by_type",
                    "type": "reduce",
                    "reduce_key": "contract_type",
                    "prompt": """Analyze {{ contract_type }} contracts and extract key legal provisions.

                    Review the following contracts:
                    {% for contract in inputs %}
                    Contract {{ loop.index }}:
                    - Party Names: {{ contract.parties }}
                    - Key Terms: {{ contract.key_terms }}
                    - Obligations: {{ contract.obligations }}
                    - Termination Clauses: {{ contract.termination }}
                    - Risk Factors: {{ contract.risks }}
                    {% endfor %}

                    Provide comprehensive analysis including:
                    - Common contractual patterns across all contracts
                    - Standard terms and deviations
                    - Risk assessment summary
                    - Compliance requirements identified
                    - Recommendations for contract standardization""",
                    "output": {
                        "schema": {
                            "common_patterns": "list[str]",
                            "standard_terms": "list[str]",
                            "risk_assessment": "string",
                            "compliance_requirements": "list[str]",
                            "standardization_recommendations": "list[str]",
                        }
                    },
                },
                target_ops=["analyze_contracts_by_type"],
                expected_behavior="Should add gleaning validation to ensure all contract provisions are captured, risk analysis is thorough, and recommendations are legally sound",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="research_literature_synthesis",
                description="Should add validation for comprehensive literature synthesis by research topic",
                input_config={
                    "name": "synthesize_literature_by_topic",
                    "type": "reduce",
                    "reduce_key": "research_topic",
                    "prompt": """Synthesize research literature on {{ research_topic }}.

                    Analyze the following academic papers:
                    {% for paper in inputs %}
                    Paper {{ loop.index }}:
                    - Title: {{ paper.title }}
                    - Abstract: {{ paper.abstract }}
                    - Methodology: {{ paper.methodology }}
                    - Key Results: {{ paper.results }}
                    - Conclusions: {{ paper.conclusions }}
                    - Limitations: {{ paper.limitations }}
                    {% endfor %}

                    Create a literature synthesis with:
                    - Theoretical frameworks identified across papers
                    - Consensus findings and contradictory results
                    - Methodological approaches comparison
                    - Research gaps and limitations summary
                    - Future research directions and recommendations""",
                    "output": {
                        "schema": {
                            "theoretical_frameworks": "list[str]",
                            "consensus_findings": "list[str]",
                            "contradictory_results": "list[str]",
                            "methodological_comparison": "string",
                            "research_gaps": "list[str]",
                            "future_directions": "list[str]",
                        }
                    },
                },
                target_ops=["synthesize_literature_by_topic"],
                expected_behavior="Should add gleaning validation to ensure all papers are properly synthesized, contradictions are identified, and research gaps are accurately captured",
                should_pass=True,
            ),
        ]
    )

    def __eq__(self, other):
        return isinstance(other, ReduceGleaningDirective)

    def __hash__(self):
        return hash("ReduceGleaningDirective")

    def to_string_for_instantiate(self, original_op: Dict) -> str:
        """
        Generate a prompt for an agent to instantiate this directive.

        Args:
            original_op (str): The YAML or string representation of the original operation.

        Returns:
            str: The agent prompt for instantiating the directive.
        """
        return (
            f"You are an expert at adding validation and refinement loops to reduce operations for document processing tasks.\n\n"
            f"Original Operation:\n"
            f"{str(original_op)}\n\n"
            f"Directive: {self.name}\n"
            f"Your task is to instantiate this directive by generating a configuration that adds validation loops to the reduce operation. "
            f"The gleaning configuration should include a validation prompt that evaluates the quality of the reduce output, "
            f"focusing on document analysis criteria such as:\n"
            f"- Completeness: Are all input documents/items properly considered and synthesized?\n"
            f"- Accuracy: Are the extracted insights, patterns, and conclusions accurate?\n"
            f"- Structure: Does the output follow the required format and include all requested fields?\n"
            f"- Comprehensiveness: Are key themes, patterns, and insights captured across all inputs?\n"
            f"- Consistency: Are the analysis and recommendations internally consistent?\n\n"
            f"For reduce operations, the LLM processes groups of related documents and creates consolidated, synthesized outputs. "
            f"The validation should ensure proper document analysis, synthesis quality, and adherence to output requirements.\n\n"
            f"Example:\n"
            f"{self.example}\n\n"
            f"Please output only the GleaningInstantiateSchema object that specifies how to validate and refine the output of the reduce operation."
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
                azure=True,
                response_format=GleaningInstantiateSchema,
            )
            call_cost = resp._hidden_params["response_cost"]
            try:
                parsed_res = json.loads(resp.choices[0].message.content)
                schema = GleaningInstantiateSchema(**parsed_res)
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
        rewrite: GleaningInstantiateSchema,
    ) -> List[Dict]:
        """
        Apply the directive to the pipeline config by adding gleaning configuration to the target reduce operator.
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
        ), "There must be exactly one target op to instantiate this reduce gleaning directive"
        target_op_config = [op for op in operators if op["name"] == target_ops[0]][0]

        # Verify the target operation is a reduce operation
        if target_op_config.get("type") != "reduce":
            raise ValueError(
                f"ReduceGleaningDirective can only be applied to reduce operations, got {target_op_config.get('type')}"
            )

        # Instantiate the directive
        rewrite, message_history, call_cost = self.llm_instantiate(
            target_op_config, agent_llm, message_history
        )

        # Apply the rewrite to the operators
        return (
            self.apply(global_default_model, operators, target_ops[0], rewrite),
            message_history, call_cost
        )
