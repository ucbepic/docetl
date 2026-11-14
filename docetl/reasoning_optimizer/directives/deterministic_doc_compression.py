import json
from copy import deepcopy
from typing import Dict, List, Type

from litellm import completion
from pydantic import BaseModel, Field

from docetl.reasoning_optimizer.instantiate_schemas import (
    DeterministicDocCompressionInstantiateSchema,
)

from .base import MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS, Directive, DirectiveTestCase

# TODO: For the agent instantiating the rewrite directive,
# we might want to allow it to look at some example documents /
# enough documents until it feels like it has a good understanding of the data.


class DeterministicDocCompressionDirective(Directive):
    name: str = Field(
        default="deterministic_doc_compression", description="The name of the directive"
    )
    formal_description: str = Field(default="Op => Regex Compress -> Op")
    nl_description: str = Field(
        default="Reduces LLM processing costs by keeping only the paragraphs of a document that match regular expressions (a regex_compress operation, no LLM call) before expensive downstream operations, removing irrelevant content that could distract the LLM"
    )
    when_to_use: str = Field(
        default="When documents contain identifiable patterns or keywords and you want to reduce token costs for downstream LLM operations while improving accuracy by eliminating distracting irrelevant content"
    )

    instantiate_schema_type: Type[BaseModel] = Field(
        default=DeterministicDocCompressionInstantiateSchema
    )

    example: str = Field(
        default="""
        Target Operations:
        - name: analyze_regulatory_compliance
          type: map
          prompt: |
            Analyze regulatory compliance issues in this legal document: {{ input.legal_document }}
            Focus on identifying violations, required actions, and compliance deadlines.
          output:
            schema:
              violations: "list[str]"
              required_actions: "list[str]"
              deadlines: "list[str]"

        Example InstantiateSchema (what the agent should output):
        {
          "name": "keep_compliance_paragraphs",
          "document_key": "legal_document",
          "keep_patterns": [
            "(?i)\\b(violat(e|ion)|breach|non-?complian(ce|t))",
            "(?i)\\b(deadline|due date|expir(e|es|ation)|within \\d+ days)",
            "(?i)\\b(shall|must|required|mandatory|obligation)\\b"
          ],
          "drop_patterns": ["(?i)^\\s*table of contents"],
          "remove_patterns": ["(?m)^\\s*Page \\d+( of \\d+)?\\s*$"],
          "keep_first_chars": 1500
        }
        """
    )

    test_cases: List[DirectiveTestCase] = Field(
        default_factory=lambda: [
            DirectiveTestCase(
                name="detailed_merger_agreement_analysis",
                description="Should compress merger agreement for comprehensive legal analysis",
                input_config={
                    "name": "analyze_merger_agreement_terms",
                    "type": "map",
                    "prompt": """Perform a comprehensive legal analysis of this merger agreement: {{ input.merger_agreement }}

                    Analyze and extract the following:
                    1. Purchase price structure and payment mechanisms (cash, stock, earnouts, escrow arrangements)
                    2. Material adverse change (MAC) definitions and carve-outs that could affect deal completion
                    3. Representations and warranties with survival periods and liability caps
                    4. Closing conditions precedent, including regulatory approvals and third-party consents
                    5. Termination rights and associated breakup fees or reverse breakup fees
                    6. Indemnification provisions including baskets, caps, and survival periods
                    7. Employee retention arrangements and change-in-control provisions
                    8. Integration planning requirements and operational restrictions during pendency
                    9. Dispute resolution mechanisms and governing law provisions
                    10. Post-closing adjustments and working capital mechanisms

                    For each area, provide specific clause references, dollar amounts where applicable,
                    time periods, and risk assessment (High/Medium/Low) with justification.""",
                    "output": {
                        "schema": {
                            "purchase_price_analysis": "string",
                            "mac_provisions": "list[str]",
                            "representations_warranties": "list[str]",
                            "closing_conditions": "list[str]",
                            "termination_rights": "string",
                            "indemnification_terms": "string",
                            "employee_provisions": "list[str]",
                            "integration_restrictions": "list[str]",
                            "dispute_resolution": "string",
                            "post_closing_adjustments": "string",
                            "risk_assessment": "string",
                        }
                    },
                },
                target_ops=["analyze_merger_agreement_terms"],
                expected_behavior="Should add a regex_compress operation on merger_agreement whose keep_patterns match legal terms, financial provisions, and risk-related clauses.",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="multi_document_analysis_compression",
                description="Should compress document for multiple analysis operations",
                input_config=[
                    {
                        "name": "extract_financial_metrics",
                        "type": "map",
                        "prompt": "Extract revenue, profit, and expense figures from: {{ input.earnings_report }}",
                        "output": {
                            "schema": {
                                "revenue": "string",
                                "profit": "string",
                                "expenses": "list[str]",
                            }
                        },
                    },
                    {
                        "name": "assess_financial_risks",
                        "type": "map",
                        "prompt": "Identify financial risks and warning signs in: {{ input.earnings_report }}",
                        "output": {
                            "schema": {
                                "risks": "list[str]",
                                "warning_signs": "list[str]",
                            }
                        },
                    },
                ],
                target_ops=["extract_financial_metrics", "assess_financial_risks"],
                expected_behavior="Should add a regex_compress operation on earnings_report whose keep_patterns match the financial content both operations need.",
                should_pass=True,
            ),
        ]
    )

    def __eq__(self, other):
        return isinstance(other, DeterministicDocCompressionDirective)

    def __hash__(self):
        return hash("DeterministicDocCompressionDirective")

    def to_string_for_instantiate(self, target_ops_configs: List[Dict]) -> str:
        """
        Generate a prompt that asks the agent to output the instantiate schema.
        This prompt explains to the LLM what configuration it needs to generate.
        """
        ops_str = "\n".join(
            [
                f"Operation {i+1}:\n{str(op)}\n"
                for i, op in enumerate(target_ops_configs)
            ]
        )

        return (
            f"You are an expert at document processing and regular expressions.\n\n"
            f"Target Operations:\n"
            f"{ops_str}\n\n"
            f"Directive: {self.name}\n"
            f"Your task is to instantiate this directive by configuring a regex_compress operation, "
            f"which DocETL inserts before the target operations. It runs no LLM and no Python you write:\n"
            f"1. It deletes every match of each remove_patterns entry from the document_key field.\n"
            f"2. It splits the text into paragraphs at blank lines and drops empty paragraphs.\n"
            f"3. It keeps a paragraph when no drop_patterns entry matches it, and either the paragraph "
            f"starts within the first keep_first_chars characters or a keep_patterns entry matches it.\n"
            f"4. It joins the kept paragraphs with blank lines, with [...] where paragraphs were left out, "
            f"and writes the result back to the same field. When no paragraph is kept, the field keeps "
            f"the whole text from step 1.\n\n"
            f"Output:\n"
            f"- name: a descriptive name for the operation\n"
            f"- document_key: the document field the target operations read (for example 'document' "
            f"for {{{{ input.document }}}})\n"
            f"- keep_patterns: regular expressions for the content ALL target operations need; favor "
            f"recall, since a paragraph left out is lost to the downstream operations\n"
            f"- drop_patterns: regular expressions for paragraphs to always leave out, such as a table of contents\n"
            f"- remove_patterns: regular expressions for noise to delete, such as page numbers or repeated headers\n"
            f"- keep_first_chars: how much of the document's start to always keep (title, parties, dates)\n\n"
            f"Patterns are searched anywhere in a paragraph (use ^ with (?m) to anchor to a line). Use "
            f"syntax both Python's re and Rust's regex accept: inline flags such as (?i) and (?m), "
            f"character classes, \\b, \\s, \\d, alternation, and non-capturing or plain groups. "
            f"Do not use lookahead, lookbehind, named groups, or backreferences.\n\n"
            f"Example:\n"
            f"{self.example}\n\n"
            f"Please output only the DeterministicDocCompressionInstantiateSchema as JSON "
            f"that specifies how to apply this directive to the target operations."
        )

    def llm_instantiate(
        self,
        target_ops_configs: List[Dict],
        agent_llm: str,
        message_history: list = [],
    ):
        """
        Call the LLM to generate the instantiate schema.
        The LLM will output structured data matching DeterministicDocCompressionInstantiateSchema.
        """

        message_history.extend(
            [
                {
                    "role": "user",
                    "content": self.to_string_for_instantiate(target_ops_configs),
                },
            ]
        )

        last_error = None
        for _ in range(MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS):
            resp = completion(
                model=agent_llm,
                messages=message_history,
                response_format=DeterministicDocCompressionInstantiateSchema,
            )
            call_cost = resp._hidden_params.get("response_cost", 0)
            try:
                parsed_res = json.loads(resp.choices[0].message.content)
                schema = DeterministicDocCompressionInstantiateSchema(**parsed_res)

                # Validate against target operations
                schema.validate_against_target_ops(target_ops_configs)

                message_history.append(
                    {"role": "assistant", "content": resp.choices[0].message.content}
                )
                return schema, message_history, call_cost
            except Exception as err:
                last_error = err
                error_message = f"Validation error: {err}\nPlease try again."
                message_history.append({"role": "user", "content": error_message})

        raise Exception(
            f"Failed to instantiate directive after {MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS} attempts. Last error: {last_error}"
        )

    def apply(
        self,
        global_default_model: str,
        ops_list: List[Dict],
        target_ops: List[str],
        rewrite: DeterministicDocCompressionInstantiateSchema,
    ) -> List[Dict]:
        """
        Apply the directive using the instantiate schema configuration.
        Inserts a regex_compress operation before the first target operation.
        """
        new_ops_list = deepcopy(ops_list)

        # Find the position of the first target operation
        first_target_pos = min(
            [i for i, op in enumerate(ops_list) if op["name"] in target_ops]
        )

        compress_op = {
            "name": rewrite.name,
            "type": "regex_compress",
            "document_key": rewrite.document_key,
            "keep_patterns": rewrite.keep_patterns,
            "drop_patterns": rewrite.drop_patterns,
            "remove_patterns": rewrite.remove_patterns,
            "keep_first_chars": rewrite.keep_first_chars,
        }
        new_ops_list.insert(first_target_pos, compress_op)

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
        Main method that orchestrates directive instantiation:
        1. Get agent to generate instantiate schema for all target operations
        2. Apply the transformation using that schema
        """
        assert len(target_ops) >= 1, "This directive requires at least one target op"

        # Get configurations for all target operations
        target_ops_configs = [op for op in operators if op["name"] in target_ops]

        # Step 1: Agent generates the instantiate schema considering all target ops
        rewrite, message_history, call_cost = self.llm_instantiate(
            target_ops_configs, agent_llm, message_history
        )

        # Step 2: Apply transformation using the schema
        return (
            self.apply(global_default_model, operators, target_ops, rewrite),
            message_history,
            call_cost,
        )
