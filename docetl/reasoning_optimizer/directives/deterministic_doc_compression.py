import json
import os
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
    formal_description: str = Field(default="Op => Code Map -> Op")
    nl_description: str = Field(
        default="Reduces LLM processing costs by using deterministic logic (regex, patterns) to compress documents before expensive downstream operations, removing irrelevant content that could distract the LLM"
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
        "name": "extract_compliance_sections",
        "code": '''
def transform(input_doc):
    import re

    legal_document = input_doc.get('legal_document', '')

    # Patterns to identify compliance-related content
    compliance_patterns = [
        r'(?i)(violat[e|ion]|breach|non-complian[ce|t])',
        r'(?i)(deadline|due date|expir[e|ation]|within.*days?)',
        r'(?i)(shall|must|required|mandatory|obligation)',
        r'(?i)(section|article|clause)\\s+\\d+.*(complian[ce|t]|regulat[e|ory])'
    ]

    relevant_spans = []

    # Find all matches and extract context around them
    for pattern in compliance_patterns:
        for match in re.finditer(pattern, legal_document):
            start_pos = match.start()
            end_pos = match.end()

            # Extract 300 chars before and 800 chars after the match
            context_start = max(0, start_pos - 300)
            context_end = min(len(legal_document), end_pos + 800)

            # Extract the context around the match
            context = legal_document[context_start:context_end]
            relevant_spans.append((context_start, context_end, context))

    # Merge overlapping spans and remove duplicates
    if relevant_spans:
        # Sort by start position
        relevant_spans.sort(key=lambda x: x[0])
        merged_spans = [relevant_spans[0]]

        for current_start, current_end, current_text in relevant_spans[1:]:
            last_start, last_end, last_text = merged_spans[-1]

            if current_start <= last_end + 100:  # Merge if close enough
                # Extend the last span
                new_end = max(last_end, current_end)
                new_text = legal_document[last_start:new_end]
                merged_spans[-1] = (last_start, new_end, new_text)
            else:
                merged_spans.append((current_start, current_end, current_text))

        # Extract just the text portions
        compressed_text = '\\n\\n--- SECTION BREAK ---\\n\\n'.join([span[2] for span in merged_spans])
    else:
        compressed_text = legal_document  # Fallback if no matches

    return {
        'legal_document': compressed_text
    }
            '''
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
                expected_behavior="Should add Code Map operation that extracts merger agreement sections using regex patterns for legal terms, financial provisions, and risk-related clauses. The return dictionary of the transform function should be {'merger_agreement': ....} only.",
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
                expected_behavior="Should add Code Map operation that extracts financial content needed for both operations. The return dictionary of the transform function should be {'earnings_report': ....} only.",
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
            f"You are an expert at document processing and Python programming.\n\n"
            f"Target Operations:\n"
            f"{ops_str}\n\n"
            f"Directive: {self.name}\n"
            f"Your task is to instantiate this directive by specifying a Code Map operation "
            f"that specifies how to compress the input document using deterministic logic.\n\n"
            f"The directive will insert a Code Map operation that:\n"
            f"1. Takes document field(s) from the input\n"
            f"2. Uses deterministic logic (regex, keyword matching, pattern extraction) to compress them\n"
            f"3. Returns a dictionary with the EXACT SAME document field key and compressed content\n"
            f"4. Reduces token usage and improves focus for the downstream operations\n\n"
            f"The agent must output the configuration specifying:\n"
            f"- name: A descriptive name for the Code Map operation\n"
            f"- code: Python code defining a 'transform' function that:\n"
            f"  * Takes input_doc as parameter\n"
            f"  * Imports 're' and other standard libraries WITHIN the function itself\n"
            f"  * Only uses standard Python libraries (re, string, json, etc.) - no external packages\n"
            f"  * Uses deterministic logic to extract relevant content patterns\n"
            f"  * For each pattern match, extracts context around it (e.g., ±500 chars, or -300 to +800 chars)\n"
            f"  * Use your judgment to determine appropriate character counts that capture enough context\n"
            f"  * Merges overlapping context spans to avoid duplication\n"
            f"  * Returns a dictionary with the EXACT document key and compressed value: {{document_key: compressed_content}}\n\n"
            f"CRITICAL: The returned dictionary must use the exact same document field names as the original, "
            f"not modified versions like 'document_key_compressed'. The downstream operations expect the exact same field names.\n\n"
            f"IMPORTANT: Focus on extracting the minimal content necessary for ALL target operations "
            f"using deterministic pattern matching. Analyze each operation's prompt to identify what types "
            f"of content patterns to look for.\n\n"
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

        for _ in range(MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS):
            resp = completion(
                model=agent_llm,
                messages=message_history,
                api_key=os.environ.get("AZURE_API_KEY"),
                api_base=os.environ.get("AZURE_API_BASE"),
                api_version=os.environ.get("AZURE_API_VERSION"),
                azure=True,
                response_format=DeterministicDocCompressionInstantiateSchema,
            )
            call_cost = resp._hidden_params["response_cost"]
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
                error_message = f"Validation error: {err}\nPlease try again."
                message_history.append({"role": "user", "content": error_message})

        raise Exception(
            f"Failed to instantiate directive after {MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS} attempts."
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
        Inserts a Code Map operation before the first target operation.
        """
        new_ops_list = deepcopy(ops_list)

        # Find the position of the first target operation
        first_target_pos = min(
            [i for i, op in enumerate(ops_list) if op["name"] in target_ops]
        )

        # Create the Code Map operation
        code_map_op = {
            "name": rewrite.name,
            "type": "code_map",
            "code": rewrite.code,
        }

        # Insert the Code Map operation before the first target operation
        new_ops_list.insert(first_target_pos, code_map_op)

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
