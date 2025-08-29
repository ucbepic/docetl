import json
import os
from copy import deepcopy
from typing import Dict, List, Type

from litellm import completion
from pydantic import BaseModel, Field

from docetl.reasoning_optimizer.instantiate_schemas import (
    DocumentChunkingTopKInstantiateSchema,
)

from .base import MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS, Directive, DirectiveTestCase


class DocumentChunkingTopKDirective(Directive):
    name: str = Field(
        default="doc_chunking_topk", description="The name of the directive"
    )
    formal_description: str = Field(
        default="Map/Filter => Split -> TopK -> Reduce (-> Code Filter if original was Filter)"
    )
    nl_description: str = Field(
        default="Cost optimization directive for documents where only certain portions are relevant to the task (when at least half the document is irrelevant). Works with both Map and Filter operations. Transforms into a retrieval-augmented pipeline: splits documents into chunks, uses topk to retrieve the most relevant chunks, processes them in a reduce operation. For Filter operations, adds a final code_filter step to return boolean results. Ideal when processing full documents would be wasteful due to irrelevant content."
    )
    when_to_use: str = Field(
        default="Use when only certain portions of documents are relevant to the task and at least half of the document content is irrelevant. Perfect for complex filters (e.g., 'does this review mention competitor products more favorably?') or targeted extraction from documents with localized relevant sections. Works with both Map (extraction) and Filter (boolean decision) operations. The retrieval step (embedding or FTS) finds the relevant chunks, avoiding processing irrelevant content. For filters, the final code_filter converts the reduce output to True/False."
    )

    instantiate_schema_type: Type[BaseModel] = Field(
        default=DocumentChunkingTopKInstantiateSchema
    )

    example: str = Field(
        default="""
            # Example 1: Complex Filter on Long Customer Reviews
            Original Op:
            - name: filter_competitor_mentions
              type: filter
              prompt: |
                Analyze this customer review to determine if it mentions competitor products
                more positively than our product.

                Our Product: {{ input.our_product }}
                Review: {{ input.review_text }}
                Review ID: {{ input.review_id }}

                Return true if the review speaks more favorably about competitor products than ours.
                Consider: feature comparisons, performance mentions, value assessments, recommendations.
              output:
                schema:
                  mentions_competitors_more_positively: bool

            InstantiateSchema (filter with embedding search):
            {
              "chunk_size": 10000,
              "split_key": "review_text",
              "reduce_prompt": "Analyze this customer review to determine if it mentions competitor products more positively than our product.\\n\\nOur Product: {{ inputs[0].our_product }}\\nReview: the top {{ inputs|length }} most relevant chunks from the document (ordered by relevance):\\n{% for input in inputs|sort(attribute='_topk_filter_competitor_mentions_chunks_rank') %}\\nChunk (Rank {{ input._topk_filter_competitor_mentions_chunks_rank }}, Score {{ input._topk_filter_competitor_mentions_chunks_score }}):\\n{{ input.review_text_chunk }}\\n{% endfor %}\\nReview ID: {{ inputs[0].review_id }}\\n\\nReturn true if the review speaks more favorably about competitor products than ours.\\nConsider: feature comparisons, performance mentions, value assessments, recommendations.",
              "topk_config": {
                "method": "embedding",
                "k": 5,
                "query": "competitor comparison versus alternative better than superior inferior worse features performance value recommendation prefer instead",
                "keys": ["review_text_chunk"],
                "embedding_model": "text-embedding-3-small"
              }
            }

            # Example 2: Map Operation - Extract Specific Sections from Long Documents
            Original Op:
            - name: extract_methodology_from_paper
              type: map
              prompt: |
                Extract detailed methodology from this research paper:

                Paper: {{ input.paper_content }}
                Title: {{ input.title }}

                Extract: study design, sample size, data collection methods,
                statistical analyses, and validation approaches.
              output:
                schema:
                  study_design: str
                  sample_size: dict
                  data_collection: list[str]
                  statistical_methods: list[str]
                  validation: str

            InstantiateSchema (map with embedding search):
            {
              "chunk_size": 15000,
              "split_key": "paper_content",
              "reduce_prompt": "Extract detailed methodology from this research paper:\\n\\nPaper: the top {{ inputs|length }} most relevant chunks from the document (ordered by relevance):\\n{% for input in inputs|sort(attribute='_topk_extract_methodology_from_paper_chunks_rank') %}\\nChunk (Rank {{ input._topk_extract_methodology_from_paper_chunks_rank }}, Score {{ input._topk_extract_methodology_from_paper_chunks_score }}):\\n{{ input.paper_content_chunk }}\\n{% endfor %}\\nTitle: {{ inputs[0].title }}\\n\\nExtract: study design, sample size, data collection methods, statistical analyses, and validation approaches.",
              "topk_config": {
                "method": "embedding",
                "k": 8,
                "query": "methodology methods study design sample size participants data collection statistical analysis validation procedure protocol experimental",
                "keys": ["paper_content_chunk"],
                "embedding_model": "text-embedding-3-small"
              }
            }

            # Example 3: Filter with FTS - Check Contract Compliance
            Original Op:
            - name: filter_contracts_with_liability_caps
              type: filter
              prompt: |
                Determine if this contract contains liability cap provisions
                that limit damages to less than $1 million.

                Contract: {{ input.contract_text }}
                Contract ID: {{ input.contract_id }}
                Party: {{ input.counterparty }}

                Return true if contract caps liability below $1M, false otherwise.
              output:
                schema:
                  has_low_liability_cap: bool

            InstantiateSchema (filter with FTS for legal terms):
            {
              "chunk_size": 12000,
              "split_key": "contract_text",
              "reduce_prompt": "Determine if this contract contains liability cap provisions that limit damages to less than $1 million.\\n\\nContract: the top {{ inputs|length }} most relevant chunks from the document (ordered by relevance):\\n{% for input in inputs|sort(attribute='_topk_filter_contracts_with_liability_caps_chunks_rank') %}\\nSection (Rank {{ input._topk_filter_contracts_with_liability_caps_chunks_rank }}, Score {{ input._topk_filter_contracts_with_liability_caps_chunks_score }}):\\n{{ input.contract_text_chunk }}\\n{% endfor %}\\nContract ID: {{ inputs[0].contract_id }}\\nParty: {{ inputs[0].counterparty }}\\n\\nReturn true if contract caps liability below $1M, false otherwise.",
              "topk_config": {
                "method": "fts",
                "k": 10,
                "query": "liability limitation cap maximum damages indirect consequential million dollars aggregate total exposure indemnification",
                "keys": ["contract_text_chunk"]
              }
            }
        """,
    )

    test_cases: List[DirectiveTestCase] = Field(
        default_factory=lambda: [
            DirectiveTestCase(
                name="clinical_trial_adverse_events_extraction",
                description="Should transform clinical trial safety analysis into chunking pipeline with embedding-based topk for thematic content",
                input_config={
                    "name": "extract_clinical_trial_safety",
                    "type": "map",
                    "prompt": """Analyze this clinical trial protocol and safety report to extract comprehensive safety information:

                    Protocol Number: {{ input.protocol_id }}
                    Study Phase: {{ input.study_phase }}
                    Document: {{ input.trial_document }}

                    Extract and analyze:
                    1. ALL adverse events (AEs) with:
                       - Event description and medical terminology (MedDRA preferred terms)
                       - Severity grade (1-5 per CTCAE v5.0)
                       - Relationship to study drug (definitely, probably, possibly, unlikely, not related)
                       - Onset timing relative to treatment start
                       - Resolution status and duration
                       - Actions taken (dose reduced, interrupted, discontinued)

                    2. Serious adverse events (SAEs) with additional details:
                       - Hospitalization requirements
                       - Life-threatening classification
                       - Death outcomes with causality assessment
                       - Expedited reporting timeline compliance

                    3. Laboratory abnormalities:
                       - Clinically significant lab value shifts
                       - Grade 3/4 laboratory toxicities
                       - Liver function test elevations (ALT, AST, bilirubin)
                       - Renal function changes (creatinine, eGFR)
                       - Hematologic abnormalities

                    4. Dose-limiting toxicities (DLTs) and maximum tolerated dose (MTD) determination

                    5. Safety run-in period results if applicable

                    6. Data safety monitoring board (DSMB) recommendations and protocol modifications

                    Ensure all safety data is captured with appropriate medical coding and regulatory compliance.""",
                    "output": {
                        "schema": {
                            "adverse_events": "list[dict]",
                            "serious_adverse_events": "list[dict]",
                            "lab_abnormalities": "list[dict]",
                            "dose_limiting_toxicities": "list[dict]",
                            "dsmb_recommendations": "list[str]",
                            "safety_summary": "dict",
                        }
                    },
                },
                target_ops=["extract_clinical_trial_safety"],
                expected_behavior="Should create chunking pipeline with topk using embedding search to find sections discussing adverse events, safety data, laboratory results, and DSMB recommendations. Chunks should be 5-8k tokens with k=10-15 to capture all safety-related sections",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="sec_filing_risk_factors_extraction",
                description="Should transform SEC filing analysis into chunking pipeline with FTS-based topk for specific regulatory terms",
                input_config={
                    "name": "extract_sec_risk_disclosures",
                    "type": "map",
                    "prompt": """Extract and analyze all risk factor disclosures from this SEC 10-K filing:

                    Company: {{ input.company_ticker }}
                    Filing Period: {{ input.filing_period }}
                    Document: {{ input.form_10k }}
                    Industry: {{ input.industry_classification }}

                    Identify and categorize:

                    1. BUSINESS AND OPERATIONAL RISKS:
                       - Supply chain vulnerabilities and dependencies
                       - Key customer concentration (customers >10% revenue)
                       - Competition and market share threats
                       - Product obsolescence and innovation risks
                       - Manufacturing and quality control risks
                       - Intellectual property disputes and patent expirations

                    2. FINANCIAL AND MARKET RISKS:
                       - Liquidity and cash flow concerns
                       - Debt covenants and refinancing risks
                       - Foreign exchange exposure by currency
                       - Interest rate sensitivity analysis
                       - Credit risk and counterparty exposure
                       - Goodwill and intangible asset impairment risks

                    3. REGULATORY AND COMPLIANCE RISKS:
                       - SEC investigation disclosures
                       - FDA/regulatory approval dependencies
                       - Environmental liabilities and remediation costs
                       - Tax disputes and uncertain tax positions
                       - FCPA and anti-corruption compliance
                       - Data privacy (GDPR, CCPA) obligations

                    4. CYBERSECURITY AND TECHNOLOGY RISKS:
                       - Data breach history and potential impacts
                       - IT system dependencies and modernization needs
                       - Third-party technology provider risks
                       - Business continuity and disaster recovery

                    5. LITIGATION AND LEGAL RISKS:
                       - Material pending litigation with potential damages
                       - Class action lawsuit exposure
                       - Warranty and product liability claims
                       - Employment and labor disputes

                    6. ESG AND REPUTATIONAL RISKS:
                       - Climate change physical and transition risks
                       - Social license to operate concerns
                       - Executive succession planning
                       - Related party transaction risks

                    For each risk, extract:
                    - Risk description and specific company exposure
                    - Quantitative impact estimates if disclosed
                    - Mitigation strategies mentioned
                    - Changes from prior year disclosure
                    - Forward-looking statements and warnings""",
                    "output": {
                        "schema": {
                            "business_operational_risks": "list[dict]",
                            "financial_market_risks": "list[dict]",
                            "regulatory_compliance_risks": "list[dict]",
                            "cybersecurity_technology_risks": "list[dict]",
                            "litigation_legal_risks": "list[dict]",
                            "esg_reputational_risks": "list[dict]",
                            "risk_factor_changes": "list[dict]",
                            "material_risk_summary": "dict",
                        }
                    },
                },
                target_ops=["extract_sec_risk_disclosures"],
                expected_behavior="Should create chunking pipeline with topk using FTS to search for specific risk-related keywords and sections (Item 1A, risk factors, legal proceedings, etc.). Chunks should be 6-10k tokens with k=15-20 to ensure comprehensive risk coverage",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="insurance_claim_analysis_with_dynamic_query",
                description="Should transform insurance claim analysis with Jinja template query based on claim type",
                input_config={
                    "name": "analyze_insurance_claim",
                    "type": "map",
                    "prompt": """Analyze this insurance claim file for coverage determination and fraud indicators:

                    Claim Number: {{ input.claim_id }}
                    Policy Number: {{ input.policy_number }}
                    Claim Type: {{ input.claim_type }}
                    Claimed Amount: {{ input.claimed_amount }}
                    Policy Documents: {{ input.policy_documents }}
                    Claim Documents: {{ input.claim_submission }}
                    Prior Claims History: {{ input.claims_history }}

                    Perform comprehensive analysis:

                    1. COVERAGE DETERMINATION:
                       - Verify incident date falls within policy period
                       - Check specific peril coverage for {{ input.claim_type }} claims
                       - Identify applicable policy limits and sublimits
                       - Calculate deductibles and co-insurance
                       - Review exclusions that may apply
                       - Assess pre-existing condition clauses (if medical)
                       - Verify additional living expense limits (if property)

                    2. CLAIM VALIDATION:
                       - Cross-reference damage description with photos/evidence
                       - Verify repair estimates against market rates
                       - Validate medical treatment necessity and coding
                       - Check for duplicate submissions or double-dipping
                       - Verify loss circumstances match policy terms

                    3. FRAUD INDICATORS ASSESSMENT:
                       - Pattern analysis against known fraud schemes
                       - Inconsistencies in statements or documentation
                       - Suspicious timing (policy inception, premium issues)
                       - Inflated valuations or treatment costs
                       - Missing or altered documentation
                       - Prior suspicious claims pattern

                    4. THIRD-PARTY LIABILITY:
                       - Subrogation opportunities
                       - Other insurance coverage available
                       - Responsible party identification
                       - Coordination of benefits requirements

                    5. REGULATORY COMPLIANCE:
                       - State-specific claim handling requirements
                       - Unfair claim settlement practices act compliance
                       - Required notices and timelines
                       - Bad faith claim indicators

                    6. SETTLEMENT RECOMMENDATION:
                       - Covered amount calculation
                       - Recommended settlement range
                       - Payment breakdown by category
                       - Reserve recommendations
                       - Special investigation unit (SIU) referral if warranted""",
                    "output": {
                        "schema": {
                            "coverage_analysis": "dict",
                            "claim_validation": "dict",
                            "fraud_indicators": "list[dict]",
                            "third_party_liability": "dict",
                            "regulatory_compliance": "dict",
                            "settlement_recommendation": "dict",
                            "siu_referral": "bool",
                            "reserve_amount": "float",
                        }
                    },
                },
                target_ops=["analyze_insurance_claim"],
                expected_behavior="Should create chunking pipeline with topk using dynamic Jinja query that incorporates claim_type to search for relevant policy sections and prior claims. Query should adapt based on whether it's property, auto, medical, or liability claim. Chunks should be 5-7k tokens with k=12-18",
                should_pass=True,
            ),
        ]
    )

    def __eq__(self, other):
        return isinstance(other, DocumentChunkingTopKDirective)

    def __hash__(self):
        return hash("DocumentChunkingTopKDirective")

    def to_string_for_instantiate(self, original_op: Dict) -> str:
        """
        Generate a prompt for an agent to instantiate this directive.

        Args:
            original_op (Dict): The original operation.

        Returns:
            str: The agent prompt for instantiating the directive.
        """
        op_type = original_op.get("type", "map")
        return (
            f"You are an expert at transforming document processing operations into chunking pipelines with intelligent topk-based retrieval.\n\n"
            f"Original Operation:\n"
            f"{str(original_op)}\n\n"
            f"Directive: {self.name}\n"
            f"Your task is to instantiate this directive by creating a configuration that transforms the original {op_type.capitalize()} operation "
            f"into a Split -> TopK -> Reduce pipeline for processing very long documents with intelligent chunk selection.\n"
            f"{'For Filter operations, a final code_filter step will be automatically added to return boolean results.' if op_type == 'filter' else ''}\n\n"
            f"Key requirements:\n"
            f"1. chunk_size: Choose an appropriate token count (typically 10000-15000) for cost-effective processing of long documents\n"
            f"2. split_key: Identify the document field to split from the original operation's prompt (the longest text field)\n"
            f"3. reduce_prompt: Use the EXACT SAME prompt as the original, with ONE change:\n"
            f"   - Where the original references '{{{{ input.<split_key> }}}}', replace it with:\n"
            f"     'the top {{{{ inputs|length }}}} most relevant chunks from the document (ordered by relevance):\\n{{% for input in inputs|sort(attribute='_<topk_name>_rank') %}}\\nChunk (Rank {{{{ input._<topk_name>_rank }}}}, Score {{{{ input._<topk_name>_score }}}}):\\n{{{{ input.<split_key>_chunk }}}}\\n{{% endfor %}}'\n"
            f"   - Keep EVERYTHING else identical - same instructions, same output requirements\n"
            f"   - For other context fields (non-document fields), use {{{{ inputs[0].field_name }}}} instead of {{{{ input.field_name }}}}\n"
            f"4. topk_config: REQUIRED - Configure intelligent chunk selection:\n"
            f"   - method: Choose 'embedding' for semantic similarity or 'fts' for keyword matching\n"
            f"     * Use 'embedding' when: looking for conceptual comparisons, themes, or abstract relationships\n"
            f"     * Use 'fts' when: searching for specific terms, legal clauses, technical codes, or exact phrases\n"
            f"   - k: Number of chunks to retrieve (typically 5-10 for comprehensive coverage)\n"
            f"     * For complex tasks needing most of the document as context: k=10\n"
            f"     * For targeted extraction from specific sections: k=5\n"
            f"   - query: Craft carefully to find chunks relevant to the {'filter decision' if op_type == 'filter' else 'extraction task'}\n"
            f"     * For embedding: use terms related to the comparison/decision criteria\n"
            f"     * For fts: use specific keywords that appear in relevant sections\n"
            f"     * Can use Jinja: '{{{{ input.competitor_name }}}} comparison versus {{{{ input.our_product }}}}'\n"
            f"   - keys: Always use the chunk key, typically ['<split_key>_chunk']\n"
            f"   - embedding_model: (optional, only for embedding method) defaults to 'text-embedding-3-small'\n"
            f"The topk query should be carefully crafted to find the most relevant chunks.\n"
            f"The reduce_prompt must process chunks directly and {'output a boolean decision' if op_type == 'filter' else 'preserve the original output schema'}.\n\n"
            f"Example:\n"
            f"{self.example}\n\n"
            f"Please output only the DocumentChunkingTopKInstantiateSchema object as JSON."
        )

    def llm_instantiate(
        self,
        operators,
        input_file_path,
        original_op: Dict,
        agent_llm: str,
        message_history: list = [],
    ):
        """
        Use LLM to instantiate this directive by creating chunking configuration with topk.

        Args:
            original_op (Dict): The original operation.
            agent_llm (str): The LLM model to use.
            message_history (List, optional): Conversation history for context.

        Returns:
            DocumentChunkingTopKInstantiateSchema: The structured output from the LLM.
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
                response_format=DocumentChunkingTopKInstantiateSchema,
            )
            call_cost = resp._hidden_params["response_cost"]
            try:
                parsed_res = json.loads(resp.choices[0].message.content)
                schema = DocumentChunkingTopKInstantiateSchema(**parsed_res)
                schema.validate_split_key_exists_in_input(input_file_path)
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
        rewrite: DocumentChunkingTopKInstantiateSchema,
    ) -> List[Dict]:
        """
        Apply the directive to the pipeline config by replacing the target operation
        with a split -> topk -> reduce sequence.
        """

        # Create a copy of the pipeline config
        new_ops_list = deepcopy(ops_list)
        target_op_config = [op for op in new_ops_list if op["name"] == target_op][0]
        original_model = target_op_config.get("model", global_default_model)

        # Find position of the target op to replace
        pos_to_replace = [
            i for i, op in enumerate(ops_list) if op["name"] == target_op
        ][0]

        original_op = ops_list[pos_to_replace]

        # Create operation names based on the original operation name
        split_name = f"split_{target_op}"
        topk_name = f"topk_{target_op}_chunks"
        reduce_name = f"reduce_{target_op}"

        # Create the split operation
        split_op = {
            "name": split_name,
            "type": "split",
            "split_key": rewrite.split_key,
            "method": "token_count",
            "method_kwargs": {
                "num_tokens": rewrite.chunk_size,
                "model": original_model,
            },
        }

        # Create the topk operation
        topk_op = {
            "name": topk_name,
            "type": "topk",
            "method": rewrite.topk_config.method,
            "k": rewrite.topk_config.k,
            "keys": rewrite.topk_config.keys,
            "query": rewrite.topk_config.query,
            "stratify_key": [f"{split_name}_id"],
        }

        # Add stratify_key if specified
        if rewrite.topk_config.stratify_key:
            topk_op["stratify_key"] = topk_op["stratify_key"] + [
                rewrite.topk_config.stratify_key
            ]

        # Add embedding_model for embedding method
        if (
            rewrite.topk_config.method == "embedding"
            and rewrite.topk_config.embedding_model
        ):
            topk_op["embedding_model"] = rewrite.topk_config.embedding_model

        # Check if original operation is a filter
        is_filter = original_op.get("type") == "filter"

        # Create the reduce operation that directly processes the chunks
        if is_filter:
            # For filter operations, reduce should output a boolean field
            reduce_output = deepcopy(original_op["output"])
            # Ensure we have a boolean field in the output schema
            if "schema" in reduce_output:
                # Get the first boolean field name from the schema
                bool_field = None
                for field_name, field_type in reduce_output["schema"].items():
                    if "bool" in field_type.lower():
                        bool_field = field_name
                        break
                if not bool_field:
                    # If no boolean field found, create one
                    bool_field = "filter_result"
                    reduce_output["schema"] = {bool_field: "bool"}
        else:
            reduce_output = deepcopy(original_op["output"])

        reduce_op = {
            "name": reduce_name,
            "type": "reduce",
            "reduce_key": f"{split_name}_id",
            "prompt": rewrite.reduce_prompt,
            "model": original_model,
            "litellm_completion_kwargs": {"temperature": 0},
            "output": reduce_output,
            "associative": False,  # Order matters for chunks
            "pass_through": True,
        }

        # Construct operation sequence
        ops_sequence = [split_op, topk_op, reduce_op]

        # If original was a filter, add a code_filter operation
        if is_filter:
            # Find the boolean field name in the output schema
            bool_field = None
            if "schema" in reduce_output:
                for field_name, field_type in reduce_output["schema"].items():
                    if "bool" in field_type.lower():
                        bool_field = field_name
                        break

            if bool_field:
                code_filter_op = {
                    "name": f"code_filter_{target_op}",
                    "type": "code_filter",
                    "code": f"def transform(input_doc):\n    return input_doc.get('{bool_field}', False)",
                }
                ops_sequence.append(code_filter_op)

        # Replace the target operation with the new sequence
        new_ops_list[pos_to_replace : pos_to_replace + 1] = ops_sequence

        return new_ops_list

    def instantiate(
        self,
        operators: List[Dict],
        target_ops: List[str],
        agent_llm: str,
        message_history: list = [],
        global_default_model: str = None,
        **kwargs,
    ) -> tuple:
        """
        Instantiate the directive for a list of operators.
        """
        # Assert that there is only one target op
        assert (
            len(target_ops) == 1
        ), "There must be exactly one target op to instantiate this document chunking topk directive"
        target_op_config = [op for op in operators if op["name"] == target_ops[0]][0]

        input_file_path = kwargs.get("input_file_path", None)

        # Validate that the target operation is a map or filter operation
        if target_op_config.get("type") not in ["map", "filter"]:
            raise ValueError(
                f"Document chunking topk directive can only be applied to map or filter operations, but target operation '{target_ops[0]}' is of type '{target_op_config.get('type')}'"
            )

        # Instantiate the directive
        rewrite, message_history, call_cost = self.llm_instantiate(
            operators, input_file_path, target_op_config, agent_llm, message_history
        )

        # Apply the rewrite to the operators
        return (
            self.apply(global_default_model, operators, target_ops[0], rewrite),
            message_history,
            call_cost,
        )
