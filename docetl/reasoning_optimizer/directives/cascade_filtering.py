import json
import re
from copy import deepcopy
from typing import Dict, List, Type

from pydantic import BaseModel, Field

from docetl.reasoning_optimizer.instantiate_schemas import (
    CascadeFilteringInstantiateSchema,
)

from .agent_utils import AgenticDirectiveRunner
from .base import Directive, DirectiveTestCase


class CascadeFilteringDirective(Directive):
    name: str = Field(
        default="cascade_filtering", description="The name of the directive"
    )
    formal_description: str = Field(
        default="Filter => (Code Filter* -> Filter(gpt-5-nano)* ->) Filter"
    )
    nl_description: str = Field(
        default="Optimizes filtering costs by injecting a cascade of cheaper filters before the main filter. Starts with deterministic code filters (cheapest), then gpt-5-nano filters (ordered by prompt length), before the original expensive filter. Pre-filters prioritize high recall (rarely rejecting valid documents) and can have lower precision (okay to let through some invalid docs that the main filter will catch)."
    )
    when_to_use: str = Field(
        default="When you have an expensive Filter operation (using costly models or complex prompts) and the data contains patterns that allow for cheaper pre-filtering. The pre-filters must have high recall (not dropping valid documents) but can have lower precision, as the final filter provides the actual precision."
    )

    instantiate_schema_type: Type[BaseModel] = Field(
        default=CascadeFilteringInstantiateSchema
    )

    example: str = Field(
        default="""
        # Example 1: Disjunction - Legal Document Relevance Filter
        Target Operation:
        - name: filter_litigation_relevant_docs
          type: filter
          model: gpt-4o
          prompt: |
            Determine if this document is relevant to our patent litigation case.

            The document is relevant if it contains ANY of:
            1. Prior art references dated before 2015-03-15 describing similar technology
            2. Internal emails discussing the patent claims or invalidity arguments
            3. Technical specifications that contradict our patent claims
            4. License agreements mentioning the patents-in-suit
            5. Expert testimony or declarations about the technology
            6. Financial damages calculations or royalty discussions

            Patent numbers in suit: 8,123,456 and 9,234,567
            Technology area: adaptive bitrate streaming for video delivery

            Document: {{ input.document_text }}
            Metadata: {{ input.document_metadata }}
          output:
            schema:
              is_relevant: boolean

        Example InstantiateSchema (agent recognizes the OR conditions can be split):
        CascadeFilteringInstantiateSchema(
            code_pre_filters=[
                CodePreFilter(
                    name="has_patent_numbers",
                    code=\"\"\"def transform(input_doc):
    text = (input_doc.get('document_text', '') + ' ' + str(input_doc.get('document_metadata', {}))).lower()
    # If patent numbers are mentioned, likely relevant
    return '8,123,456' in text or '9,234,567' in text or 'patent' in text\"\"\",
                    reasoning="Documents mentioning our patent numbers or patents in general might be relevant"
                ),
                CodePreFilter(
                    name="has_tech_or_legal_terms",
                    code=\"\"\"def transform(input_doc):
    text = input_doc.get('document_text', '').lower()
    # Must have streaming tech or legal terms to possibly be relevant
    terms = ['streaming', 'bitrate', 'video', 'codec', 'prior art', 'license',
             'damages', 'royalty', 'testimony', 'expert', 'invalidity']
    return any(term in text for term in terms)\"\"\",
                    reasoning="Documents without technical or legal terminology cannot be relevant"
                )
            ],
            llm_pre_filters=[
                LLMPreFilter(
                    name="check_prior_art_date",
                    prompt="Does this mention technology from before March 2015? {{ input.document_metadata }} If yes; set 'keep' to true. If no; set 'keep' to false.",
                    reasoning="Prior art must predate the patent filing"
                ),
                LLMPreFilter(
                    name="check_email_discussion",
                    prompt="Is this an email or discussion about patents? {{ input.document_text }} If yes; set 'keep' to true. If no; set 'keep' to false.",
                    reasoning="Internal emails about patents might be relevant"
                ),
                LLMPreFilter(
                    name="check_technical_specs",
                    prompt="Does this contain technical specifications? {{ input.document_text }} If yes; set 'keep' to true. If no; set 'keep' to false.",
                    reasoning="Technical specs might contradict our claims"
                )
            ],
            analysis_summary="Filter has 6 OR criteria. Pre-filters check each criterion cheaply, eliminating 70% of documents before expensive analysis"
        )

        # Example 2: Complex Reasoning - Misinformation Detection Filter
        Target Operation:
        - name: filter_misinformation
          type: filter
          model: gpt-4o
          prompt: |
            Analyze this social media post to determine if it contains health misinformation.

            Consider:
            - Claims that contradict scientific consensus
            - Misleading statistics or cherry-picked data
            - Conspiracy theories about health organizations
            - Dangerous medical advice without proper qualifications
            - Manipulation of legitimate studies to support false conclusions
            - Emotional manipulation to spread fear about vaccines/treatments

            Requires nuanced understanding of:
            - Context and speaker credibility
            - Difference between opinion and factual claims
            - Satirical vs serious content
            - Cultural/religious beliefs vs dangerous misinformation

            Post: {{ input.post_content }}
            Author Profile: {{ input.author_info }}
            Engagement Metrics: {{ input.engagement_stats }}
          output:
            schema:
              is_misinformation: boolean

        Example InstantiateSchema (agent deduces proxy filters for complex reasoning):
        CascadeFilteringInstantiateSchema(
            code_pre_filters=[
                CodePreFilter(
                    name="has_health_content",
                    code=\"\"\"def transform(input_doc):
    text = input_doc.get('post_content', '').lower()
    # Must mention health/medical topics to potentially be health misinfo
    health_terms = ['vaccine', 'covid', 'cancer', 'cure', 'treatment', 'doctor',
                    'medical', 'health', 'disease', 'virus', 'immune', 'pharmaceutical']
    return any(term in text for term in health_terms)\"\"\",
                    reasoning="Posts without health-related content cannot be health misinformation"
                ),
                CodePreFilter(
                    name="has_claims_language",
                    code=\"\"\"def transform(input_doc):
    text = input_doc.get('post_content', '').lower()
    # Look for language that makes claims rather than just sharing experience
    claim_markers = ['proven', 'studies show', 'research', 'scientists', 'they don\\'t want',
                     'truth about', 'actually', 'fact', 'evidence', 'causes', 'prevents']
    return any(marker in text for marker in claim_markers) or '!' in text\"\"\",
                    reasoning="Posts without claim-making language are less likely to be misinformation"
                )
            ],
            llm_pre_filters=[
                LLMPreFilter(
                    name="check_medical_claims",
                    prompt="Does this post make medical or health claims? {{ input.post_content }} If yes; set 'keep' to true. If no; set 'keep' to false.",
                    reasoning="Posts making medical claims need detailed analysis"
                ),
                LLMPreFilter(
                    name="check_high_engagement",
                    prompt="Is this a high-engagement post (>1000 shares or viral)? {{ input.engagement_stats }} If yes; set 'keep' to true. If no; set 'keep' to false.",
                    reasoning="High-engagement posts have more potential for harm if they contain misinformation"
                )
            ],
            analysis_summary="Complex task requiring reasoning about credibility, context, and scientific accuracy. Pre-filters identify posts that definitely need review (30% of total) by checking for health content, claim-making language, and viral spread potential"
        )
        """
    )

    test_cases: List[DirectiveTestCase] = Field(
        default_factory=lambda: [
            DirectiveTestCase(
                name="single_filter_cascade",
                description="Should create cascade of filters before expensive filter",
                input_config={
                    "name": "filter_quality",
                    "type": "filter",
                    "model": "gpt-4o",
                    "prompt": "Is this a high-quality research paper? Paper: {{ input.paper }}",
                    "output": {"schema": {"is_quality": "boolean"}},
                },
                target_ops=["filter_quality"],
                expected_behavior="Should inject cheaper pre-filters (code and/or gpt-5-nano) before the expensive gpt-4o filter",
                should_pass=True,
            )
        ]
    )

    def __eq__(self, other):
        return isinstance(other, CascadeFilteringDirective)

    def __hash__(self):
        return hash("CascadeFilteringDirective")

    def _extract_input_keys(self, prompt: str) -> List[str]:
        """Extract input field names from a Jinja2 prompt template."""
        # Match {{ input.fieldname }} patterns
        pattern = r"\{\{\s*input\.(\w+)\s*\}\}"
        matches = re.findall(pattern, prompt)
        return list(set(matches))  # Return unique field names

    def to_string_for_instantiate(
        self, target_ops_configs: List[Dict], pipeline_code: Dict = None
    ) -> str:
        """Generate prompt for agent to analyze data and create cascade filters."""
        assert (
            len(target_ops_configs) == 1
        ), "CascadeFiltering directive only supports single target operation"
        assert (
            target_ops_configs[0]["type"] == "filter"
        ), "Target operation must be a filter"

        op = target_ops_configs[0]
        input_keys = self._extract_input_keys(op.get("prompt", ""))

        pipeline_context = ""
        if pipeline_code:
            pipeline_context = f"""
Pipeline Context:
{json.dumps(pipeline_code, indent=2)}

The target filter '{op['name']}' fits into this broader pipeline. Consider what types of documents flow into this filter and what the downstream operations expect.
"""

        return (
            f"You are an expert at optimizing filter operations by creating efficient filter cascades.\n\n"
            f"Target Filter Operation:\n"
            f"{json.dumps(op, indent=2)}\n\n"
            f"Input fields used in the original prompt: {input_keys}\n\n"
            f"{pipeline_context}\n"
            f"Your task is to analyze sample input data and create a cascade of cheaper pre-filters that can eliminate many documents before they reach the expensive main filter.\n\n"
            f"You will be given access to sample input data through a read_next_docs() function. Use this to:\n"
            f"1. Understand what documents should pass vs. fail the main filter\n"
            f"2. Identify simple patterns that can predict filter outcomes\n"
            f"3. Design code-based filters for deterministic patterns (cheapest)\n"
            f"4. Design gpt-5-nano filters for simple semantic patterns (still cheap)\n\n"
            f"Guidelines for code_pre_filters:\n"
            f"- Must be Python functions with signature: def transform(input_doc): return bool\n"
            f"- Should use regex, keyword matching, length checks, or simple logic\n"
            f"- Must be deterministic and fast\n"
            f"- Return True to keep the document, False to filter it out\n"
            f"- Access document fields using input_doc.get('fieldname', default_value)\n\n"
            f"Guidelines for llm_pre_filters:\n"
            f"- Must be Jinja2 templates that reference input fields using {{{{ input.fieldname }}}} syntax\n"
            f"- Must reference the same input fields as the original prompt\n"
            f"- Should be simple, focused prompts that elicit a yes/no or true/false response\n"
            f"- Keep prompts SHORT - they will be ordered by length automatically\n"
            f"- Must use at least one of these input fields: {input_keys}\n\n"
            f"General Guidelines:\n"
            f"- Pre-filters MUST have HIGH RECALL (rarely reject documents that would pass the main filter)\n"
            f"- Pre-filters can have LOWER PRECISION (okay to let through documents that fail - the main filter will catch them)\n"
            f"- When in doubt, let the document through (return True) - false negatives are worse than false positives\n\n"
            f"- All documents will have the same keys. So don't write code that checks if a particular key exists in a document or not.\n"
            f"Example transformation:\n"
            f"{self.example}\n\n"
            f"Analyze samples strategically - look for patterns that distinguish documents that pass vs fail the filter.\n"
            f"When you have identified enough patterns to create effective pre-filters, output your result.\n\n"
            f"Remember: The goal is to reduce costs by filtering out documents early with cheaper methods while maintaining the same final accuracy."
        )

    def llm_instantiate(
        self,
        target_ops_configs: List[Dict],
        input_file_path: str,
        agent_llm: str,
        message_history: list = [],
        pipeline_code: Dict = None,
    ):
        """Use agentic approach to analyze data and generate cascade filters."""
        # Load sample input data
        try:
            with open(input_file_path, "r") as f:
                input_data = json.load(f)

            if not isinstance(input_data, list) or len(input_data) == 0:
                raise ValueError(
                    "Input file must contain a non-empty list of sample data"
                )

        except Exception as e:
            raise Exception(
                f"Failed to load input data from {input_file_path}: {str(e)}"
            )

        # Extract input keys from original prompt for validation
        original_prompt = target_ops_configs[0].get("prompt", "")
        expected_input_keys = self._extract_input_keys(original_prompt)

        def validate_llm_prompts(schema_instance):
            """Validate that LLM prompts use correct input fields."""
            for llm_filter in schema_instance.llm_pre_filters:
                used_keys = set(
                    re.findall(r"\{\{\s*input\.(\w+)\s*\}\}", llm_filter.prompt)
                )
                invalid_keys = used_keys - set(expected_input_keys)
                if invalid_keys:
                    raise ValueError(
                        f"LLM filter '{llm_filter.name}' uses invalid input fields: {invalid_keys}. "
                        f"Available fields from original prompt: {expected_input_keys}"
                    )
                if not used_keys:
                    raise ValueError(
                        f"LLM filter '{llm_filter.name}' must reference at least one input field "
                        f"from the original prompt: {expected_input_keys}"
                    )

        # Set up agentic runner with validation
        runner = AgenticDirectiveRunner(
            input_data=input_data,
            agent_llm=agent_llm,
            validation_func=validate_llm_prompts,
        )

        # Create system prompt
        system_prompt = (
            "You are an expert at optimizing data processing pipelines by creating efficient filter cascades. "
            "You analyze data to identify patterns that allow for cheap pre-filtering before expensive operations. "
            "Your goal is to reduce costs while maintaining accuracy by using progressively more expensive filters."
        )

        # Create initial user message
        initial_message = self.to_string_for_instantiate(
            target_ops_configs, pipeline_code
        )

        # Run the agentic loop
        try:
            schema, updated_message_history, call_cost = runner.run_agentic_loop(
                system_prompt=system_prompt,
                initial_user_message=initial_message,
                response_schema=CascadeFilteringInstantiateSchema,
            )

            # Update message history
            message_history.extend(updated_message_history)

            return schema, message_history, call_cost

        except Exception as e:
            raise Exception(
                f"Failed to instantiate cascade_filtering directive: {str(e)}"
            )

    def apply(
        self,
        global_default_model: str,
        ops_list: List[Dict],
        target_ops: List[str],
        rewrite: CascadeFilteringInstantiateSchema,
    ) -> List[Dict]:
        """Apply the directive by injecting pre-filters before the target filter."""
        new_ops_list = []

        for op in ops_list:
            if op["name"] in target_ops:
                # First, add all code filters
                for i, code_filter in enumerate(rewrite.code_pre_filters):
                    code_filter_op = {
                        "name": f"{code_filter.name}_{op['name']}",
                        "type": "code_filter",
                        "code": code_filter.code,
                    }
                    new_ops_list.append(code_filter_op)

                # Sort LLM filters by prompt length (shortest first for cost efficiency)
                sorted_llm_filters = sorted(
                    rewrite.llm_pre_filters, key=lambda f: len(f.prompt)
                )

                # Then, add all LLM filters (using gpt-5-nano)
                for i, llm_filter in enumerate(sorted_llm_filters):
                    llm_filter_op = {
                        "name": f"{llm_filter.name}_{op['name']}",
                        "type": "filter",
                        "model": "gpt-5-nano",
                        "prompt": llm_filter.prompt,
                        "output": {"schema": {"keep": "boolean"}},
                    }
                    new_ops_list.append(llm_filter_op)

                # Finally, add the original filter
                new_ops_list.append(deepcopy(op))
            else:
                # Keep other operations as-is
                new_ops_list.append(deepcopy(op))

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
        1. Use agentic approach to analyze data and generate cascade filters
        2. Apply the transformation by injecting pre-filters
        """
        assert (
            len(target_ops) == 1
        ), "CascadeFiltering directive requires exactly one target operation"

        input_file_path = kwargs.get("input_file_path", None)
        pipeline_code = kwargs.get("pipeline_code", None)

        if not input_file_path:
            raise ValueError(
                "input_file_path is required for CascadeFiltering directive"
            )

        # Get configuration for target operation
        target_ops_configs = [op for op in operators if op["name"] in target_ops]

        if not target_ops_configs:
            raise ValueError(f"Target operation {target_ops[0]} not found in operators")

        if target_ops_configs[0]["type"] != "filter":
            raise ValueError(
                f"Target operation {target_ops[0]} must be a filter operation"
            )

        # Step 1: Agent analyzes data and generates cascade filters
        rewrite, message_history, call_cost = self.llm_instantiate(
            target_ops_configs,
            input_file_path,
            agent_llm,
            message_history,
            pipeline_code,
        )

        # Step 2: Apply transformation
        return (
            self.apply(global_default_model, operators, target_ops, rewrite),
            message_history,
            call_cost,
        )
