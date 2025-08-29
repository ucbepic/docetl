import json
import os
from copy import deepcopy
from typing import Dict, List, Type

from litellm import completion
from pydantic import BaseModel, Field

from docetl.reasoning_optimizer.instantiate_schemas import (
    DocumentChunkingInstantiateSchema,
)

from .base import MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS, Directive, DirectiveTestCase


class DocumentChunkingDirective(Directive):
    name: str = Field(default="doc_chunking", description="The name of the directive")
    formal_description: str = Field(
        default="Map => Split -> Gather -> [Sample] -> Map -> Reduce"
    )
    nl_description: str = Field(
        default="Transforms a single Map operation into a chunking pipeline: splits long documents into chunks, gathers context around each chunk, optionally samples a subset of chunks for efficiency, processes chunks with a new Map operation, then reduces the results. By default, sampling is applied unless the task requires processing ALL chunks. This directive can only be applied to a top-level Map operation, not to a sub-map within a pipeline that already contains a split, gather, or reduce sequence."
    )
    when_to_use: str = Field(
        default="Use when you need to process long documents to extract information, and the document is too long for a single Map operation. The agent will automatically decide whether to sample chunks (for tasks like categorization, theme extraction) or process all chunks (for comprehensive extraction of all instances). Do not apply if the target operation is already part of a split -> gather -> map -> reduce pipeline. Use different gather configs: 'previous.head' for documents with key metadata/definitions at the start, 'previous.tail' for maintaining references, and 'next.head' only for tables/clauses spanning chunks."
    )

    instantiate_schema_type: Type[BaseModel] = Field(
        default=DocumentChunkingInstantiateSchema
    )

    example: str = Field(
        default="""
            Original Op (MapOpConfig):
            - name: extract_contract_terms
              type: map
              prompt: |
                Extract all payment terms, deadlines, and penalty clauses from the contract:
                {{ input.contract_text }}
                Return a comprehensive list of all terms found.
              output:
                schema:
                  contract_terms: list[str]

            Example InstantiateSchema Options (what the agent should output):

            # Basic context - good for most cases:
            {
              "chunk_size": 10000,
              "split_key": "contract_text",
              "sub_prompt": "You are analyzing a chunk of a larger document. Extract all payment terms, deadlines, and penalty clauses from this contract chunk: {{ input.contract_text_chunk_rendered }}. Return a comprehensive list of all terms found.",
              "reduce_prompt": "Combine results from multiple document chunks: Extract all payment terms, deadlines, and penalty clauses by combining the results from each chunk: {% for input in inputs %}{{ input.contract_terms | join(', ') }}{% if not loop.last %}, {% endif %}{% endfor %}. Remove duplicates and return a comprehensive list of all terms found.",
              "gather_config": {
                "previous": {
                  "tail": {
                    "count": 0.5
                  }
                }
              },
            }

            # Rich context - for complex documents needing document-level metadata:
            {
              "gather_config": {
                "previous": {
                  "head": {
                    "count": 1,
                    "content_key": "contract_text_chunk"
                  },
                  "tail": {
                    "count": 2,
                    "content_key": "contract_text_chunk"
                  }
                }
              }
            }

            # Forward context - for tables or clauses spanning chunks:
            {
              "gather_config": {
                "previous": {
                  "tail": {
                    "count": 1,
                    "content_key": "contract_text_chunk"
                  }
                },
                "next": {
                  "head": {
                    "count": 1,
                    "content_key": "contract_text_chunk"
                  }
                }
              }
            }
        """,
    )

    test_cases: List[DirectiveTestCase] = Field(
        default_factory=lambda: [
            DirectiveTestCase(
                name="comprehensive_legal_analysis",
                description="Should transform complex legal document analysis into chunking pipeline",
                input_config={
                    "name": "analyze_legal_document",
                    "type": "map",
                    "prompt": "From this legal document, extract all liability clauses with risk ratings (1-10), identify all parties and their obligations, find all monetary amounts with currencies, extract all dates and deadlines with legal consequences, and list all governing laws or jurisdictions mentioned. For each liability clause, assess the risk level considering industry standards and provide specific reasoning. Group findings by document section if clearly indicated. Return comprehensive analysis ensuring no critical legal elements are missed: {{ input.legal_document }}",
                    "output": {
                        "schema": {
                            "liability_analysis": "list[str]",
                            "parties_obligations": "list[str]",
                            "financial_terms": "list[str]",
                            "critical_dates": "list[str]",
                            "governing_laws": "list[str]",
                            "risk_assessment": "str",
                        }
                    },
                },
                target_ops=["analyze_legal_document"],
                expected_behavior="Should create chunking pipeline where sub_prompt covers all extraction tasks (liability, parties, financial, dates, laws) with same risk assessment criteria, and reduce_prompt aggregates all findings maintaining the complete analytical framework and output schema from original prompt",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="clinical_trial_comprehensive_extraction",
                description="Should transform detailed clinical research analysis into chunking pipeline",
                input_config={
                    "name": "extract_clinical_data",
                    "type": "map",
                    "prompt": "Analyze this clinical trial document and extract: primary and secondary endpoints with measurement criteria, all adverse events categorized by severity (mild/moderate/severe), patient demographics including inclusion/exclusion criteria, statistical significance results with p-values and confidence intervals, drug dosages and administration protocols, and study methodology details. For each adverse event, determine if it's treatment-related based on temporal relationship and biological plausibility. Calculate overall safety profile score (1-10) considering frequency and severity of events. Ensure all regulatory compliance elements are captured: {{ input.trial_document }}",
                    "output": {
                        "schema": {
                            "endpoints": "list[str]",
                            "adverse_events": "list[str]",
                            "demographics": "str",
                            "statistical_results": "list[str]",
                            "protocols": "list[str]",
                            "safety_assessment": "str",
                            "compliance_status": "str",
                        }
                    },
                },
                target_ops=["extract_clinical_data"],
                expected_behavior="Should create chunking pipeline where sub_prompt preserves all clinical analysis requirements (endpoints, adverse events, demographics, statistics, protocols) with same assessment criteria, and reduce_prompt combines results maintaining complete clinical framework and safety scoring methodology from original prompt",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="financial_comprehensive_analysis",
                description="Should transform complex financial document analysis into chunking pipeline",
                input_config={
                    "name": "analyze_financial_report",
                    "type": "map",
                    "prompt": "From this annual financial report, extract all revenue streams with growth rates and market segments, identify all risk factors with impact assessments (low/medium/high), find all forward-looking statements and their associated uncertainties, extract key financial ratios and calculate trend analysis over mentioned periods, identify all subsidiaries with their contribution to consolidated results, and analyze competitive positioning statements. For each risk factor, assess potential financial impact in dollar ranges and likelihood percentages. Ensure all material information affecting investor decisions is captured and categorized by urgency level: {{ input.financial_report }}",
                    "output": {
                        "schema": {
                            "revenue_analysis": "list[str]",
                            "risk_factors": "list[str]",
                            "forward_statements": "list[str]",
                            "financial_ratios": "str",
                            "subsidiaries": "list[str]",
                            "competitive_analysis": "str",
                            "material_disclosures": "list[str]",
                        }
                    },
                },
                target_ops=["analyze_financial_report"],
                expected_behavior="Should create chunking pipeline where sub_prompt maintains all financial analysis requirements (revenue, risks, statements, ratios, subsidiaries, competitive analysis) with same impact assessment methodology, and reduce_prompt aggregates preserving complete financial analytical framework and materiality assessments from original prompt",
                should_pass=True,
            ),
        ]
    )

    def __eq__(self, other):
        return isinstance(other, DocumentChunkingDirective)

    def __hash__(self):
        return hash("DocumentChunkingDirective")

    def to_string_for_instantiate(self, original_op: Dict) -> str:
        """
        Generate a prompt for an agent to instantiate this directive.

        Args:
            original_op (Dict): The original operation.

        Returns:
            str: The agent prompt for instantiating the directive.
        """
        return (
            f"You are an expert at transforming document processing operations into chunking pipelines.\n\n"
            f"Original Operation:\n"
            f"{str(original_op)}\n\n"
            f"Directive: {self.name}\n"
            f"Your task is to instantiate this directive by creating a configuration that transforms the original Map operation "
            f"into a Split -> Gather -> Map -> Reduce pipeline for processing long documents in chunks.\n\n"
            f"Key requirements:\n"
            f"1. chunk_size: Choose an appropriate token count (typically 10000-15000) based on the complexity of the task\n"
            f"2. split_key: Identify the document field to split from the original operation's prompt. Make sure to use the same field as the original operation's prompt.\n"
            f"3. sub_prompt: Take the original prompt exactly but:\n"
            f"   - Add instruction at start: 'You are analyzing a chunk of a larger document.'\n"
            f"   - Replace '{{{{ input.<split_key> }}}}' with '{{{{ input.<split_key>_chunk_rendered }}}}'\n"
            f"   - Keep everything else identical (same task, same output schema)\n"
            f"4. reduce_prompt: Take original task instructions but adapt for aggregation:\n"
            f"   - Start with: 'Combine results from multiple document chunks:'\n"
            f"   - Include same task context/requirements as original prompt\n"
            f"   - Use '{{% for input in inputs %}}' to iterate over chunk results\n"
            f"   - Combine/deduplicate to match original output schema exactly\n"
            f"5. sampling_config: IMPORTANT - Include sampling by default UNLESS the task requires ALL chunks:\n"
            f"   - ALWAYS use sampling for: categorization, theme identification, sentiment analysis, document type classification\n"
            f"   - NEVER use sampling for: comprehensive extraction ('extract ALL instances'), complete analysis requiring every chunk\n"
            f"   - Default sampling: method='uniform' with stratify_key, samples=5-10 chunks\n"
            f"   - For simple tasks (categorization): samples=1-3 chunks\n"
            f"   - For complex analysis: samples=5-15 chunks\n"
            f"   - For stratified sampling: specify method='uniform' and stratify_key (note: split document ID is automatically included)\n"
            f"   - Set sampling_config=null only if you need to process every single chunk\n"
            f"6. gather_config: Configure context from surrounding chunks. Structure:\n"
            f"   gather_config:\n"
            f"     previous:  # chunks before current chunk\n"
            f"       head:    # first chunk(s) in document\n"
            f"         count: 1\n"
            f"         content_key: full_content_chunk  # optional, defaults to main key chunk\n"
            f"       tail:    # chunk(s) immediately before current\n"
            f"         count: 2\n"
            f"         content_key: full_content_chunk  # optional\n"
            f"     next:      # chunks after current chunk\n"
            f"       head:    # chunk(s) immediately after current\n"
            f"         count: 1\n"
            f"         content_key: full_content_chunk  # optional\n"
            f"   Usage guidelines:\n"
            f"   - Use 'previous.head' when document has important metadata/definitions at start\n"
            f"   - Use 'previous.tail' to maintain references and immediate context\n"
            f"   - Use 'next.head' only for tables/clauses that span chunk boundaries\n"
            f"   - Minimize context: only include what's needed for accurate operation\n"
            f"   - Count can be float (e.g., 0.5 for half chunk, 1.5 for chunk and a half)\n"
            f"   - More context increases token usage and cost - be judicious\n"
            f"   - Default to 0.5 previous tail if unsure about context needs\n"
            f"The sub_prompt should focus on the main chunk content and extract the same type of information as the original.\n"
            f"The reduce_prompt must produce the same output schema as the original operation.\n\n"
            f"Example:\n"
            f"{self.example}\n\n"
            f"Please output only the DocumentChunkingInstantiateSchema object as JSON."
        )

    def llm_instantiate(
        self,
        operators,
        input_file_path,
        original_op: Dict,
        agent_llm: str,
        message_history: list = [],
    ) -> tuple:
        """
        Use LLM to instantiate this directive by creating chunking configuration.

        Args:
            original_op (Dict): The original operation.
            agent_llm (str): The LLM model to use.
            message_history (List, optional): Conversation history for context.

        Returns:
            DocumentChunkingInstantiateSchema: The structured output from the LLM.
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
            call_cost = 0.0
            resp = completion(
                model=agent_llm,
                messages=message_history,
                api_key=os.environ.get("AZURE_API_KEY"),
                api_base=os.environ.get("AZURE_API_BASE"),
                api_version=os.environ.get("AZURE_API_VERSION"),
                azure=True,
                response_format=DocumentChunkingInstantiateSchema,
            )
            call_cost = resp._hidden_params["response_cost"]
            try:
                parsed_res = json.loads(resp.choices[0].message.content)
                schema = DocumentChunkingInstantiateSchema(**parsed_res)
                schema.validate_stratify_key_in_pipeline(operators)
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
        rewrite: DocumentChunkingInstantiateSchema,
    ) -> List[Dict]:
        """
        Apply the directive to the pipeline config by replacing the target operation
        with a split -> gather -> map -> reduce sequence.
        """

        # Create a copy of the pipeline config
        new_ops_list = deepcopy(ops_list)
        target_op_config = [op for op in new_ops_list if op["name"] == target_op][0]

        # Find position of the target op to replace
        pos_to_replace = [
            i for i, op in enumerate(ops_list) if op["name"] == target_op
        ][0]

        original_op = ops_list[pos_to_replace]
        original_model = target_op_config.get("model", global_default_model)

        # Create operation names based on the original operation name
        split_name = f"split_{target_op}"
        gather_name = f"gather_{target_op}"
        sample_name = f"sample_{target_op}_chunks"
        map_name = f"map_{target_op}_chunks"
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

        # Create the gather operation with agent-configured context
        # Convert Pydantic model to dict, excluding None values
        gather_config_dict = (
            rewrite.gather_config.model_dump(exclude_none=True)
            if rewrite.gather_config
            else {}
        )
        # Use default config if the gather_config is empty (all fields were None)
        if not gather_config_dict:
            gather_config_dict = {"previous": {"tail": {"count": 1}}}

        gather_op = {
            "name": gather_name,
            "type": "gather",
            "content_key": f"{rewrite.split_key}_chunk",
            "doc_id_key": f"{split_name}_id",
            "order_key": f"{split_name}_chunk_num",
            "peripheral_chunks": gather_config_dict,
        }

        # Create the map operation for processing chunks
        map_op = {
            "name": map_name,
            "type": "map",
            "prompt": rewrite.sub_prompt,
            "model": original_model,
            "litellm_completion_kwargs": {"temperature": 0},
            "output": deepcopy(original_op["output"]),  # Same output schema as original
        }

        # Create the reduce operation
        reduce_op = {
            "name": reduce_name,
            "type": "reduce",
            "reduce_key": f"{split_name}_id",
            "prompt": rewrite.reduce_prompt,
            "model": original_model,
            "litellm_completion_kwargs": {"temperature": 0},
            "output": deepcopy(original_op["output"]),  # Same output schema as original
            "associative": False,  # Order matters for chunks,
            "pass_through": True,
        }

        # Construct operation sequence
        ops_sequence = [split_op, gather_op]

        # Add sample operation if sampling config is provided
        if rewrite.sampling_config:
            sample_op = {
                "name": sample_name,
                "type": "sample",
                "method": rewrite.sampling_config.method,
                "samples": rewrite.sampling_config.samples,
            }

            # Always stratify by split document ID and set samples_per_group
            stratify_keys = [f"{split_name}_id"]

            # Add agent-specified stratify key if provided
            if rewrite.sampling_config.stratify_key:
                stratify_keys.append(rewrite.sampling_config.stratify_key)
            sample_op["stratify_key"] = stratify_keys

            if rewrite.sampling_config.samples_per_group:
                sample_op["samples_per_group"] = rewrite.sampling_config.samples_per_group
            
            
            # Add optional fields if provided
            if rewrite.sampling_config.random_state is not None:
                sample_op["random_state"] = rewrite.sampling_config.random_state
            
            if rewrite.sampling_config.method_kwargs is not None:
                try:
                    sample_op["method_kwargs"] = json.loads(rewrite.sampling_config.method_kwargs)
                except Exception as e:
                    raise ValueError(f"Invalid method_kwargs: {e}")

            ops_sequence.append(sample_op)

        # Add map and reduce operations
        ops_sequence.extend([map_op, reduce_op])

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
    ):
        """
        Instantiate the directive for a list of operators.
        """
        # Assert that there is only one target op
        assert (
            len(target_ops) == 1
        ), "There must be exactly one target op to instantiate this document chunking directive"
        target_op_config = [op for op in operators if op["name"] == target_ops[0]][0]

        input_file_path = kwargs.get("input_file_path", None)

        # Validate that the target operation is a map operation
        if target_op_config.get("type") != "map":
            raise ValueError(
                f"Document chunking directive can only be applied to map operations, but target operation '{target_ops[0]}' is of type '{target_op_config.get('type')}'"
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
