import json
import os
from copy import deepcopy
from typing import Dict, List, Type

from litellm import completion
from pydantic import BaseModel, Field

from docetl.reasoning_optimizer.instantiate_schemas import (
    ChunkHeaderSummaryInstantiateSchema,
)

from .base import MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS, Directive, DirectiveTestCase


class ChunkHeaderSummaryDirective(Directive):
    name: str = Field(
        default="chunk_header_summary", description="The name of the directive"
    )
    formal_description: str = Field(default="Split -> Gather => Split -> Map -> Gather")
    nl_description: str = Field(
        default="Transforms an existing Split -> Gather pipeline by inserting a Map operation between them that extracts headers and creates summaries from each chunk. The Gather operation is then modified to use summaries for middle chunks and headers for document structure. This directive enhances chunking pipelines with header extraction and chunk summarization capabilities. Only use this if it is clear that chunk-level analysis is insufficient because the chunk requires headers and summaries from other chunks to be interpreted correctly."
    )
    when_to_use: str = Field(
        default="Use only when you have an existing chunking pipeline (Split -> Gather) processing documents with clear hierarchical structure (legal contracts, technical manuals, research papers), and it is evident that chunk-level analysis is not accurate because the chunk needs headers and summaries from other chunks to make sense. This is beneficial when full chunk content in gather would be too verbose, and summarized or structured context is required for correct downstream processing. The target operators should be the split and gather. Make sure you specify these two operators when choosing this directive."
    )

    instantiate_schema_type: Type[BaseModel] = Field(
        default=ChunkHeaderSummaryInstantiateSchema
    )

    example: str = Field(
        default="""
            Original Pipeline (Split -> Gather):
            - name: split_contract_analysis
              type: split
              split_key: contract_text
              method: token_count
              method_kwargs:
                num_tokens: 1000

            - name: gather_contract_context
              type: gather
              content_key: contract_text_chunk
              doc_id_key: split_contract_analysis_id
              order_key: split_contract_analysis_chunk_num
              peripheral_chunks:
                previous:
                  tail:
                    count: 1

            Example InstantiateSchema Options (what the agent should output):

            # Basic header and summary extraction:
            {
              "header_extraction_prompt": "Extract any section headers or subsection titles from this contract chunk: {{ input.contract_text_chunk }}. Return the headers with their hierarchical levels.",
              "summary_prompt": "Summarize the key legal concepts and clause types in this contract chunk: {{ input.contract_text_chunk }}. Focus on liability, indemnification, and related contractual obligations.",
            }
        """,
    )

    test_cases: List[DirectiveTestCase] = Field(
        default_factory=lambda: [
            DirectiveTestCase(
                name="legal_document_with_structure",
                description="Should transform split->gather pipeline to include header extraction and summarization",
                input_config=[
                    {
                        "name": "split_legal_docs",
                        "type": "split",
                        "split_key": "agreement_text",
                        "method": "token_count",
                        "method_kwargs": {"num_tokens": 1000},
                    },
                    {
                        "name": "gather_legal_context",
                        "type": "gather",
                        "content_key": "agreement_text_chunk",
                        "doc_id_key": "split_legal_docs_id",
                        "order_key": "split_legal_docs_chunk_num",
                        "peripheral_chunks": {
                            "previous": {"tail": {"count": 1}},
                            "next": {"head": {"count": 1}},
                        },
                    },
                ],
                target_ops=["split_legal_docs", "gather_legal_context"],
                expected_behavior="Should insert parallel_map between split and gather for header extraction and summarization, with gather using doc_header_key",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="technical_manual_analysis",
                description="Should transform split->gather pipeline for technical documentation with header and summary context",
                input_config=[
                    {
                        "name": "split_manual",
                        "type": "split",
                        "split_key": "manual_text",
                        "method": "token_count",
                        "method_kwargs": {"num_tokens": 800},
                    },
                    {
                        "name": "gather_manual_context",
                        "type": "gather",
                        "content_key": "manual_text_chunk",
                        "doc_id_key": "split_manual_id",
                        "order_key": "split_manual_chunk_num",
                        "peripheral_chunks": {
                            "previous": {"tail": {"count": 2}},
                            "next": {"head": {"count": 1}},
                        },
                    },
                ],
                target_ops=["split_manual", "gather_manual_context"],
                expected_behavior="Should insert parallel_map between split and gather for header extraction and chunk summarization, enhancing technical documentation processing",
                should_pass=True,
            ),
        ]
    )

    def __eq__(self, other):
        return isinstance(other, ChunkHeaderSummaryDirective)

    def __hash__(self):
        return hash("ChunkHeaderSummaryDirective")

    def to_string_for_instantiate(self, original_op: Dict) -> str:
        """
        Generate a prompt for an agent to instantiate this directive.

        Args:
            original_op (Dict): The original operation.

        Returns:
            str: The agent prompt for instantiating the directive.
        """
        return (
            f"You are an expert at enhancing document processing pipelines with header extraction and chunk summarization.\n\n"
            f"Original Pipeline:\n"
            f"{str(original_op)}\n\n"
            f"Directive: {self.name}\n"
            f"Your task is to instantiate this directive by creating a configuration that enhances an existing Split -> Gather pipeline "
            f"by inserting a Map operation between them for header extraction and summarization.\n\n"
            f"Key requirements:\n"
            f"1. header_extraction_prompt: Create a prompt to extract headers/section titles from each chunk:\n"
            f"   - Use '{{{{ input.<split_key>_chunk }}}}' to reference chunk content (you'll know the split_key from the existing split operation)\n"
            f"   - Focus on document structure (titles, headings, section numbers)\n"
            f"   - Should output 'headers' field with hierarchical level information\n"
            f"2. summary_prompt: Create a prompt to summarize each chunk:\n"
            f"   - Use '{{{{ input.<split_key>_chunk }}}}' to reference chunk content\n"
            f"   - Focus on key concepts relevant to the downstream processing\n"
            f"   - Should output '<split_key>_summary' field\n"
            f"   - Keep summaries concise but informative for context\n"
            f"The header extraction helps maintain document structure context.\n"
            f"The summary provides condensed context from surrounding chunks.\n"
            f"The gather operation combines both for comprehensive context.\n\n"
            f"Example:\n"
            f"{self.example}\n\n"
            f"Please output only the ChunkHeaderSummaryInstantiateSchema object as JSON."
        )

    def llm_instantiate(
        self,
        original_op: Dict,
        agent_llm: str,
        message_history: list = [],
    ):
        """
        Use LLM to instantiate this directive by creating chunking configuration.

        Args:
            original_op (Dict): The original operation.
            agent_llm (str): The LLM model to use.
            message_history (List, optional): Conversation history for context.

        Returns:
            ChunkHeaderSummaryInstantiateSchema: The structured output from the LLM.
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
                response_format=ChunkHeaderSummaryInstantiateSchema,
            )
            call_cost = resp._hidden_params["response_cost"]
            try:
                parsed_res = json.loads(resp.choices[0].message.content)
                schema = ChunkHeaderSummaryInstantiateSchema(**parsed_res)
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
        rewrite: ChunkHeaderSummaryInstantiateSchema,
    ) -> List[Dict]:
        """
        Apply the directive to the pipeline config by inserting a parallel_map operation
        between the existing split and gather operations.
        """
        # Create a copy of the pipeline config
        new_ops_list = deepcopy(ops_list)

        # Find the split and gather operations
        split_op = None
        gather_op = None
        split_idx = None
        gather_idx = None

        for i, op in enumerate(new_ops_list):
            if op["name"] in target_ops:
                if op["type"] == "split":
                    split_op = op
                    split_idx = i
                elif op["type"] == "gather":
                    gather_op = op
                    gather_idx = i

        if not split_op or not gather_op:
            raise ValueError(
                "Both split and gather operations must be provided as target operations"
            )

        if split_idx >= gather_idx:
            raise ValueError(
                "Split operation must come before gather operation in the pipeline"
            )

        if gather_idx - split_idx > 1:
            raise ValueError(
                "There should not be operators between split and gather"
            )

        # Get the split_key from the split operation
        split_key = split_op.get("split_key")
        if not split_key:
            raise ValueError(
                f"Split operation '{split_op['name']}' must have a 'split_key' field"
            )

        # Create the parallel map operation for header extraction and summarization
        parallel_map_name = f"parallel_map_{split_op['name']}_header_summary"
        parallel_map_op = {
            "name": parallel_map_name,
            "type": "parallel_map",
            "prompts": [
                {
                    "name": f"header_extraction_{split_op['name']}",
                    "prompt": rewrite.header_extraction_prompt,
                    "output_keys": ["headers"],
                },
                {
                    "name": f"summary_{split_op['name']}",
                    "prompt": rewrite.summary_prompt,
                    "output_keys": [f"{split_key}_summary"],
                },
            ],
            "model": global_default_model,
            "output": {
                "schema": {
                    "headers": "list[{header: string, level: integer}]",
                    f"{split_key}_summary": "string",
                }
            },
        }

        # Add doc_header_key to the gather operation to use extracted headers
        new_ops_list[gather_idx]["doc_header_key"] = "headers"

        # Insert the parallel map operation between split and gather
        new_ops_list.insert(gather_idx, parallel_map_op)

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
        # Assert that there are exactly two target ops
        assert (
            len(target_ops) == 2
        ), "There must be exactly two target ops (split and gather) to instantiate this chunk header summary directive"

        # Find split and gather operations
        split_op = None
        gather_op = None
        for op in operators:
            if op["name"] in target_ops:
                if op["type"] == "split":
                    split_op = op
                elif op["type"] == "gather":
                    gather_op = op

        if not split_op:
            raise ValueError(
                f"Chunk header summary directive requires a split operation among target operations, but none found in {target_ops}"
            )

        if not gather_op:
            raise ValueError(
                f"Chunk header summary directive requires a gather operation among target operations, but none found in {target_ops}"
            )

        # Create a combined context for instantiation
        pipeline_context = {
            "split_op": split_op,
            "gather_op": gather_op,
            "target_ops": target_ops,
        }

        # Instantiate the directive
        rewrite, message_history, call_cost = self.llm_instantiate(
            pipeline_context, agent_llm, message_history
        )

        # Apply the rewrite to the operators
        return (
            self.apply(global_default_model, operators, target_ops, rewrite),
            message_history,
            call_cost,
        )
