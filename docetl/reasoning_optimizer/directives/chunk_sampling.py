import json
import os
from copy import deepcopy
from typing import Dict, List, Type

from litellm import completion
from pydantic import BaseModel, Field

from docetl.reasoning_optimizer.instantiate_schemas import (
    ChunkSamplingInstantiateSchema,
)

from .base import MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS, Directive, DirectiveTestCase


class ChunkSamplingDirective(Directive):
    name: str = Field(default="chunk_sampling", description="The name of the directive")
    formal_description: str = Field(default="Gather -> Map => Gather -> Sample -> Map")
    nl_description: str = Field(
        default=(
            "Adds a Sample operation between Gather and Map operations. "
            "This reduces the number of chunks processed by sampling only a fraction of them. "
            "ONLY use when the task doesn't require examining ALL chunks - like categorization, determining X reasons, "
            "or other tasks where a representative sample is sufficient. NOT for extracting all key insights."
        )
    )
    when_to_use: str = Field(
        default=(
            "Use ONLY when the task doesn't need to look at all document chunks. Good for: categorization, "
            "'determine X reasons why...', representative analysis. BAD for: extracting all key insights, "
            "comprehensive information extraction. Target should be Gather -> Map sequence."
        )
    )

    instantiate_schema_type: Type[BaseModel] = Field(
        default=ChunkSamplingInstantiateSchema
    )

    example: str = Field(
        default="""
        Original Pipeline (Split -> Gather -> Map -> Reduce):
        [
          {
            "name": "split_document",
            "type": "split",
            "split_key": "content",
            "method": "token_count",
            "method_kwargs": {"num_tokens": 1000}
          },
          {
            "name": "gather_context",
            "type": "gather",
            "content_key": "content_chunk",
            "doc_id_key": "split_document_id",
            "order_key": "split_document_chunk_num"
          },
          {
            "name": "extract_insights",
            "type": "map",
            "prompt": "Extract key insights from: {{ input.content_chunk_rendered }}",
            "output": {"schema": {"insights": "list[str]"}}
          },
          {
            "name": "combine_insights",
            "type": "reduce",
            "reduce_key": "split_document_id",
            "prompt": "Combine insights: {% for input in inputs %}{{ input.insights | join(', ') }}{% endfor %}"
          }
        ]

        Example InstantiateSchema (what the agent should output):
        ChunkSamplingConfig(
            method="uniform",
            samples=0.2  # Process only 20% of chunks
        )

        The methods can be "uniform", "first", or "stratify".

        Result: Sample operation inserted between gather_context and extract_insights,
        reducing processing by 80% while maintaining representative insights.
        """,
    )

    test_cases: List[DirectiveTestCase] = [
        DirectiveTestCase(
            name="document_categorization",
            description="Simple categorization task needs very few samples",
            input_config=[
                {
                    "name": "gather_chunks",
                    "type": "gather",
                    "content_key": "document_chunk",
                    "doc_id_key": "split_docs_id",
                    "order_key": "split_docs_chunk_num",
                },
                {
                    "name": "categorize_document",
                    "type": "map",
                    "prompt": "What category does this document belong to? {{ input.document_chunk_rendered }}",
                    "output": {"schema": {"category": "string"}},
                },
            ],
            target_ops=["gather_chunks", "categorize_document"],
            expected_behavior="Should use small sample (0.05-0.2) for simple categorization",
            should_pass=True,
        ),
        DirectiveTestCase(
            name="determine_key_themes",
            description="Finding themes needs moderate sampling",
            input_config=[
                {
                    "name": "gather_chunks",
                    "type": "gather",
                    "content_key": "document_chunk",
                    "doc_id_key": "split_docs_id",
                    "order_key": "split_docs_chunk_num",
                },
                {
                    "name": "extract_themes",
                    "type": "map",
                    "prompt": "Determine main themes from this chunk: {{ input.document_chunk_rendered }}",
                    "output": {"schema": {"themes": "list[string]"}},
                },
            ],
            target_ops=["gather_chunks", "extract_themes"],
            expected_behavior="Should use moderate sample (0.2-0.5) for theme extraction",
            should_pass=True,
        ),
    ]

    def __eq__(self, other):
        return isinstance(other, ChunkSamplingDirective)

    def __hash__(self):
        return hash("ChunkSamplingDirective")

    def to_string_for_instantiate(
        self, operators: List[Dict], target_ops: List[str]
    ) -> str:
        """
        Generate a prompt for an agent to instantiate this directive.

        Args:
            operators: List of all operators in the pipeline
            target_ops: List of target operation names (should be the Map operation in chunking sequence)

        Returns:
            str: The agent prompt for instantiating the directive.
        """
        return (
            f"You are an expert at optimizing data processing pipelines for performance and cost.\n\n"
            f"Full Pipeline Context:\n"
            f"{str(operators)}\n\n"
            f"Target Operations (Gather -> Map sequence): {target_ops}\n\n"
            f"Directive: {self.name}\n"
            f"Your task is to instantiate this directive by generating a ChunkSamplingConfig that adds a Sample operation "
            f"between the Gather and Map operations.\n\n"
            f"Configuration steps:\n"
            f"1. Determine appropriate sampling fraction based on the task (typically 0.05-0.3)\n"
            f"2. Choose sampling method - can be 'uniform', 'first', or 'stratify'\n"
            f"   - 'uniform': Random sampling (most common)\n"
            f"   - 'first': Take first N samples\n"
            f"   - 'stratify': Sample equally from each category (needs stratify_key in method_kwargs)\n"
            f"3. Add method_kwargs if needed (e.g., stratify_key for stratified sampling)\n\n"
            f"Examples from sample operation:\n"
            f"- Uniform: method='uniform', samples=0.2\n"
            f"- Stratified: method='stratify', samples=0.2, method_kwargs={{'stratify_key': 'category'}}\n\n"
            f"The Sample operation will be inserted between the Gather and Map operations.\n\n"
            f"Example:\n"
            f"{self.example}\n\n"
            f"Please output only the InstantiateSchema (a ChunkSamplingConfig object)."
        )

    def llm_instantiate(
        self,
        operators: List[Dict],
        target_ops: List[str],
        agent_llm: str,
        message_history: list = [],
    ) -> tuple:
        """
        Use LLM to instantiate this directive by creating a sampling operation.

        Args:
            operators (List[Dict]): All operators in the pipeline.
            target_ops (List[str]): Target operation names (Map operation in chunking sequence).
            agent_llm (str): The LLM model to use.
            message_history (List, optional): Conversation history for context.

        Returns:
            ChunkSamplingInstantiateSchema: The structured output from the LLM.
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
                response_format=ChunkSamplingInstantiateSchema,
            )

            try:
                parsed_res = json.loads(resp.choices[0].message.content)
                schema = ChunkSamplingInstantiateSchema(**parsed_res)
                message_history.append(
                    {"role": "assistant", "content": resp.choices[0].message.content}
                )
                return schema, message_history
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
        rewrite: ChunkSamplingInstantiateSchema,
    ) -> List[Dict]:
        """
        Apply the directive by inserting a Sample operation between Gather and Map operations.
        """
        # Create a copy of the pipeline config
        new_ops_list = deepcopy(ops_list)

        # Find positions of the two target operations
        gather_pos = None
        map_pos = None

        for i, op in enumerate(new_ops_list):
            if op["name"] == target_ops[0]:
                gather_pos = i
            elif op["name"] == target_ops[1]:
                map_pos = i

        if gather_pos is None:
            raise ValueError(
                f"Gather operation '{target_ops[0]}' not found in pipeline"
            )
        if map_pos is None:
            raise ValueError(f"Map operation '{target_ops[1]}' not found in pipeline")

        # Validate that gather comes before map and they are adjacent
        if map_pos != gather_pos + 1:
            raise ValueError(
                f"Gather operation '{target_ops[0]}' and Map operation '{target_ops[1]}' must be adjacent"
            )

        # Validate operation types
        gather_op = new_ops_list[gather_pos]
        map_op = new_ops_list[map_pos]

        if gather_op.get("type") != "gather":
            raise ValueError(
                f"First target operation '{target_ops[0]}' must be a gather operation"
            )
        if map_op.get("type") != "map":
            raise ValueError(
                f"Second target operation '{target_ops[1]}' must be a map operation"
            )

        # Create the sample operation configuration with generated name
        sample_op = {
            "name": f"sample_{target_ops[1].replace('_', '')}",
            "type": "sample",
            "method": rewrite.method,
            "samples": rewrite.samples,
        }

        # Add optional parameters if provided
        if rewrite.method_kwargs:
            sample_op["method_kwargs"] = rewrite.method_kwargs

        # Insert the sample operation between gather and map
        new_ops_list.insert(map_pos, sample_op)

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
        # Assert that there are exactly two target ops
        assert (
            len(target_ops) == 2
        ), "There must be exactly two target ops (Gather -> Map) to instantiate this chunk sampling directive"

        gather_name, map_name = target_ops[0], target_ops[1]
        gather_op = None
        map_op = None

        # Find the target operations
        for op in operators:
            if op["name"] == gather_name:
                gather_op = op
            elif op["name"] == map_name:
                map_op = op

        if gather_op is None:
            raise ValueError(f"Gather operation '{gather_name}' not found in pipeline")
        if map_op is None:
            raise ValueError(f"Map operation '{map_name}' not found in pipeline")

        # Validate operation types
        if gather_op.get("type") != "gather":
            raise ValueError(
                f"First target operation '{gather_name}' must be a gather operation"
            )
        if map_op.get("type") != "map":
            raise ValueError(
                f"Second target operation '{map_name}' must be a map operation"
            )

        # Instantiate the directive
        rewrite, message_history = self.llm_instantiate(
            operators, target_ops, agent_llm, message_history
        )

        # Apply the rewrite to the operators
        return (
            self.apply(global_default_model, operators, target_ops, rewrite),
            message_history,
        )
