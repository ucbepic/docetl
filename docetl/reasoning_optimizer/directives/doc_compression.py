import json
import os
from copy import deepcopy
from typing import Dict, List, Type

from litellm import completion
from pydantic import BaseModel, Field

from docetl.reasoning_optimizer.instantiate_schemas import (
    DocCompressionInstantiateSchema,
)

from .base import MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS, Directive, DirectiveTestCase


class DocCompressionDirective(Directive):
    name: str = Field(
        default="doc_compression", description="The name of the directive"
    )
    formal_description: str = Field(default="Op => Extract -> Op")
    nl_description: str = Field(
        default="Reduces LLM processing costs by using an Extract operator to intelligently compress documents before expensive downstream operations, removing irrelevant content that could distract the LLM"
    )
    when_to_use: str = Field(
        default="When documents contain irrelevant content and you want to reduce token costs for downstream LLM operations while improving accuracy by having the LLM focus on only the essential content"
    )

    instantiate_schema_type: Type[BaseModel] = Field(
        default=DocCompressionInstantiateSchema
    )

    example: str = Field(
        default="""
        Target Operations:
        - name: analyze_regulatory_impact
          type: map
          prompt: |
            Analyze the potential regulatory impact described in: {{ input.legal_document }}
            Consider stakeholder groups, compliance burdens, and implementation feasibility.
          output:
            schema:
              stakeholder_impacts: "list[str]"
              compliance_changes: "string"

        - name: extract_key_dates
          type: map
          prompt: |
            Extract important deadlines and dates from: {{ input.legal_document }}
          output:
            schema:
              deadlines: "list[str]"

        Example InstantiateSchema (what the agent should output):
        DocCompressionConfig(
            name="extract_regulatory_content",
            document_key="legal_document",
            prompt="Extract the minimal content necessary spanning: sections defining new regulatory requirements, stakeholder obligations, compliance deadlines, implementation timelines, and enforcement mechanisms. Focus only on substantive regulatory changes and specific dates, not background or procedural text.",
            model="gpt-4o-mini"
        )
        """
    )

    test_cases: List[DirectiveTestCase] = Field(
        default_factory=lambda: [
            DirectiveTestCase(
                name="single_target_compression",
                description="Should insert Extract operation before single target operation",
                input_config={
                    "name": "analyze_document",
                    "type": "map",
                    "prompt": "Analyze this document: {{ input.document }}",
                    "output": {"schema": {"analysis": "string"}},
                },
                target_ops=["analyze_document"],
                expected_behavior="Should add an Extract operation that compresses the document field before the analysis. The extract operation document_keys should be 'document' only.",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="multiple_target_compression",
                description="Should insert Extract operation before first target operation and consider all targets",
                input_config=[
                    {
                        "name": "extract_findings",
                        "type": "map",
                        "prompt": "Extract key findings from: {{ input.report }}",
                        "output": {"schema": {"findings": "list[str]"}},
                    },
                    {
                        "name": "analyze_impact",
                        "type": "map",
                        "prompt": "Analyze the business impact in: {{ input.report }}",
                        "output": {"schema": {"impact": "string"}},
                    },
                ],
                target_ops=["extract_findings", "analyze_impact"],
                expected_behavior="Should add Extract operation before extract_findings that considers content needed for both operations. The extract operation document_keys should be 'report' only.",
                should_pass=True,
            ),
        ]
    )

    def __eq__(self, other):
        return isinstance(other, DocCompressionDirective)

    def __hash__(self):
        return hash("DocCompressionDirective")

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
            f"You are an expert at document analysis and optimization.\n\n"
            f"Target Operations:\n"
            f"{ops_str}\n\n"
            f"Directive: {self.name}\n"
            f"Your task is to instantiate this directive by generating a DocCompressionConfig "
            f"that specifies how to compress the input document before processing.\n\n"
            f"The directive will insert an Extract operation that:\n"
            f"1. Takes a long document field from the input\n"
            f"2. Extracts only the MINIMAL content relevant to ALL the target operations\n"
            f"3. Replaces the original document field with the compressed content\n"
            f"4. Reduces token usage and improves focus for the downstream operations\n\n"
            f"The agent must output the configuration specifying:\n"
            f"- name: A descriptive name for the Extract operation\n"
            f"- document_key: Which document field contains the long content to compress\n"
            f"- prompt: Plain text instructions for what MINIMAL content to extract (NOT a Jinja template)\n"
            f"- model: Which model to use for extraction (typically a cheaper model like gpt-4o-mini)\n\n"
            f"IMPORTANT: The extraction prompt should focus on extracting the minimal content necessary "
            f"for ALL target operations. Analyze each operation's prompt to identify the "
            f"specific information types needed across all operations, then design an extraction prompt "
            f"that gets just those essential pieces while removing all irrelevant material.\n\n"
            f"Example:\n"
            f"{self.example}\n\n"
            f"Please output only the InstantiateSchema (DocCompressionConfig object) "
            f"that specifies how to apply this directive to the target operations."
        )

    def llm_instantiate(
        self,
        target_ops_configs: List[Dict],
        input_file_path: str,
        agent_llm: str,
        message_history: list = [],
    ):
        """
        Call the LLM to generate the instantiate schema.
        The LLM will output structured data matching DocCompressionInstantiateSchema.
        """

        message_history.extend(
            [
                {
                    "role": "user",
                    "content": self.to_string_for_instantiate(target_ops_configs),
                },
            ]
        )
        error_message = ""

        for _ in range(MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS):
            resp = completion(
                model=agent_llm,
                messages=message_history,
                api_key=os.environ.get("AZURE_API_KEY"),
                api_base=os.environ.get("AZURE_API_BASE"),
                api_version=os.environ.get("AZURE_API_VERSION"),
                azure=True,
                response_format=DocCompressionInstantiateSchema,
            )
            call_cost = resp._hidden_params["response_cost"]

            try:
                parsed_res = json.loads(resp.choices[0].message.content)
                schema = DocCompressionInstantiateSchema(**parsed_res)
                schema.validate_document_keys_exists_in_input(input_file_path)
                message_history.append(
                    {"role": "assistant", "content": resp.choices[0].message.content}
                )
                return schema, message_history, call_cost
            except Exception as err:
                error_message = f"Validation error: {err}\nPlease try again."
                message_history.append({"role": "user", "content": error_message})

        raise Exception(
            f"Failed to instantiate directive after {MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS} attempts. Error: {error_message}"
        )

    def apply(
        self,
        global_default_model: str,
        ops_list: List[Dict],
        target_ops: List[str],
        rewrite: DocCompressionInstantiateSchema,
    ) -> List[Dict]:
        """
        Apply the directive using the instantiate schema configuration.
        Inserts an Extract operation before the first target operation.
        """
        new_ops_list = deepcopy(ops_list)

        # Find the position of the first target operation
        first_target_pos = min(
            [i for i, op in enumerate(ops_list) if op["name"] in target_ops]
        )

        extract_op = {
            "name": rewrite.name,
            "type": "extract",
            "prompt": rewrite.prompt,
            "document_keys": [rewrite.document_key],
            "litellm_completion_kwargs": {"temperature": 0},
            "model": rewrite.model,
        }

        # Insert the Extract operation before the first target operation
        new_ops_list.insert(first_target_pos, extract_op)

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
        input_file_path = kwargs.get("input_file_path", None)

        # Get configurations for all target operations
        target_ops_configs = [op for op in operators if op["name"] in target_ops]

        # Step 1: Agent generates the instantiate schema considering all target ops
        rewrite, message_history, call_cost = self.llm_instantiate(
            target_ops_configs, input_file_path, agent_llm, message_history
        )

        # Step 2: Apply transformation using the schema
        return (
            self.apply(global_default_model, operators, target_ops, rewrite),
            message_history,
            call_cost,
        )
