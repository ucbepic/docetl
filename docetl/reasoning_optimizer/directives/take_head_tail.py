import json
import os
from copy import deepcopy
from typing import Dict, List, Type

from litellm import completion
from pydantic import BaseModel, Field

from docetl.reasoning_optimizer.instantiate_schemas import TakeHeadTailInstantiateSchema

from .base import MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS, Directive, DirectiveTestCase


class TakeHeadTailDirective(Directive):
    name: str = Field(default="take_head_tail", description="The name of the directive")
    formal_description: str = Field(
        default="LLM_Op => Code Map -> LLM_Op",
        description="Inserts a Code Map operation before any LLM-powered operation (Map, Filter, Reduce) to truncate document content to head and tail words",
    )
    nl_description: str = Field(
        default="Reduces document length by keeping only the first k words and optionally the last l words of the longest document field. This improves cost efficiency and can enhance accuracy for tasks that only require document beginnings (like classification)."
    )
    when_to_use: str = Field(
        default="When any LLM operation (Map, Filter, Reduce) only needs the beginning (and optionally end) of documents, such as classification tasks, filtering by document type, reducing document summaries, or when full document content causes accuracy issues due to too much context."
    )

    instantiate_schema_type: Type[BaseModel] = Field(
        default=TakeHeadTailInstantiateSchema
    )

    example: str = Field(
        default="""
        Original Map Operation (Research Paper Classification):
        - name: classify_research_domain
          type: map
          prompt: |
            What research domain does this paper belong to? Classify as Computer Science, Biology, Physics, Chemistry, or Other based on: {{ input.paper_text }}
          output:
            schema:
              domain: "string"
              confidence: "float"
          model: gpt-4o-mini

        TakeHeadTailInstantiateSchema:
        {
          "name": "extract_paper_abstract",
          "document_key": "paper_text",
          "head_words": 150,
          "tail_words": 0
        }

        Resulting Pipeline:
        - name: extract_paper_abstract
          type: code_map
          code: |
            def transform(input_doc):
                paper_text_content = input_doc.get('paper_text', '')
                words = paper_text_content.split()
                if len(words) <= 150:
                    truncated = paper_text_content
                else:
                    head = ' '.join(words[:150])
                    truncated = head
                return {'paper_text': truncated}
        - name: classify_research_domain
          type: map
          prompt: |
            What research domain does this paper belong to? Classify as Computer Science, Biology, Physics, Chemistry, or Other based on: {{ input.paper_text }}
          output:
            schema:
              domain: "string"
              confidence: "float"
          model: gpt-4o-mini
        """
    )

    test_cases: List[DirectiveTestCase] = Field(
        default_factory=lambda: [
            DirectiveTestCase(
                name="research_paper_classification",
                description="Classify research papers by domain using only abstract/introduction",
                input_config={
                    "name": "classify_paper_domain",
                    "type": "map",
                    "prompt": "What research domain does this paper belong to (CS, Biology, Physics, etc.)? Base your classification on the content: {{ input.full_text }}",
                    "output": {"schema": {"domain": "string", "confidence": "float"}},
                    "model": "gpt-4o-mini",
                },
                target_ops=["classify_paper_domain"],
                expected_behavior="Should truncate full_text to first ~150 words (abstract/intro) since paper classification only needs the beginning, not the full methodology/results sections",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="document_metadata_extraction",
                description="Extract metadata from document headers/footers for indexing",
                input_config={
                    "name": "extract_document_metadata",
                    "type": "map",
                    "prompt": "Extract the title, author, date, and document type from this document: {{ input.content }}",
                    "output": {
                        "schema": {
                            "title": "string",
                            "author": "string",
                            "date": "string",
                            "doc_type": "string",
                        }
                    },
                    "model": "gpt-4o-mini",
                },
                target_ops=["extract_document_metadata"],
                expected_behavior="Should keep both head (~100 words for headers/title) and tail (~50 words for footers/signatures) since metadata appears at document beginning and end",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="email_priority_classification",
                description="Classify email priority using subject and first paragraph",
                input_config={
                    "name": "classify_email_priority",
                    "type": "map",
                    "prompt": "Classify this email as HIGH, MEDIUM, or LOW priority based on urgency indicators: {{ input.email_body }}",
                    "output": {"schema": {"priority": "string", "reasoning": "string"}},
                    "model": "gpt-4o-mini",
                },
                target_ops=["classify_email_priority"],
                expected_behavior="Should truncate email_body to first ~75 words since email priority is determined by subject line and opening, not full conversation thread",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="legal_document_type_identification",
                description="Identify legal document type from contract headers and signature blocks",
                input_config={
                    "name": "identify_legal_doc_type",
                    "type": "map",
                    "prompt": "What type of legal document is this (contract, agreement, policy, etc.)? Analyze: {{ input.legal_text }}",
                    "output": {
                        "schema": {
                            "document_type": "string",
                            "parties_involved": "list[string]",
                        }
                    },
                    "model": "gpt-4o-mini",
                },
                target_ops=["identify_legal_doc_type"],
                expected_behavior="Should keep head (~200 words for title/parties) and tail (~100 words for signature blocks) since legal doc type is indicated at beginning and parties sign at end",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="spam_email_filtering",
                description="Filter out spam emails based on subject line and opening content",
                input_config={
                    "name": "filter_spam_emails",
                    "type": "filter",
                    "prompt": "Is this email spam? Look for suspicious patterns in: {{ input.email_content }}",
                    "output": {"schema": {"_bool": "bool"}},
                    "model": "gpt-4o-mini",
                },
                target_ops=["filter_spam_emails"],
                expected_behavior="Should truncate email_content to first ~100 words since spam detection relies on subject, sender, and opening content, not full email thread",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="research_findings_synthesis",
                description="Reduce multiple research papers into a unified findings summary",
                input_config={
                    "name": "synthesize_research_findings",
                    "type": "reduce",
                    "prompt": "Synthesize the key findings from these research abstracts and conclusions: {% for doc in inputs %}{{ doc.paper_content }}{% endfor %}",
                    "output": {
                        "schema": {"synthesis": "string", "key_themes": "list[string]"}
                    },
                    "model": "gpt-4o-mini",
                },
                target_ops=["synthesize_research_findings"],
                expected_behavior="Should keep head (~200 words for abstracts) and tail (~150 words for conclusions) from each paper since synthesis needs both research goals and outcomes",
                should_pass=True,
            ),
        ]
    )

    def __eq__(self, other):
        return isinstance(other, TakeHeadTailDirective)

    def __hash__(self):
        return hash("TakeHeadTailDirective")

    def to_string_for_instantiate(self, original_op: Dict) -> str:
        op_type = original_op.get("type", "operation")
        op_type_caps = op_type.capitalize()

        return (
            f"You are an expert at optimizing document processing pipelines for cost and accuracy.\n\n"
            f"Original {op_type_caps} Operation:\n"
            f"{str(original_op)}\n\n"
            f"Directive: {self.name}\n"
            f"Your task is to instantiate this directive by generating a TakeHeadTailInstantiateSchema "
            f"that specifies how to truncate document content to improve efficiency.\n\n"
            f"The directive will insert a Code Map operation before the target {op_type_caps} that:\n"
            f"1. Identifies the document key with the longest text content\n"
            f"2. Keeps only the first 'head_words' words\n"
            f"3. Optionally keeps the last 'tail_words' words (default 0)\n"
            f"4. Returns the truncated content in the same key\n\n"
            f"Guidelines:\n"
            f"- Choose head_words based on how much context the task likely needs\n"
            f"- Set tail_words > 0 only if the task benefits from document endings (e.g., conclusions, signatures)\n"
            f"- For classification/filtering: typically 50-150 head_words, tail_words=0\n"
            f"- For summarization/reduction: typically 100-300 head_words, tail_words=50-100\n"
            f"- For metadata extraction: head=100-200, tail=50-100 (headers + footers)\n"
            f"- Identify the document_key by looking at {{{{ input.KEY }}}} references in the prompt\n\n"
            f"Operation Type Considerations:\n"
            f"- Map: Focus on what information is needed for transformation\n"
            f"- Filter: Focus on what information is needed for the boolean decision\n"
            f"- Reduce: Focus on what information is needed for aggregation/synthesis\n\n"
            f"Example configuration:\n"
            f"{self.example}\n\n"
            f"Please output only the TakeHeadTailInstantiateSchema object that specifies:\n"
            f"- name: descriptive name for the truncation operation\n"
            f"- document_key: the key containing text to truncate\n"
            f"- head_words: number of words to keep from start\n"
            f"- tail_words: number of words to keep from end (0 if not needed)"
        )

    def llm_instantiate(
        self,
        original_op: Dict,
        agent_llm: str,
        message_history: list = [],
    ):
        message_history.extend(
            [
                {
                    "role": "user",
                    "content": self.to_string_for_instantiate(original_op),
                },
            ]
        )

        for _ in range(MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS):
            call_cost = 0
            resp = completion(
                model=agent_llm,
                messages=message_history,
                api_key=os.environ.get("AZURE_API_KEY"),
                api_base=os.environ.get("AZURE_API_BASE"),
                api_version=os.environ.get("AZURE_API_VERSION"),
                azure=True,
                response_format=TakeHeadTailInstantiateSchema,
            )
            call_cost = resp._hidden_params["response_cost"]
            try:
                parsed_res = json.loads(resp.choices[0].message.content)
                schema = TakeHeadTailInstantiateSchema(**parsed_res)
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
        ops_list: List[Dict],
        target_op: str,
        rewrite: TakeHeadTailInstantiateSchema,
    ) -> List[Dict]:
        new_ops_list = deepcopy(ops_list)

        pos_to_replace = [
            i for i, op in enumerate(ops_list) if op["name"] == target_op
        ][0]

        head_words = rewrite.head_words
        tail_words = rewrite.tail_words
        document_key = rewrite.document_key

        code_map_function = f"""def transform(input_doc):
    {document_key}_content = input_doc.get('{document_key}', '')
    words = {document_key}_content.split()

    if len(words) <= {head_words + tail_words}:
        # Document is short enough, keep as is
        truncated = {document_key}_content
    else:
        head = ' '.join(words[:{head_words}])
        if {tail_words} > 0:
            tail = ' '.join(words[-{tail_words}:])
            truncated = head + ' ... ' + tail
        else:
            truncated = head

    return {{'{document_key}': truncated}}"""

        code_map_op = {
            "name": rewrite.name,
            "type": "code_map",
            "code": code_map_function,
        }

        new_ops_list.insert(pos_to_replace, code_map_op)

        return new_ops_list

    def instantiate(
        self,
        operators: List[Dict],
        target_ops: List[str],
        agent_llm: str,
        message_history: list = [],
        **kwargs,
    ):
        assert (
            len(target_ops) == 1
        ), "TakeHeadTail directive requires exactly one target op"

        target_op_config = [op for op in operators if op["name"] == target_ops[0]][0]

        rewrite, message_history, call_cost = self.llm_instantiate(
            target_op_config, agent_llm, message_history
        )

        return self.apply(operators, target_ops[0], rewrite), message_history, call_cost
