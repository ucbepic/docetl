"""Rewrite an operation's prompt into the form the Quail engine accepts.

The Quail engine (docetl/engines/quail) runs only prompts made of fixed text
and plain field references, because Quail keeps the question the same across
rows. This directive asks an LLM to rewrite any other prompt into that form,
checks the result with the engine's own parser, and retries with the parser's
error. It runs once, at config time, and ``docetl quail-rewrite`` shows the
user each rewrite to accept or edit; DocETL never rewrites a prompt at run
time.

Not in ``ALL_DIRECTIVES``: the MOAR search does not pick it, because the
rewrite targets an engine, not quality or cost.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Dict, List, Type

from litellm import completion
from pydantic import BaseModel, Field

from docetl.engines.quail.prompts import QuailPromptError, parse_prompt, parse_reduce_prompt

from .base import MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS, Directive


class QuailFormInstantiateSchema(BaseModel):
    """The rewritten prompt."""

    quail_prompt: str = Field(
        ...,
        description="The prompt in Quail form: fixed text with plain field references only.",
    )


def check_quail_form(op_type: str, prompt: str) -> None:
    """Raise QuailPromptError when ``prompt`` is not in Quail form."""
    if op_type == "reduce":
        parse_reduce_prompt(prompt)
    else:
        parse_prompt(prompt)


class QuailFormDirective(Directive):
    name: str = Field(default="quail_form", description="The name of the directive")
    formal_description: str = Field(default="Op => Op (Quail-form prompt)")
    nl_description: str = Field(
        default="Rewrites an operation's Jinja prompt into the form the Quail engine runs: fixed text with plain field references such as {{ input.body }}, and for reduce, fixed text around one {% for item in inputs %} loop over plain item fields."
    )
    when_to_use: str = Field(
        default="When a pipeline sets engine: quail and an operation's prompt uses filters, conditionals, loops, macros, or braces in its fixed text, so the operation would otherwise run on LiteLLM."
    )
    instantiate_schema_type: Type[BaseModel] = Field(default=QuailFormInstantiateSchema)
    example: str = Field(
        default="""
        Original: "{% if input.lang == 'fr' %}Translate first. {% endif %}Summarize {{ input.body | truncate(2000) }} as JSON like {\\"summary\\": ...}"
        Quail form: "Summarize the review {{ input.body }}. If the review is not in English, translate it first. Answer with the summary only."
        """
    )

    def __eq__(self, other):
        return isinstance(other, QuailFormDirective)

    def __hash__(self):
        return hash("QuailFormDirective")

    def to_string_for_instantiate(self, target_ops_configs: List[Dict], error: str | None = None) -> str:
        (op,) = target_ops_configs
        rules = (
            "Rules for the Quail form:\n"
            "- Use only fixed text and plain field references, such as {{ input.body }} "
            "or {{ input['title'] }}.\n"
            "- No Jinja filters (|), if blocks, inline ifs, macros, set statements, or "
            "other variables.\n"
            "- No braces { or } in the fixed text; describe any output format in words.\n"
            "- Keep every input field the original prompt reads, and keep its task. Move "
            "a row-dependent condition into the instruction in words.\n"
        )
        if op.get("type") == "reduce":
            rules += (
                "- This is a reduce: write fixed text around exactly one "
                "{% for item in inputs %} ... {% endfor %} loop, whose body holds only "
                "fixed text and plain item fields such as {{ item.body }}. No field "
                "references outside the loop.\n"
            )
        message = (
            "Rewrite this DocETL operation's prompt into the form the Quail engine runs.\n\n"
            f"Operation:\n{json.dumps(op, indent=2)}\n\n{rules}\nExample:\n{self.example}\n"
        )
        if error:
            message += f"\nYour last rewrite was refused: {error}\nFix it and answer again.\n"
        return message

    def llm_instantiate(self, target_ops_configs: List[Dict], agent_llm: str, **kwargs):
        """Ask ``agent_llm`` for a Quail-form prompt; retry with the parser's error.

        Returns:
            (QuailFormInstantiateSchema, cost in dollars)
        """
        (op,) = target_ops_configs
        error = None
        cost = 0.0
        for _ in range(MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS):
            response = completion(
                model=agent_llm,
                messages=[
                    {"role": "system", "content": "You rewrite prompts. Answer in JSON."},
                    {"role": "user", "content": self.to_string_for_instantiate([op], error)},
                ],
                response_format=QuailFormInstantiateSchema,
            )
            try:
                cost += response._hidden_params.get("response_cost") or 0.0
            except AttributeError:
                pass
            try:
                rewrite = QuailFormInstantiateSchema(
                    **json.loads(response.choices[0].message.content)
                )
                check_quail_form(op.get("type", "map"), rewrite.quail_prompt)
                return rewrite, cost
            except (QuailPromptError, ValueError, TypeError) as caught:
                error = str(caught)
        raise ValueError(
            f"Could not rewrite {op.get('name')!r} into Quail form after "
            f"{MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS} attempts: {error}"
        )

    def apply(
        self,
        global_default_model: str,
        ops_list: List[Dict],
        target_ops: List[str],
        rewrite: QuailFormInstantiateSchema,
    ) -> List[Dict]:
        new_ops = deepcopy(ops_list)
        for op in new_ops:
            if op["name"] in target_ops:
                op["prompt"] = rewrite.quail_prompt
        return new_ops

    def instantiate(
        self,
        operators: List[Dict],
        target_ops: List[str],
        agent_llm: str,
        message_history: list = [],
        global_default_model: str = None,
        **kwargs,
    ):
        assert len(target_ops) == 1, "quail_form rewrites one operation at a time"
        configs = [op for op in operators if op["name"] in target_ops]
        if not configs:
            raise ValueError(f"Target operation {target_ops[0]} not found in operators")
        rewrite, cost = self.llm_instantiate(configs, agent_llm)
        return self.apply(global_default_model, operators, target_ops, rewrite), message_history, cost
