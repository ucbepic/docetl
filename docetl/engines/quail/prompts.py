"""Check a Jinja prompt against the Quail form, and map it to columns.

Quail builds every model prompt as: a shared preamble, the document, the
question, and an answer cue. Quail reuses the preamble and the document across
all AI functions on one document, so the question must stay the same across
rows. A Quail operation therefore accepts only plain field references, such as
``{{ input.body }}``, with fixed text around them. Any other Jinja (filters,
loops, conditionals, macros) is refused with a message naming the construct.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from jinja2 import Environment, TemplateSyntaxError, nodes

COLUMN_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class QuailPromptError(ValueError):
    """A prompt with Jinja the Quail engine does not accept."""


@dataclass(frozen=True)
class QuailPrompt:
    """A prompt in Quail form.

    Attributes:
        template: The fixed text with ``{N}`` where field N goes, as
            Quail's ``PROMPT`` takes it.
        fields: The input fields, in placeholder order, each once.
    """

    template: str
    fields: tuple[str, ...]

    def instruction(self) -> str:
        """The fixed text with each field named in brackets, for ``AI.MAP``.

        ``AI.MAP`` reads one document and a fixed instruction with no braces,
        so the placeholders become field names.
        """
        text = self.template
        for index, field in enumerate(self.fields):
            text = text.replace("{" + str(index) + "}", f"[{field}]")
        return " ".join(text.split())


def _field(node: nodes.Node, variable: str) -> str | None:
    """Return the field name of ``input.x`` / ``input['x']``, else None."""
    if (
        isinstance(node, nodes.Getattr)
        and isinstance(node.node, nodes.Name)
        and node.node.name == variable
    ):
        return node.attr
    if (
        isinstance(node, nodes.Getitem)
        and isinstance(node.node, nodes.Name)
        and node.node.name == variable
        and isinstance(node.arg, nodes.Const)
        and isinstance(node.arg.value, str)
    ):
        return node.arg.value
    return None


def _describe(node: nodes.Node) -> str:
    names = {
        nodes.Filter: "a filter (|)",
        nodes.For: "a for loop",
        nodes.If: "an if block",
        nodes.CondExpr: "an inline if",
        nodes.Macro: "a macro",
        nodes.Call: "a function call",
        nodes.Assign: "a set statement",
        nodes.Include: "an include",
        nodes.Name: "a variable other than input",
    }
    for kind, name in names.items():
        if isinstance(node, kind):
            return name
    return f"Jinja construct {type(node).__name__}"


def parse_prompt(template: str, variable: str = "input") -> QuailPrompt:
    """Parse ``template`` into Quail form.

    Args:
        template: The operation's Jinja prompt.
        variable: The row variable the prompt reads (``input`` for map and
            filter).

    Raises:
        QuailPromptError: The prompt uses Jinja beyond plain field
            references, a field name Quail cannot use as a column, or braces
            in its fixed text.
    """
    try:
        body = Environment().parse(template).body
    except TemplateSyntaxError as error:
        raise QuailPromptError(f"the prompt is not valid Jinja: {error}") from error

    pieces: list[str] = []
    fields: list[str] = []
    for statement in body:
        if not isinstance(statement, nodes.Output):
            raise QuailPromptError(
                f"the Quail engine accepts only plain field references such as "
                f"{{{{ {variable}.body }}}}; the prompt uses {_describe(statement)}"
            )
        for child in statement.nodes:
            if isinstance(child, nodes.TemplateData):
                if "{" in child.data or "}" in child.data:
                    raise QuailPromptError(
                        "the prompt's fixed text has a brace, which Quail's "
                        "PROMPT and AI.MAP do not accept"
                    )
                pieces.append(child.data)
                continue
            field = _field(child, variable)
            if field is None:
                raise QuailPromptError(
                    f"the Quail engine accepts only plain field references such "
                    f"as {{{{ {variable}.body }}}}; the prompt uses {_describe(child)}"
                )
            if not COLUMN_NAME.match(field):
                raise QuailPromptError(
                    f"field {field!r} is not a plain column name (letters, "
                    f"digits, and underscores)"
                )
            if field not in fields:
                fields.append(field)
            pieces.append("{" + str(fields.index(field)) + "}")

    if not fields:
        raise QuailPromptError(
            f"the prompt reads no field; Quail needs at least one {{{{ {variable}.x }}}}"
        )
    return QuailPrompt("".join(pieces).strip(), tuple(fields))
