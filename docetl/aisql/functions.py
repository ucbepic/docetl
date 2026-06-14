"""The AI function surface and how each lowers to a DocETL operator.

v1 covers the SELECT-position functions that become a ``map``. Each entry
turns a parsed call (its argument column and any literal args) into a map
operation config plus the output column it produces.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AIFunctionCall:
    """A recognized AI function in a query: its name, the input column it
    reads, optional literal args (e.g. a prompt), and its output alias."""

    name: str
    column: str
    literals: tuple[str, ...]
    alias: str


def _summarize(call: AIFunctionCall) -> dict:
    return {
        "prompt": f"Summarize the following:\n\n{{{{ input.{call.column} }}}}",
        "schema": {call.alias: "string"},
    }


def _classify(call: AIFunctionCall) -> dict:
    if not call.literals:
        raise ValueError("ai_classify(column, 'instruction') needs an instruction")
    return {
        "prompt": f"{call.literals[0]}\n\n{{{{ input.{call.column} }}}}",
        "schema": {call.alias: "string"},
    }


def _generic(call: AIFunctionCall) -> dict:
    if not call.literals:
        raise ValueError("ai(column, 'prompt') needs a prompt")
    return {
        "prompt": f"{call.literals[0]}\n\n{{{{ input.{call.column} }}}}",
        "schema": {call.alias: "string"},
    }


# name -> builder returning {"prompt", "schema"} for a map op.
SELECT_FUNCTIONS = {
    "ai": _generic,
    "ai_summarize": _summarize,
    "ai_classify": _classify,
    "ai_extract": _generic,
}

# Recognized everywhere, so the compiler can detect (and reject, for now)
# AI functions outside the SELECT list.
ALL_FUNCTIONS = set(SELECT_FUNCTIONS) | {
    "ai_filter",
    "ai_agg",
    "ai_match",
    "ai_score",
    "ai_resolve",
}


def build_map_op(call: AIFunctionCall, name: str) -> dict:
    """A DocETL ``map`` op config for a SELECT-position AI function."""
    builder = SELECT_FUNCTIONS.get(call.name)
    if builder is None:
        raise NotImplementedError(
            f"AI function {call.name!r} is not yet supported in SELECT"
        )
    spec = builder(call)
    return {
        "name": name,
        "type": "map",
        "prompt": spec["prompt"],
        "output": {"schema": spec["schema"]},
    }


def build_compare_map_op(call: AIFunctionCall, name: str, output_type: str) -> dict:
    """A DocETL ``map`` for an AI function used inside a comparison
    (``ai_score(col, 'criteria') > 0.8``): it produces the value column
    that the relational comparison then filters on."""
    if not call.literals:
        raise ValueError(f"{call.name}(column, 'criteria') needs a criteria string")
    suffix = "\n\nRespond with a single number." if output_type == "number" else ""
    return {
        "name": name,
        "type": "map",
        "prompt": f"{call.literals[0]}\n\n{{{{ input.{call.column} }}}}{suffix}",
        "output": {"schema": {call.alias: output_type}},
    }


def build_reduce_op(call: AIFunctionCall, name: str, reduce_key: list[str]) -> dict:
    """A DocETL ``reduce`` for ``GROUP BY k ... ai_agg(col, 'instruction')``.
    The prompt folds the grouped rows (``{{ inputs }}``)."""
    if not call.literals:
        raise ValueError("ai_agg(column, 'instruction') needs an instruction")
    prompt = (
        f"{call.literals[0]}\n\n"
        f"{{% for item in inputs %}}{{{{ item.{call.column} }}}}\n{{% endfor %}}"
    )
    return {
        "name": name,
        "type": "reduce",
        "reduce_key": reduce_key,
        "prompt": prompt,
        "output": {"schema": {call.alias: "string"}},
    }


def build_equijoin_op(name: str, left_col: str, right_col: str, prompt: str) -> dict:
    """A DocETL ``equijoin`` for ``JOIN ... ON ai_match(l.col, r.col, 'q')``.
    The comparison prompt reads ``{{ left.col }}`` and ``{{ right.col }}``."""
    return {
        "name": name,
        "type": "equijoin",
        "comparison_prompt": (
            f"{prompt}\n\nLeft: {{{{ left.{left_col} }}}}\n"
            f"Right: {{{{ right.{right_col} }}}}"
        ),
    }


def build_resolve_op(name: str, column: str, prompt: str, output_key: str) -> dict:
    """A DocETL ``resolve`` for ``ai_resolve(table, on := col, prompt := q)``.
    Compares pairs on ``column`` and canonicalizes the matched value."""
    return {
        "name": name,
        "type": "resolve",
        "comparison_prompt": (
            f"{prompt}\n\nRecord 1: {{{{ input1.{column} }}}}\n"
            f"Record 2: {{{{ input2.{column} }}}}"
        ),
        "resolution_prompt": (
            f"Produce one canonical value for {column} from these records:\n"
            f"{{% for item in inputs %}}{{{{ item.{column} }}}}\n{{% endfor %}}"
        ),
        "output": {"schema": {output_key: "string"}},
    }


def build_filter_op(call: AIFunctionCall, name: str) -> dict:
    """A DocETL ``filter`` op for an ``ai_filter(column, 'question')``
    predicate in WHERE. The decision key is consumed by the filter, so it
    doesn't appear in the output rows."""
    if not call.literals:
        raise ValueError("ai_filter(column, 'question') needs a question")
    return {
        "name": name,
        "type": "filter",
        "prompt": f"{call.literals[0]}\n\n{{{{ input.{call.column} }}}}",
        "output": {"schema": {"keep": "boolean"}},
    }
