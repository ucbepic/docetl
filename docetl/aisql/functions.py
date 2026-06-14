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
