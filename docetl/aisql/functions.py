"""The AI function surface and how each lowers to a DocETL operator.

Each builder turns a parsed call (its argument column and any literal
args) into a DocETL operation config. The shared helpers below keep the
op-config envelope and the ``input.col`` template fragment in one place.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AIFunctionCall:
    """A recognized AI function in a query: its name, the input column it
    reads, optional literal args (e.g. a prompt), and its output alias."""

    name: str
    column: str
    literals: tuple[str, ...]
    alias: str
    negated: bool = False


def _doc(call: AIFunctionCall) -> str:
    """The Jinja fragment that renders the call's input column."""
    return f"{{{{ input.{call.column} }}}}"


def _require_literal(call: AIFunctionCall) -> str:
    """The call's first literal arg (the instruction/prompt), or raise."""
    if not call.literals:
        raise ValueError(f"{call.name}(column, '...') needs an instruction string")
    return call.literals[0]


def _map_op(name: str, prompt: str, schema: dict) -> dict:
    return {"name": name, "type": "map", "prompt": prompt, "output": {"schema": schema}}


_TYPE_MAP = {
    # DuckDB text types
    "varchar": "string",
    "text": "string",
    "string": "string",
    "str": "string",
    # DuckDB integer types
    "int": "integer",
    "integer": "integer",
    "bigint": "integer",
    "smallint": "integer",
    "tinyint": "integer",
    "hugeint": "integer",
    # DuckDB float types
    "float": "number",
    "double": "number",
    "real": "number",
    "decimal": "number",
    "number": "number",
    "numeric": "number",
    # DuckDB boolean
    "boolean": "boolean",
    "bool": "boolean",
    # list
    "list": "list",
    "array": "list",
}


def _parse_schema_spec(spec: str) -> dict[str, str]:
    """Parse ``'key1:TYPE, key2:TYPE'`` into a DocETL output schema dict.

    Types are DuckDB names (VARCHAR, INT, DOUBLE, BOOLEAN, etc.) mapped
    to DocETL schema types (string, integer, number, boolean, list).
    A bare key without a type defaults to string.
    """
    schema = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            key, typ = part.split(":", 1)
            schema[key.strip()] = _TYPE_MAP.get(
                typ.strip().lower(), typ.strip().lower()
            )
        else:
            schema[part] = "string"
    return schema


def _output_schema(call: AIFunctionCall) -> dict[str, str]:
    """Schema from the call's second literal, or ``{alias: 'string'}``."""
    if len(call.literals) >= 2:
        return _parse_schema_spec(call.literals[1])
    return {call.alias: "string"}


def _select_spec(call: AIFunctionCall, preamble: str) -> dict:
    return {"prompt": f"{preamble}\n\n{_doc(call)}", "schema": _output_schema(call)}


def _summarize(call: AIFunctionCall) -> dict:
    return _select_spec(call, "Summarize the following:")


def _prompted(call: AIFunctionCall) -> dict:
    return _select_spec(call, _require_literal(call))


def _extract_output_col(call: AIFunctionCall) -> str:
    """The column name that a DocETL extract op will produce."""
    return f"{call.column}_extracted"


# name -> builder returning {"prompt", "schema"} for a map op.
SELECT_MAP_FUNCTIONS = {
    "ai": _prompted,
    "ai_map": _prompted,
    "ai_summarize": _summarize,
    "ai_classify": _prompted,
}

EXTRACT_FUNCTIONS = {"ai_extract"}

# Recognized everywhere, so the compiler can detect (and reject, for now)
# AI functions outside the SELECT list.
ALL_FUNCTIONS = (
    set(SELECT_MAP_FUNCTIONS)
    | EXTRACT_FUNCTIONS
    | {
        "ai_filter",
        "ai_agg",
        "ai_match",
        "ai_score",
        "ai_resolve",
    }
)


def build_map_op(call: AIFunctionCall, name: str) -> dict:
    """A DocETL ``map`` op config for a SELECT-position AI function."""
    builder = SELECT_MAP_FUNCTIONS.get(call.name)
    if builder is None:
        raise NotImplementedError(
            f"AI function {call.name!r} is not yet supported in SELECT"
        )
    spec = builder(call)
    return _map_op(name, spec["prompt"], spec["schema"])


def build_extract_op(call: AIFunctionCall, name: str) -> dict:
    """A DocETL ``extract`` op for ``ai_extract(col, 'what to find') AS alias``.

    Uses the line-number strategy to pull verbatim passages from the text.
    Output column is ``{col}_extracted``; the compiler renames it to the
    SQL alias in the final projection.
    """
    return {
        "name": name,
        "type": "extract",
        "document_keys": [call.column],
        "prompt": _require_literal(call),
        "extraction_key_suffix": "_extracted",
    }


def build_compare_map_op(call: AIFunctionCall, name: str, output_type: str) -> dict:
    """A DocETL ``map`` for an AI function used inside a comparison
    (``ai_score(col, 'criteria') > 0.8``): it produces the value column
    that the relational comparison then filters on."""
    suffix = "\n\nRespond with a single number." if output_type == "number" else ""
    prompt = f"{_require_literal(call)}\n\n{_doc(call)}{suffix}"
    return _map_op(name, prompt, {call.alias: output_type})


def build_reduce_op(call: AIFunctionCall, name: str, reduce_key: list[str]) -> dict:
    """A DocETL ``reduce`` for ``GROUP BY k ... ai_agg(col, 'instruction')``.
    The prompt folds the grouped rows (``{{ inputs }}``)."""
    prompt = (
        f"{_require_literal(call)}\n\n"
        f"{{% for item in inputs %}}{{{{ item.{call.column} }}}}\n{{% endfor %}}"
    )
    return {
        "name": name,
        "type": "reduce",
        "reduce_key": reduce_key,
        "prompt": prompt,
        "output": {"schema": _output_schema(call)},
    }


def build_filter_op(call: AIFunctionCall, name: str) -> dict:
    """A DocETL ``filter`` op for an ``ai_filter(column, 'question')``
    predicate in WHERE. The decision key is consumed by the filter, so it
    doesn't appear in the output rows.

    When ``call.negated`` is True (``NOT ai_filter(...)``), the prompt
    asks the LLM to negate the question so that rows matching the
    original question are filtered out."""
    prompt = _require_literal(call)
    if call.negated:
        prompt = f"Answer the OPPOSITE of the following question (return true if the answer to the original question is false, and vice versa):\n\n{prompt}"
    return {
        "name": name,
        "type": "filter",
        "prompt": f"{prompt}\n\n{_doc(call)}",
        "output": {"schema": {"keep": "boolean"}},
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
