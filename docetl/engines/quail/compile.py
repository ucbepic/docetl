"""Compile a run of DocETL filter and map operations into one Quail SQL query.

Each operation becomes one common table expression (CTE) over the one before
it, so a later operation can read an earlier operation's output as a column,
and Quail plans the whole run as one query:

- ``filter`` adds ``WHERE AI.IF(PROMPT(...))``; its decision key is dropped,
  as the filter operation does.
- ``map`` adds one column for every item in the output schema. The item's type
  picks the AI function: ``bool`` to ``AI.IF``, ``enum`` to ``AI.CLASSIFY``,
  numbers to ``AI.MAP`` with ``TRY_CAST``, strings to ``AI.MAP``, and lists
  and objects to ``AI.MAP`` with a JSON instruction, parsed in DocETL.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from docetl.engines.quail.prompts import COLUMN_NAME, QuailPromptError, parse_prompt

ID_COLUMN = "_docetl_id"
SUPPORTED_TYPES = ("filter", "map")

# Operation settings with no Quail form: each needs follow-up LLM calls or
# whole-row context that one SQL query cannot express.
UNSUPPORTED_SETTINGS = (
    "gleaning",
    "validate",
    "agent",
    "tools",
    "pdf_url_key",
    "batch_prompt",
    "calibrate",
    "retriever",
    "cascade",
    "limit",
)


class QuailUnsupported(ValueError):
    """An operation the Quail engine cannot run."""


def engine_for(op_config: dict, pipeline_config: dict) -> dict | None:
    """Return the Quail engine settings for one operation, or None for LiteLLM.

    ``engine`` on the pipeline applies to every operation; ``engine`` on an
    operation overrides it: ``litellm`` opts out, ``quail`` opts in with the
    pipeline's settings, and a dict sets its own.
    """
    pipeline = pipeline_config.get("engine")
    if isinstance(pipeline, str):
        pipeline = {"type": pipeline}
    pipeline = pipeline or {}

    chosen = op_config.get("engine")
    if chosen is None:
        chosen = pipeline
    elif isinstance(chosen, str):
        chosen = {**pipeline, "type": chosen}
    else:
        chosen = {**pipeline, **chosen}

    if chosen.get("type", "litellm") == "litellm":
        return None
    if chosen["type"] != "quail":
        raise ValueError(f"Unknown engine {chosen['type']!r}; use 'quail' or 'litellm'")
    if not chosen.get("endpoint"):
        raise ValueError(
            "The Quail engine needs an endpoint, such as "
            "engine: {type: quail, endpoint: grpc://host:50051}"
        )
    fallback = chosen.get("fallback", "litellm")
    if fallback not in ("litellm", "strict"):
        raise ValueError(f"engine.fallback must be 'litellm' or 'strict', not {fallback!r}")
    return {"type": "quail", "endpoint": chosen["endpoint"], "fallback": fallback}


@dataclass(frozen=True)
class OutputColumn:
    """How DocETL turns one result column back into a row value."""

    name: str
    kind: str  # "string", "integer", "number", "boolean", "enum", "json"


def _item_kind(type_str: str) -> tuple[str, list[str]]:
    """Return (kind, enum labels) for one output schema type string."""
    text = type_str.strip()
    lowered = text.lower()
    if lowered in ("str", "text", "string", "varchar"):
        return "string", []
    if lowered in ("int", "integer"):
        return "integer", []
    if lowered in ("float", "decimal", "number"):
        return "number", []
    if lowered in ("bool", "boolean"):
        return "boolean", []
    if lowered.startswith("enum[") and lowered.endswith("]"):
        labels = [label.strip() for label in text[5:-1].split(",") if label.strip()]
        if not labels:
            raise QuailUnsupported(f"enum type {type_str!r} has no labels")
        return "enum", labels
    if lowered.startswith("list[") or (lowered.startswith("{") and lowered.endswith("}")):
        return "json", []
    raise QuailUnsupported(f"output type {type_str!r} has no Quail form")


def check_eligible(op_config: dict) -> None:
    """Raise QuailUnsupported when the Quail engine cannot run ``op_config``."""
    name = op_config.get("name", "?")
    if op_config.get("type") not in SUPPORTED_TYPES:
        raise QuailUnsupported(
            f"operation {name!r} has type {op_config.get('type')!r}; the Quail "
            f"engine runs {', '.join(SUPPORTED_TYPES)}"
        )
    for setting in UNSUPPORTED_SETTINGS:
        if op_config.get(setting):
            raise QuailUnsupported(
                f"operation {name!r} sets {setting!r}, which has no Quail form"
            )
    schema = (op_config.get("output") or {}).get("schema") or {}
    if not schema:
        raise QuailUnsupported(f"operation {name!r} has no output schema")
    for key, type_str in schema.items():
        if not COLUMN_NAME.match(key):
            raise QuailUnsupported(
                f"operation {name!r} writes {key!r}, which is not a plain column name"
            )
        kind, _ = _item_kind(str(type_str))
        if op_config["type"] == "filter" and kind != "boolean":
            raise QuailUnsupported(
                f"filter {name!r} must have one boolean output, not {type_str!r}"
            )
    if op_config["type"] == "filter" and len(schema) != 1:
        raise QuailUnsupported(f"filter {name!r} must have exactly one output")
    try:
        parse_prompt(op_config.get("prompt") or "")
    except QuailPromptError as error:
        raise QuailUnsupported(f"operation {name!r}: {error}") from error


def _lit(text: str) -> str:
    """A SQL string literal."""
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _col(alias: str, name: str) -> str:
    return f"{alias}.`{name}`"


@dataclass
class CompiledRun:
    """One Quail query for a run of operations.

    Attributes:
        sql: The query.
        input_fields: The row fields DocETL uploads (besides the id).
        outputs: How to turn each written column back into a row value.
        dropped: Fields the run removes from every row (filter keys,
            ``drop_keys``).
    """

    sql: str
    input_fields: list[str]
    outputs: dict[str, OutputColumn] = field(default_factory=dict)
    dropped: list[str] = field(default_factory=list)


def _map_columns(op: dict, alias: str, max_tokens: int | None) -> list[tuple[str, str, OutputColumn]]:
    prompt = parse_prompt(op["prompt"])
    schema = op["output"]["schema"]
    columns = [_col(alias, f) for f in prompt.fields]
    if len(columns) == 1:
        document = columns[0]
    else:
        document = "CONCAT(" + ", ' ', ".join(columns) + ")"
    args = ", ".join(columns)
    several = len(schema) > 1
    token_arg = f", {int(max_tokens)}" if max_tokens else ""

    built = []
    for key, type_str in schema.items():
        kind, labels = _item_kind(str(type_str))
        ask = f" Answer only the field {key}." if several else ""
        if kind == "boolean":
            expr = f"AI.IF(PROMPT({_lit(prompt.template + ask)}, {args}))"
        elif kind == "enum":
            label_list = ", ".join(_lit(label) for label in labels)
            expr = f"AI.CLASSIFY(PROMPT({_lit(prompt.template + ask)}, {args}), [{label_list}])"
        elif kind in ("integer", "number"):
            instruction = prompt.instruction() + ask + " Answer with a number only."
            expr = f"TRY_CAST(AI.MAP({document}, {_lit(instruction)}{token_arg}) AS DOUBLE)"
        elif kind == "json":
            instruction = prompt.instruction() + ask + f" Answer with JSON only, of type {type_str}."
            expr = f"AI.MAP({document}, {_lit(instruction)}{token_arg})"
        else:
            expr = f"AI.MAP({document}, {_lit(prompt.instruction() + ask)}{token_arg})"
        built.append((key, expr, OutputColumn(key, kind)))
    return built


def compile_run(ops: list[dict], table: str) -> CompiledRun:
    """Compile consecutive filter and map operations into one query over ``table``.

    Args:
        ops: The operation configs, each one passing ``check_eligible``.
        table: The name of the uploaded table, which has ``_docetl_id`` and
            each of the run's input fields.
    """
    for op in ops:
        check_eligible(op)

    written: set[str] = set()
    input_fields: list[str] = []
    for op in ops:
        for f in parse_prompt(op["prompt"]).fields:
            if f not in written and f not in input_fields:
                input_fields.append(f)
        written.update(op["output"]["schema"].keys())

    columns = [ID_COLUMN, *input_fields]
    ctes = [f"c0 AS (SELECT * FROM `{table}`)"]
    outputs: dict[str, OutputColumn] = {}
    dropped: list[str] = []

    for index, op in enumerate(ops, start=1):
        prev = f"c{index - 1}"
        if op["type"] == "filter":
            prompt = parse_prompt(op["prompt"])
            (key,) = op["output"]["schema"].keys()
            args = ", ".join(_col(prev, f) for f in prompt.fields)
            keep = [c for c in columns if c != key]
            select = ", ".join(f"{_col(prev, c)} AS `{c}`" for c in keep)
            ctes.append(
                f"c{index} AS (SELECT {select} FROM {prev} "
                f"WHERE AI.IF(PROMPT({_lit(prompt.template)}, {args})))"
            )
            columns = keep
            outputs.pop(key, None)
            dropped.append(key)
        else:
            max_tokens = (op.get("litellm_completion_kwargs") or {}).get("max_tokens")
            built = _map_columns(op, prev, max_tokens)
            new = [name for name, _, _ in built]
            keep = [c for c in columns if c not in new]
            parts = [f"{_col(prev, c)} AS `{c}`" for c in keep]
            parts += [f"{expr} AS `{name}`" for name, expr, _ in built]
            ctes.append(f"c{index} AS (SELECT {', '.join(parts)} FROM {prev})")
            columns = keep + new
            for name, _, out in built:
                outputs[name] = out
                if name in dropped:
                    dropped.remove(name)
        for key in op.get("drop_keys") or []:
            if key in columns and key != ID_COLUMN:
                columns.remove(key)
            outputs.pop(key, None)
            dropped.append(key)

    last = f"c{len(ops)}"
    select = ", ".join(f"{_col(last, c)} AS `{c}`" for c in columns)
    sql = "WITH " + ",\n".join(ctes) + f"\nSELECT {select} FROM {last}"
    return CompiledRun(sql, input_fields, outputs, dropped)


def convert(value: Any, out: OutputColumn) -> Any:
    """Turn one result cell into the row value DocETL writes."""
    if value is None:
        return None
    if out.kind == "integer":
        return int(round(float(value)))
    if out.kind == "number":
        return float(value)
    if out.kind == "boolean":
        return bool(value)
    if out.kind == "json":
        text = str(value).strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    return str(value)
