"""Compile a run of DocETL filter, map, and reduce operations into one Quail query.

Each operation becomes one common table expression (CTE) over the one before
it, so a later operation can read an earlier operation's output as a column,
and Quail plans the whole run as one query:

- ``filter`` adds ``WHERE AI.IF(PROMPT(...))``; its decision key is dropped,
  as the filter operation does.
- ``map`` adds one column for every item in the output schema. The item's type
  picks the AI function: ``bool`` to ``AI.IF``, ``enum`` to ``AI.CLASSIFY``,
  numbers to ``AI.MAP`` with ``TRY_CAST``, strings to ``AI.MAP``, and lists
  and objects to ``AI.MAP`` with a JSON instruction, parsed in DocETL.
- ``reduce`` groups by ``reduce_key`` and runs the same AI functions over the
  group's items, joined with ``STRING_AGG`` in row order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from docetl.engines.quail.prompts import (
    COLUMN_NAME,
    QuailPromptError,
    parse_prompt,
    parse_reduce_prompt,
)

ID_COLUMN = "_docetl_id"
# Lists and objects come back as JSON text; Quail's AI.MAP default of 256
# tokens cuts them off, and a cut-off answer does not parse.
DEFAULT_JSON_MAX_TOKENS = 4096
SUPPORTED_TYPES = ("filter", "map", "parallel_map", "extract", "reduce")
DEFAULT_MAX_GROUP_TOKENS = 24_000

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
    # reduce: fold and merge, sampling, and pass-through need more than one
    # call per group
    "fold_prompt",
    "merge_prompt",
    "value_sampling",
)


class QuailUnsupported(ValueError):
    """An operation the Quail engine cannot run."""


QUAIL_MODEL_PREFIX = "quail/"


def engine_for(op_config: dict, pipeline_config: dict) -> dict | None:
    """Return the Quail engine settings for one operation, or None for LiteLLM.

    Two ways pick the Quail engine:

    - A model name ``quail/<model>`` (the operation's ``model``, else the
      pipeline's ``default_model``) runs the operation on the Quail Server
      that ``quail_models.<model>.endpoint`` names. Such an operation cannot
      run on LiteLLM, so it fails when it has no Quail form. Optimizers such
      as MOAR pick these names like any other model.
    - ``engine`` on the pipeline applies to every operation; ``engine`` on an
      operation overrides it: ``litellm`` opts out, ``quail`` opts in with
      the pipeline's settings, and a dict sets its own.

    ``gpu_cost_per_hour`` (on ``engine`` or a ``quail_models`` entry) prices
    Quail Server's time, so cost comparisons with hosted models are fair.
    """
    model = op_config.get("model") or pipeline_config.get("default_model") or ""
    if isinstance(model, str) and model.startswith(QUAIL_MODEL_PREFIX):
        name = model[len(QUAIL_MODEL_PREFIX):]
        served = (pipeline_config.get("quail_models") or {}).get(name)
        if not served or not served.get("endpoint"):
            raise ValueError(
                f"Model {model!r} needs a Quail Server: set quail_models: "
                f"{{{name}: {{endpoint: grpc://host:50051}}}} in the pipeline"
            )
        return {
            "type": "quail",
            "endpoint": served["endpoint"],
            "fallback": "strict",
            "max_group_tokens": int(served.get("max_group_tokens", DEFAULT_MAX_GROUP_TOKENS)),
            "gpu_cost_per_hour": float(served.get("gpu_cost_per_hour", 0.0)),
            "model": name,
        }

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
    return {
        "type": "quail",
        "endpoint": chosen["endpoint"],
        "fallback": fallback,
        "max_group_tokens": int(chosen.get("max_group_tokens", DEFAULT_MAX_GROUP_TOKENS)),
        "gpu_cost_per_hour": float(chosen.get("gpu_cost_per_hour", 0.0)),
    }


@dataclass(frozen=True)
class OutputColumn:
    """How DocETL turns one result column back into a row value."""

    name: str
    kind: str  # "string", "integer", "number", "boolean", "enum", "json", "key"


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


def _split_top(text: str) -> list[str]:
    """Split on commas outside brackets and braces."""
    parts, depth, current = [], 0, ""
    for char in text:
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    if current.strip():
        parts.append(current)
    return [part.strip() for part in parts]


def describe_type(type_str: str) -> str:
    """Describe a DocETL output type in words, with no braces.

    AI.MAP refuses an instruction with braces, so a JSON item's type, such as
    ``list[{clause_type: string, text_span: string}]``, is spelled out:
    "a list of objects with fields clause_type (string) and text_span (string)".
    """
    text = type_str.strip()
    lowered = text.lower()
    if lowered.startswith("list[") and text.endswith("]"):
        inner = describe_type(text[5:-1])
        if inner.startswith("an object "):
            return "a list of objects " + inner[len("an object "):]
        return f"a list of {inner} values"
    if text.startswith("{") and text.endswith("}"):
        fields = []
        for part in _split_top(text[1:-1]):
            name, _, value = part.partition(":")
            fields.append(f"{name.strip()} ({describe_type(value)})")
        joined = ", ".join(fields[:-1]) + (" and " if len(fields) > 1 else "") + fields[-1]
        return f"an object with fields {joined}"
    if lowered.startswith("enum[") and text.endswith("]"):
        return "one of " + ", ".join(label.strip() for label in text[5:-1].split(","))
    return text


def reduce_keys(op_config: dict) -> list[str]:
    """The reduce's group keys; ``_all`` (one group) gives none."""
    keys = op_config.get("reduce_key")
    keys = [keys] if isinstance(keys, str) else list(keys or [])
    return [k for k in keys if k != "_all"]


def check_eligible(op_config: dict) -> None:
    """Raise QuailUnsupported when the Quail engine cannot run ``op_config``."""
    name = op_config.get("name", "?")
    kind = op_config.get("type")
    if kind not in SUPPORTED_TYPES:
        raise QuailUnsupported(
            f"operation {name!r} has type {kind!r}; the Quail engine runs "
            f"{', '.join(SUPPORTED_TYPES)}"
        )
    for setting in UNSUPPORTED_SETTINGS:
        if op_config.get(setting):
            raise QuailUnsupported(
                f"operation {name!r} sets {setting!r}, which has no Quail form"
            )
    if kind == "extract":
        _check_extract(op_config)
        return
    schema = (op_config.get("output") or {}).get("schema") or {}
    if not schema:
        raise QuailUnsupported(f"operation {name!r} has no output schema")
    for key, type_str in schema.items():
        if not COLUMN_NAME.match(key):
            raise QuailUnsupported(
                f"operation {name!r} writes {key!r}, which is not a plain column name"
            )
        item_kind, _ = _item_kind(str(type_str))
        if kind == "filter" and item_kind != "boolean":
            raise QuailUnsupported(
                f"filter {name!r} must have one boolean output, not {type_str!r}"
            )
    if kind == "filter" and len(schema) != 1:
        raise QuailUnsupported(f"filter {name!r} must have exactly one output")
    if kind == "reduce":
        if not op_config.get("reduce_key"):
            raise QuailUnsupported(f"reduce {name!r} has no reduce_key")
        for key in reduce_keys(op_config):
            if not COLUMN_NAME.match(key):
                raise QuailUnsupported(
                    f"reduce {name!r} groups by {key!r}, which is not a plain column name"
                )
    if kind == "parallel_map":
        prompts = op_config.get("prompts") or []
        if not prompts:
            raise QuailUnsupported(f"parallel_map {name!r} has no prompts")
        for prompt_config in prompts:
            if prompt_config.get("model"):
                raise QuailUnsupported(
                    f"parallel_map {name!r}: a prompt with its own model has no Quail form"
                )
            if set(prompt_config.get("output_keys") or []) - set(schema):
                raise QuailUnsupported(
                    f"parallel_map {name!r}: output_keys must be in the output schema"
                )
            try:
                parse_prompt(prompt_config.get("prompt") or "")
            except QuailPromptError as error:
                raise QuailUnsupported(f"parallel_map {name!r}: {error}") from error
        return
    try:
        if kind == "reduce":
            parse_reduce_prompt(op_config.get("prompt") or "")
        else:
            parse_prompt(op_config.get("prompt") or "")
    except QuailPromptError as error:
        raise QuailUnsupported(f"operation {name!r}: {error}") from error


def _extract_suffix(op_config: dict) -> str:
    return op_config.get("extraction_key_suffix") or f"_extracted_{op_config.get('name', '')}"


def _check_extract(op_config: dict) -> None:
    """An extract runs as AI_EXTRACT: its prompt is the fixed question."""
    name = op_config.get("name", "?")
    keys = op_config.get("document_keys") or []
    if not keys:
        raise QuailUnsupported(f"extract {name!r} has no document_keys")
    for key in keys:
        for column in (key, f"{key}{_extract_suffix(op_config)}"):
            if not COLUMN_NAME.match(column):
                raise QuailUnsupported(f"extract {name!r}: {column!r} is not a plain column name")
    try:
        parse_prompt(op_config.get("prompt") or "")
    except QuailPromptError as error:
        if "reads no field" in str(error):
            return  # fixed text: the question AI_EXTRACT asks of each document
        raise QuailUnsupported(f"extract {name!r}: {error}") from error
    raise QuailUnsupported(
        f"extract {name!r}: an AI_EXTRACT question is fixed text, but the prompt reads fields"
    )


SQL_TYPES = ("split", "gather")
WHITESPACE = "CONCAT(' ', CHR(9), CHR(10), CHR(11), CHR(12), CHR(13))"


def check_sql_eligible(op_config: dict) -> None:
    """Raise QuailUnsupported unless ``op_config`` is a split or gather that
    compiles to plain SQL with DocETL's exact output.

    - ``split`` with ``method: delimiter``. (``token_count`` needs a tokenizer
      in SQL, so it runs in DocETL.)
    - ``gather`` whose peripheral sections use only ``count`` (head, tail,
      middle), without ``doc_header_key`` or per-section ``content_key``.
    """
    name = op_config.get("name", "?")
    kind = op_config.get("type")
    if kind == "split":
        if op_config.get("method") != "delimiter":
            raise QuailUnsupported(f"split {name!r}: only method delimiter compiles to SQL")
        kwargs = op_config.get("method_kwargs") or {}
        if not isinstance(kwargs.get("delimiter"), str) or not kwargs["delimiter"]:
            raise QuailUnsupported(f"split {name!r} needs a delimiter")
        if int(kwargs.get("num_splits_to_group", 1)) < 1:
            raise QuailUnsupported(f"split {name!r}: num_splits_to_group must be at least 1")
        names = [op_config.get("split_key", ""), f"{op_config.get('split_key')}_chunk",
                 f"{name}_id", f"{name}_chunk_num"]
    elif kind == "gather":
        if op_config.get("doc_header_key"):
            raise QuailUnsupported(f"gather {name!r}: doc_header_key does not compile to SQL")
        peripheral = op_config.get("peripheral_chunks") or {}
        for side, sections in peripheral.items():
            if side not in ("previous", "next") or not isinstance(sections, dict):
                raise QuailUnsupported(f"gather {name!r}: unknown peripheral side {side!r}")
            for section, settings in sections.items():
                if section not in ("head", "tail", "middle") or set(settings or {}) - {"count"}:
                    raise QuailUnsupported(
                        f"gather {name!r}: only head, tail, and middle counts compile to SQL"
                    )
        names = [op_config.get(k, "") for k in ("content_key", "doc_id_key", "order_key")]
        names.append(f"{op_config.get('content_key')}_rendered")
    else:
        raise QuailUnsupported(f"operation {name!r} of type {kind!r} has no SQL form")
    for column in names:
        if not COLUMN_NAME.match(str(column)):
            raise QuailUnsupported(f"operation {name!r}: {column!r} is not a plain column name")


def _sql_reads(op: dict) -> list[str]:
    if op["type"] == "split":
        return [op["split_key"]]
    return [op["content_key"], op["doc_id_key"], op["order_key"]]


def _sql_writes(op: dict) -> dict[str, str]:
    if op["type"] == "split":
        return {
            f"{op['split_key']}_chunk": "string",
            f"{op['name']}_id": "string",
            f"{op['name']}_chunk_num": "integer",
        }
    return {f"{op['content_key']}_rendered": "string"}


def _split_ctes(op: dict, prev: str, index: int, columns: list[str], id_in: str, id_out: str) -> list[str]:
    """CTEs for a delimiter split, matching SplitOperation.

    The split pieces are numbered, whitespace-only pieces dropped, and every
    num_splits_to_group pieces joined with the delimiter and stripped. Each
    chunk row's id, in ``id_out``, is <parent id>.<chunk number, zero-padded>,
    so ids stay unique and keep row order. ``id_out`` is a new column: Quail
    carries the uploaded table's ``_docetl_id`` through the query to track
    documents, so the split must not redefine it.
    """
    key, name = op["split_key"], op["name"]
    delimiter = _lit(op["method_kwargs"]["delimiter"])
    group = int(op["method_kwargs"].get("num_splits_to_group", 1))
    carried = [c for c in columns if c not in (id_in, ID_COLUMN)]
    p, n, f, r = (f"c{index}_{x}" for x in ("p", "n", "f", "r"))
    # "/" divides as decimals in this dialect: group numbers come from floor
    # over doubles, cast back to integers, computed as a column so GROUP BY
    # groups on it.
    g = f"c{index}_g"
    chunk = f"{g}.`__chunk`"
    piece = f"array_element({n}.`__parts`, {n}.`__i`)"
    keep = ", ".join(f"MIN(c{index}_g.`{c}`) AS `{c}`" for c in carried)
    return [
        f"{p} AS (SELECT {prev}.*, string_to_array({_col(prev, key)}, {delimiter}) AS `__parts` FROM {prev})",
        f"{n} AS (SELECT {p}.*, UNNEST(generate_series(CAST(1 AS BIGINT), "
        f"CAST(array_length({p}.`__parts`) AS BIGINT))) AS `__i` FROM {p})",
        f"{f} AS (SELECT {n}.`{id_in}` AS `{id_in}`, "
        + "".join(f"{n}.`{c}` AS `{c}`, " for c in carried)
        + f"{n}.`__i` AS `__i`, {piece} AS `__piece` FROM {n} "
        f"WHERE btrim({piece}, {WHITESPACE}) <> '')",
        f"{r} AS (SELECT {f}.*, CAST(ROW_NUMBER() OVER (PARTITION BY {f}.`{id_in}` "
        f"ORDER BY {f}.`__i`) AS BIGINT) AS `__k` FROM {f})",
        f"{g} AS (SELECT {r}.*, CAST(floor(CAST({r}.`__k` - 1 AS DOUBLE) / {group}) AS BIGINT) + 1 "
        f"AS `__chunk` FROM {r})",
        # Group into internal names first; the planner refuses an output
        # column named like a grouped column.
        f"c{index}_s AS (SELECT {g}.`{id_in}` AS `__parent`, {chunk} AS `__chunkno`, "
        + (keep + ", " if keep else "")
        + f"btrim(STRING_AGG({g}.`__piece`, {delimiter} ORDER BY {g}.`__k`), {WHITESPACE}) "
        f"AS `__text` FROM {g} GROUP BY {g}.`{id_in}`, {chunk})",
        f"c{index} AS (SELECT CONCAT(c{index}_s.`__parent`, '.', "
        f"lpad(CAST(c{index}_s.`__chunkno` AS VARCHAR), 6, '0')) AS `{id_out}`, "
        + "".join(f"c{index}_s.`{c}` AS `{c}`, " for c in carried)
        + f"c{index}_s.`__text` AS `{key}_chunk`, c{index}_s.`__parent` AS `{name}_id`, "
        f"c{index}_s.`__chunkno` AS `{name}_chunk_num` FROM c{index}_s)",
    ]


def _gather_side(lab: str, cum: str, start: str, end: str, settings: dict) -> str:
    """The text of one peripheral side, as GatherOperation renders it.

    ``start``/``end`` bound the side's chunks (1-based positions in the
    document, inclusive); head is the first ``head`` of them, tail the last
    ``tail``, and the rest is shown (middle) or replaced by a skipped-chars
    note.
    """
    head = int((settings.get("head") or {}).get("count", 0))
    tail = int((settings.get("tail") or {}).get("count", 0))
    total = f"({end} - {start} + 1)"
    if "middle" in settings:
        return f"array_to_string(array_slice({lab}, {start}, {end}), CHR(10))"

    def cum_at(pos: str) -> str:
        return f"(CASE WHEN {pos} <= 0 THEN 0 ELSE array_element({cum}, {pos}) END)"

    head_end = f"({start} - 1 + LEAST({head}, {total}))"
    tail_start = f"({start} + GREATEST({head}, {total} - {tail}))"
    skip_end = f"({end} - {tail})"
    skip_start = f"({start} + {head})"
    head_text = f"array_to_string(array_slice({lab}, {start}, {head_end}), CHR(10))"
    tail_text = f"array_to_string(array_slice({lab}, {tail_start}, {end}), CHR(10))"
    skipped = f"({cum_at(skip_end)} - {cum_at(f'({skip_start} - 1)')})"
    skip_text = (
        f"CASE WHEN {total} - {tail} > {head} THEN CONCAT('[... ', CAST({skipped} AS VARCHAR), "
        f"' characters skipped ...]') END"
    )
    return (
        f"concat_ws(CHR(10), NULLIF({head_text}, ''), {skip_text}, NULLIF({tail_text}, ''))"
    )


def _gather_ctes(op: dict, prev: str, index: int) -> list[str]:
    """CTEs for a gather, matching GatherOperation's rendering."""
    content, doc, order = op["content_key"], op["doc_id_key"], op["order_key"]
    peripheral = op.get("peripheral_chunks") or {}
    out = f"{content}_rendered"
    if not peripheral:
        return [f"c{index} AS (SELECT {prev}.*, {_col(prev, content)} AS `{out}` FROM {prev})"]
    w, a = f"c{index}_w", f"c{index}_a"
    part = f"PARTITION BY {_col(prev, doc)} ORDER BY TRY_CAST({_col(prev, order)} AS DOUBLE), {_col(prev, order)}"
    label = (
        f"CONCAT('[Chunk ', CAST({_col(prev, order)} AS VARCHAR), ']', CHR(10), {_col(prev, content)})"
    )
    start = op.get("main_chunk_start", "--- Begin Main Chunk ---")
    end = op.get("main_chunk_end", "--- End Main Chunk ---")
    previous = _gather_side(f"{a}.`__lab`", f"{a}.`__cums`", "1", f"({a}.`__pos` - 1)",
                            peripheral.get("previous") or {})
    following = _gather_side(f"{a}.`__lab`", f"{a}.`__cums`", f"({a}.`__pos` + 1)",
                             f"{a}.`__n`", peripheral.get("next") or {})
    rendered = (
        f"concat_ws(CHR(10), '--- Previous Context ---', NULLIF({previous}, ''), "
        f"CONCAT('--- End Previous Context ---', CHR(10)), {_lit(start)}, {a}.`{content}`, "
        f"{_lit(end)}, CONCAT(CHR(10), '--- Next Context ---'), NULLIF({following}, ''), "
        f"'--- End Next Context ---')"
    )
    return [
        f"{w} AS (SELECT {prev}.*, CAST(ROW_NUMBER() OVER ({part}) AS BIGINT) AS `__pos`, "
        f"CAST(COUNT(*) OVER (PARTITION BY {_col(prev, doc)}) AS BIGINT) AS `__n`, "
        f"CAST(SUM(char_length({_col(prev, content)})) OVER ({part} ROWS BETWEEN UNBOUNDED PRECEDING "
        f"AND CURRENT ROW) AS BIGINT) AS `__cum`, {label} AS `__label` FROM {prev})",
        f"{a} AS (SELECT {w}.*, ARRAY_AGG({w}.`__label`) OVER (PARTITION BY {w}.`{doc}` ORDER BY "
        f"{w}.`__pos` ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS `__lab`, "
        f"ARRAY_AGG({w}.`__cum`) OVER (PARTITION BY {w}.`{doc}` ORDER BY {w}.`__pos` "
        f"ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS `__cums` FROM {w})",
        f"c{index} AS (SELECT {a}.*, {rendered} AS `{out}` FROM {a})",
    ]


def _instruction(text: str) -> str:
    """An AI.MAP instruction: AI.MAP refuses braces, so none get through."""
    return text.replace("{", "(").replace("}", ")")


def _lit(text: str) -> str:
    """A SQL string literal."""
    escaped = (
        text.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return "'" + escaped + "'"


NEWLINE = "'\\n'"  # a SQL literal holding one newline


def _col(alias: str, name: str) -> str:
    return f"{alias}.`{name}`"


@dataclass
class CompiledRun:
    """One Quail query for a run of operations.

    Attributes:
        sql: The query.
        input_fields: The row fields DocETL uploads (besides the id).
        outputs: How to turn each result column back into a row value.
        dropped: Fields the run removes from every row (filter keys,
            ``drop_keys``); used when rows map back to input rows.
        grouped: Whether the run has a reduce, so result rows are groups
            rather than input rows.
        group_reads: For the run's first reduce, (keys, item fields) when all
            of them are uploaded fields, so DocETL can size the groups
            before the query; else None.
    """

    sql: str
    input_fields: list[str]
    outputs: dict[str, OutputColumn] = field(default_factory=dict)
    dropped: list[str] = field(default_factory=list)
    grouped: bool = False
    group_reads: tuple[list[str], list[str]] | None = None
    id_column: str = ID_COLUMN


def _ai_columns(
    schema: dict,
    template: str,
    args: list[str],
    document: str,
    instruction: str,
    max_tokens: int | None,
) -> list[tuple[str, str, OutputColumn]]:
    """One AI function per output schema item."""
    arg_list = ", ".join(args)
    several = len(schema) > 1
    token_arg = f", {int(max_tokens)}" if max_tokens else ""
    built = []
    for key, type_str in schema.items():
        kind, labels = _item_kind(str(type_str))
        ask = f" Answer only the field {key}." if several else ""
        if kind == "boolean":
            # AI.IF answers yes or no, so a boolean item needs a question.
            question = f" Is this true: {key.replace('_', ' ')}?" if several else ""
            expr = f"AI.IF(PROMPT({_lit(template + question)}, {arg_list}))"
        elif kind == "enum":
            label_list = ", ".join(_lit(label) for label in labels)
            expr = f"AI.CLASSIFY(PROMPT({_lit(template + ask)}, {arg_list}), [{label_list}])"
        elif kind in ("integer", "number"):
            text = instruction + ask + " Answer with a number only."
            expr = f"TRY_CAST(AI.MAP({document}, {_lit(_instruction(text))}{token_arg}) AS DOUBLE)"
        elif kind == "json":
            text = instruction + ask + f" Answer with JSON only: {describe_type(str(type_str))}."
            json_tokens = f", {int(max_tokens or DEFAULT_JSON_MAX_TOKENS)}"
            expr = f"AI.MAP({document}, {_lit(_instruction(text))}{json_tokens})"
        else:
            expr = f"AI.MAP({document}, {_lit(_instruction(instruction + ask))}{token_arg})"
        built.append((key, expr, OutputColumn(key, kind)))
    return built


def _row_columns(op: dict, prev: str, schema: dict) -> list[tuple[str, str, OutputColumn]]:
    """The AI columns of a map, parallel_map, or extract over ``prev``."""
    if op["type"] == "extract":
        question = _lit(_instruction(" ".join(op["prompt"].split())))
        kind = "string" if op.get("format_extraction", True) else "string_list"
        built = []
        for key in op["document_keys"]:
            # AI_EXTRACT copies the answer from the document (the AI.EXTRACT
            # spelling clashes with SQL's EXTRACT keyword).
            expr = f"get_field(AI_EXTRACT({_col(prev, key)}, named_struct('answer', {question})), 'answer')"
            name = f"{key}{_extract_suffix(op)}"
            built.append((name, expr, OutputColumn(name, kind)))
        return built
    prompts = (
        [(pc["prompt"], {k: schema[k] for k in pc["output_keys"]}) for pc in op["prompts"]]
        if op["type"] == "parallel_map"
        else [(op["prompt"], schema)]
    )
    built = []
    for text, sub_schema in prompts:
        prompt = parse_prompt(text)
        args = [_col(prev, f) for f in prompt.fields]
        document = args[0] if len(args) == 1 else "CONCAT(" + ", ' ', ".join(args) + ")"
        built += _ai_columns(
            sub_schema, prompt.template, args, document, prompt.instruction(), _max_tokens(op)
        )
    return built


def _max_tokens(op: dict) -> int | None:
    return (op.get("litellm_completion_kwargs") or {}).get("max_tokens")


def compile_run(ops: list[dict], table: str) -> CompiledRun:
    """Compile consecutive filter, map, and reduce operations into one query.

    Args:
        ops: The operation configs, each one passing ``check_eligible``.
        table: The name of the uploaded table, which has ``_docetl_id`` and
            each of the run's input fields.
    """
    for op in ops:
        if op.get("type") in SQL_TYPES:
            check_sql_eligible(op)
        else:
            check_eligible(op)

    # Fields the run reads before any operation in it writes them come from
    # the uploaded table.
    written: set[str] = set()
    input_fields: list[str] = []
    for op in ops:
        if op["type"] in SQL_TYPES:
            reads = _sql_reads(op)
        elif op["type"] == "extract":
            reads = list(op["document_keys"])
        elif op["type"] == "parallel_map":
            reads = [f for pc in op["prompts"] for f in parse_prompt(pc["prompt"]).fields]
        elif op["type"] == "reduce":
            reads = [*reduce_keys(op), *parse_reduce_prompt(op["prompt"]).fields]
        else:
            reads = list(parse_prompt(op["prompt"]).fields)
        for f in reads:
            if f not in written and f not in input_fields:
                input_fields.append(f)
        if op["type"] in SQL_TYPES:
            written.update(_sql_writes(op))
            continue
        if op["type"] == "extract":
            written.update(f"{k}{_extract_suffix(op)}" for k in op["document_keys"])
            continue
        written.update(op["output"]["schema"].keys())
        if op["type"] == "reduce":
            written.update(reduce_keys(op))

    id_col = ID_COLUMN  # the current row id; a split moves it to a new column
    columns = [ID_COLUMN, *input_fields]
    ctes = [f"c0 AS (SELECT * FROM `{table}`)"]
    outputs: dict[str, OutputColumn] = {}
    dropped: list[str] = []
    grouped = False
    group_reads = None

    for index, op in enumerate(ops, start=1):
        prev = f"c{index - 1}"
        if op["type"] in SQL_TYPES:
            if op["type"] == "split":
                if id_col not in columns:
                    raise QuailUnsupported(f"split {op['name']!r} after a reduce in the run")
                new_id = f"__docetl_row_{index}"
                ctes.extend(_split_ctes(op, prev, index, columns, id_col, new_id))
                columns = [new_id, *[c for c in columns if c not in (id_col, ID_COLUMN)]]
                id_col = new_id
            else:
                ctes.extend(_gather_ctes(op, prev, index))
            for name, kind in _sql_writes(op).items():
                if name not in columns:
                    columns.append(name)
                outputs[name] = OutputColumn(name, kind)
                if name in dropped:
                    dropped.remove(name)
            continue
        schema = (op.get("output") or {}).get("schema") or {}
        if op["type"] == "filter":
            prompt = parse_prompt(op["prompt"])
            (key,) = schema.keys()
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
        elif op["type"] in ("map", "parallel_map", "extract"):
            built = _row_columns(op, prev, schema)
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
        else:  # reduce
            prompt = parse_reduce_prompt(op["prompt"])
            keys = reduce_keys(op)
            if not grouped and all(k in input_fields for k in keys) and all(
                f in input_fields for f in prompt.fields
            ):
                group_reads = (keys, list(prompt.fields))
            pieces = [
                _lit(value) if kind == "text" else _col(prev, value)
                for kind, value in prompt.item
            ]
            item = pieces[0] if len(pieces) == 1 else "CONCAT(" + ", ".join(pieces) + ")"
            order = f" ORDER BY {_col(prev, id_col)}" if id_col in columns else ""
            document = f"STRING_AGG({item}, {NEWLINE}{order})"
            built = _ai_columns(
                schema, prompt.template, [document], document, prompt.instruction(), _max_tokens(op)
            )
            count = f"_counts_prereduce_{op['name']}"
            parts = [f"{_col(prev, k)} AS `{k}`" for k in keys]
            parts += [f"{expr} AS `{name}`" for name, expr, _ in built]
            parts.append(f"COUNT(*) AS `{count}`")
            group_by = (" GROUP BY " + ", ".join(_col(prev, k) for k in keys)) if keys else ""
            new = [name for name, _, _ in built]
            if op.get("pass_through") and id_col in columns:
                # pass_through keeps the fields of each group's first row:
                # group, keep the first row's id, then join that row back.
                # The result keeps its id, so it maps onto that input row.
                parts.append(f"MIN({_col(prev, id_col)}) AS `__docetl_first`")
                ctes.append(f"c{index}_g AS (SELECT {', '.join(parts)} FROM {prev}{group_by})")
                rest = [c for c in columns if c not in keys and c not in new and c != id_col]
                select = [f"g.`__docetl_first` AS `{id_col}`"]
                select += [f"g.`{c}` AS `{c}`" for c in [*keys, *new, count]]
                select += [f"f.`{c}` AS `{c}`" for c in rest]
                ctes.append(
                    f"c{index} AS (SELECT {', '.join(select)} FROM c{index}_g g "
                    f"JOIN {prev} f ON f.`{id_col}` = g.`__docetl_first`)"
                )
                columns = [id_col, *[k for k in keys if k not in new], *new, count, *rest]
                outputs = {
                    name: out for name, out in outputs.items() if name in rest or name in keys
                }
            else:
                if op.get("pass_through"):
                    raise QuailUnsupported(
                        f"reduce {op['name']!r} sets pass_through after another reduce in the run"
                    )
                ctes.append(f"c{index} AS (SELECT {', '.join(parts)} FROM {prev}{group_by})")
                columns = [k for k in keys if k not in new] + new + [count]
                outputs = {k: outputs.get(k, OutputColumn(k, "key")) for k in keys if k not in new}
                grouped = True
            for name, _, out in built:
                outputs[name] = out
            outputs[count] = OutputColumn(count, "integer")
        for key in op.get("drop_keys") or []:
            if key in columns and key != id_col:
                columns.remove(key)
            outputs.pop(key, None)
            dropped.append(key)

    last = f"c{len(ops)}"
    select = ", ".join(f"{_col(last, c)} AS `{c}`" for c in columns)
    sql = "WITH " + ",\n".join(ctes) + f"\nSELECT {select} FROM {last}"
    return CompiledRun(sql, input_fields, outputs, dropped, grouped, group_reads, id_col)


def convert(value: Any, out: OutputColumn) -> Any:
    """Turn one result cell into the row value DocETL writes."""
    if value is None:
        return [] if out.kind == "string_list" else None
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
    if out.kind == "key":
        return value
    if out.kind == "string_list":
        return [str(value)]
    return str(value)
