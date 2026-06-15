"""Pre-execution checks for DocETL operations.

When ``agent_mode=True`` (set per-operation, per-pipeline, or globally via
``docetl.agent_mode``), these checks run before LLM calls and raise
informative errors that an agent can act on.  Without agent mode, long text
is handled by DocETL's normal truncation (with a warning pointing to
split/gather).

All checks operate on PyArrow tables internally.  Callers that have
``list[dict]`` data (operations) use :func:`run_checks_for_op` which
converts once at entry; callers that already have Arrow (AI SQL) call the
check functions directly.

TODO: additional checks to add:
- Smarter ChunkOverflowError guidance: scan text for structure markers
  (HTML tags, markdown headers, double newlines) and suggest a split
  strategy with estimated chunk counts, instead of a generic message.
- Sparse output warning (post-execution): flag when most AI outputs are
  null/empty, meaning the prompt likely didn't match the content.
- Loop detection: detect ChunkOverflow → TooManyRows cycles and suggest
  filter-then-chunk instead of chunk-then-filter.
- Coverage gap (post-execution): after range queries, check that output
  rows actually span the full predicate range.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.compute as pc

if TYPE_CHECKING:
    pass

_DEFAULT_MAX_CHUNK_TOKENS = 100_000
_CHARS_PER_TOKEN = 4
_HIGH_CARDINALITY_RATIO = 0.8
_DEFAULT_MAX_AI_ROWS = 100

_PROMPT_COL_RE = re.compile(r"\{\{\s*input\.(\w+)\s*\}\}")


# ── agent-facing pre-execution errors ─────────────────────────────────────


class AgentCheckError(ValueError):
    """Base class for all agent-mode check errors.

    Catch this to handle any agent-mode validation failure."""


class ChunkOverflowError(AgentCheckError):
    """Text columns are too long for AI functions.  Raised when
    ``agent_mode=True`` and text exceeds *max_chunk_tokens*."""

    def __init__(self, num_rows: int, max_tokens: int, columns: dict[str, dict]):
        self.num_rows = num_rows
        self.max_tokens = max_tokens
        self.columns = columns
        parts = [
            f"Text too long for AI functions (limit: ~{max_tokens:,} tokens). "
            f"{num_rows} rows matched your relational predicates.",
        ]
        for col, info in columns.items():
            parts.append(
                f"  Column '{col}': longest value ~{info['max_tokens']:,} tokens, "
                f"{info['rows_over_limit']}/{info['rows']} rows exceed the limit."
            )
        parts.append(
            "Split text into chunks in a subquery before AI functions, "
            "e.g. unnest(regexp_split_to_array(text, '\\n\\n+')) AS chunk."
        )
        super().__init__("\n".join(parts))


class HighCardinalityError(AgentCheckError):
    """A GROUP BY / reduce key has near-unique values, so the LLM would
    run one group per row.  Raised when ``agent_mode=True``."""

    def __init__(
        self,
        num_rows: int,
        reduce_key: list[str],
        n_distinct: int,
        op_type: str,
    ):
        self.num_rows = num_rows
        self.reduce_key = reduce_key
        self.n_distinct = n_distinct
        self.op_type = op_type
        key_str = ", ".join(reduce_key)
        super().__init__(
            f"High cardinality: GROUP BY ({key_str}) has {n_distinct:,} distinct "
            f"values across {num_rows:,} rows — the LLM would run ~{n_distinct:,} "
            f"times with ~{max(1, num_rows // n_distinct)} row(s) per group. "
            f"Consider canonicalizing with ai_extract or deduplicating with "
            f"ai_resolve first."
        )


class EmptyInputError(AgentCheckError):
    """Relational predicates matched zero rows.  Raised when
    ``agent_mode=True`` and a semantic stage would receive no input."""

    def __init__(self, sql_hint: str | None = None):
        msg = "Your relational predicates matched 0 rows — nothing to process."
        if sql_hint:
            msg += f" Last relational SQL: {sql_hint}"
        super().__init__(msg)


class MissingColumnError(AgentCheckError):
    """A prompt references columns that don't exist in the table."""

    def __init__(self, missing: set[str], available: list[str]):
        self.missing = missing
        self.available = available
        super().__init__(
            f"Prompt references columns that don't exist: {sorted(missing)}. "
            f"Available columns: {sorted(available)}."
        )


class TooManyRowsError(AgentCheckError):
    """The semantic stage would process too many rows.  Raised when
    ``agent_mode=True`` and the input exceeds a configurable threshold."""

    def __init__(self, num_rows: int, threshold: int):
        self.num_rows = num_rows
        self.threshold = threshold
        super().__init__(
            f"Query sends {num_rows:,} rows to the LLM (threshold: {threshold:,}). "
            f"Add WHERE predicates to narrow the input."
        )


class AmbiguousSourceError(AgentCheckError):
    """Multiple distinct data regions match the query terms, but the
    operation only consumed a subset.  Raised when ``agent_mode=True``
    and the input contains rows from multiple provenance regions that
    all match the key terms in the prompt.

    This is format-agnostic: "regions" are detected by row provenance
    (source file, section header, or other grouping key), not by parsing
    HTML tables or any specific format.
    """

    def __init__(
        self,
        term: str,
        used_sources: list[str],
        all_sources: list[str],
    ):
        self.term = term
        self.used_sources = used_sources
        self.all_sources = all_sources
        unused = [s for s in all_sources if s not in used_sources]
        super().__init__(
            f"Ambiguous data source: '{term}' appears in {len(all_sources)} "
            f"distinct regions but only {len(used_sources)} were used. "
            f"Regions used: {used_sources}. "
            f"Also found in: {unused}. "
            f"Consider whether a different source better matches your query."
        )


# ── check helpers ────────────────────────────────────────────────────────


def text_cols_for_op(op_config: dict[str, Any]) -> set[str]:
    """Extract column names referenced in an operation's prompt template."""
    cols: set[str] = set()
    prompt = op_config.get("prompt") or op_config.get("comparison_prompt") or ""
    cols.update(_PROMPT_COL_RE.findall(prompt))
    for dk in op_config.get("document_keys") or []:
        cols.add(dk)
    return cols


# ── Arrow-native checks ─────────────────────────────────────────────────


def check_empty(table: pa.Table, *, sql_hint: str | None = None) -> None:
    if table.num_rows == 0:
        raise EmptyInputError(sql_hint)


def check_row_count(
    table: pa.Table,
    *,
    max_ai_rows: int = _DEFAULT_MAX_AI_ROWS,
) -> None:
    if table.num_rows > max_ai_rows:
        raise TooManyRowsError(table.num_rows, max_ai_rows)


def check_missing_cols(
    table: pa.Table,
    op_config: dict[str, Any],
) -> None:
    """Check that columns referenced in the prompt exist in the data."""
    if table.num_rows == 0:
        return
    text_cols = text_cols_for_op(op_config)
    if not text_cols:
        return
    available = table.column_names
    missing = text_cols - set(available)
    if missing:
        raise MissingColumnError(missing, available)


def check_cardinality(
    table: pa.Table,
    op_config: dict[str, Any],
) -> None:
    """Check for near-unique reduce keys (high cardinality)."""
    rk = op_config.get("reduce_key")
    op_type = op_config.get("type", "reduce")
    if not rk or table.num_rows < 2:
        return
    if isinstance(rk, str):
        rk = [rk]
    col_names = set(table.column_names)
    if any(k not in col_names for k in rk):
        return
    if len(rk) == 1:
        n_distinct = pc.count_distinct(table.column(rk[0])).as_py()
    else:
        struct = pa.StructArray.from_arrays(
            [table.column(k) for k in rk],
            names=rk,
        )
        n_distinct = pc.count_distinct(struct).as_py()
    ratio = n_distinct / table.num_rows
    if ratio >= _HIGH_CARDINALITY_RATIO:
        raise HighCardinalityError(table.num_rows, rk, n_distinct, op_type)


def check_chunk_overflow(
    table: pa.Table,
    op_config: dict[str, Any],
    *,
    max_chunk_tokens: int = _DEFAULT_MAX_CHUNK_TOKENS,
) -> None:
    """Check for text columns that exceed the token limit."""
    text_cols = text_cols_for_op(op_config)
    if not text_cols or table.num_rows == 0:
        return
    max_chars = max_chunk_tokens * _CHARS_PER_TOKEN
    long_cols: dict[str, dict] = {}
    col_names = set(table.column_names)
    for col in text_cols:
        if col not in col_names:
            continue
        arr = table.column(col)
        if not pa.types.is_string(arr.type) and not pa.types.is_large_string(arr.type):
            arr = pc.cast(arr, pa.string())
        lengths = pc.utf8_length(arr)
        max_len = pc.max(lengths).as_py()
        if max_len is None or max_len <= max_chars:
            continue
        count = len(lengths) - pc.sum(pc.is_null(lengths)).as_py()
        over_limit = pc.sum(pc.greater(lengths, max_chars)).as_py()
        long_cols[col] = {
            "max_tokens": max_len // _CHARS_PER_TOKEN,
            "rows": count,
            "rows_over_limit": over_limit,
        }
    if long_cols:
        raise ChunkOverflowError(table.num_rows, max_chunk_tokens, long_cols)


# ── entry point for operation callers (list[dict] → Arrow) ──────────────


def run_checks_for_op(
    input_data: list[dict],
    op_config: dict[str, Any],
    *,
    max_chunk_tokens: int = _DEFAULT_MAX_CHUNK_TOKENS,
    max_ai_rows: int = _DEFAULT_MAX_AI_ROWS,
) -> None:
    """Run all applicable agent-mode checks for an operation.

    Converts ``input_data`` to Arrow once, then runs the shared checks.
    Raises the first error found.  Call this at the top of ``execute()``
    when ``agent_mode`` is enabled.
    """
    if not input_data:
        raise EmptyInputError()

    table = pa.Table.from_pylist(input_data)
    op_type = op_config.get("type", "")

    if max_ai_rows and op_type not in ("reduce", "resolve"):
        check_row_count(table, max_ai_rows=max_ai_rows)

    check_missing_cols(table, op_config)

    if op_type in ("reduce", "resolve"):
        check_cardinality(table, op_config)

    check_chunk_overflow(table, op_config, max_chunk_tokens=max_chunk_tokens)


# ── post-execution checks ───────────────────────────────────────────────

_SOURCE_KEY_CANDIDATES = (
    "source",
    "source_file",
    "filename",
    "file",
    "path",
    "_source",
)
_SECTION_KEY_CANDIDATES = (
    "section",
    "section_header",
    "table_title",
    "heading",
    "_section",
)


def _detect_key(row: dict, candidates: tuple[str, ...]) -> str | None:
    """Find the first matching column from *candidates* in a row."""
    for key in candidates:
        if key in row:
            return key
    return None


def _region_label(row: dict, source_key: str | None, section_key: str | None) -> str:
    """Build a human-readable region label from provenance columns."""
    parts = []
    if source_key and row.get(source_key):
        parts.append(str(row[source_key]))
    if section_key and row.get(section_key):
        parts.append(str(row[section_key]))
    return " > ".join(parts) if parts else ""


def _extract_prompt_terms(prompt_template: str) -> list[str]:
    """Extract literal search terms from a prompt template.

    Looks for quoted strings (the most common way agents specify what
    to search for) and returns them lowercased.
    """
    terms = re.findall(r"""['"]([^'"]{3,})['"]""", prompt_template)
    return [t.lower().strip() for t in terms if t.strip()]


def check_ambiguous_source(
    input_data: list[dict],
    output_data: list[dict],
    op_config: dict[str, Any],
) -> None:
    """Post-execution check: did the LLM only engage with a subset of
    matching data regions?

    Compares the provenance of rows that produced non-null output against
    all input rows that contain the prompt's key terms. If the terms
    appear in multiple regions but only some were used, raises
    :class:`AmbiguousSourceError`.
    """
    if not input_data or not output_data:
        return

    source_key = _detect_key(input_data[0], _SOURCE_KEY_CANDIDATES)
    section_key = _detect_key(input_data[0], _SECTION_KEY_CANDIDATES)
    if not source_key and not section_key:
        return

    prompt = op_config.get("prompt", "")
    terms = _extract_prompt_terms(prompt)
    if not terms:
        return

    text_cols = text_cols_for_op(op_config)
    if not text_cols:
        return

    def row_text(row: dict) -> str:
        return " ".join(str(row.get(c, "")) for c in text_cols).lower()

    def row_matches_any_term(row: dict) -> bool:
        text = row_text(row)
        return any(t in text for t in terms)

    matching_regions: dict[str, list[int]] = {}
    for i, row in enumerate(input_data):
        if row_matches_any_term(row):
            label = _region_label(row, source_key, section_key)
            if label:
                matching_regions.setdefault(label, []).append(i)

    if len(matching_regions) < 2:
        return

    output_schema_keys = set((op_config.get("output") or {}).get("schema") or {})
    used_indices: set[int] = set()
    for out_row in output_data:
        has_output = any(
            out_row.get(k) is not None and out_row.get(k) != ""
            for k in output_schema_keys
        )
        if not has_output:
            continue
        for i, inp_row in enumerate(input_data):
            if all(
                inp_row.get(k) == out_row.get(k)
                for k in inp_row
                if k not in output_schema_keys
            ):
                used_indices.add(i)
                break

    used_regions = set()
    for idx in used_indices:
        label = _region_label(input_data[idx], source_key, section_key)
        if label:
            used_regions.add(label)

    all_regions = list(matching_regions.keys())
    if used_regions and len(used_regions) < len(all_regions):
        first_term = terms[0] if terms else "query"
        raise AmbiguousSourceError(
            term=first_term,
            used_sources=sorted(used_regions),
            all_sources=sorted(all_regions),
        )
