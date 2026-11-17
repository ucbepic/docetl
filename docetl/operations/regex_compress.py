"""Compress a document to the paragraphs that match regular expressions.

A ``regex_compress`` operation cuts a long document down before an LLM reads
it, with no LLM call:

1. Delete every match of each ``remove_patterns`` entry (page numbers,
   repeated headers).
2. Split the text into paragraphs at blank lines; drop empty paragraphs.
3. Keep a paragraph when no ``drop_patterns`` entry matches it, and either it
   starts within the first ``keep_first_chars`` characters or a
   ``keep_patterns`` entry matches it.
4. Join the kept paragraphs with blank lines, with ``gap_marker`` where
   paragraphs were left out.

When no paragraph is kept, the operation keeps the whole text from step 1.
The output goes to ``output_key`` (``document_key`` by default).

On a Quail engine the operation compiles to SQL inside the Quail query next
to it (``docetl.engines.quail.compile``), with the same output. Patterns
therefore use the syntax both Python's ``re`` and Rust's ``regex`` accept:
inline flags such as ``(?i)``, classes, ``\\b``, alternation, and no
lookaround or backreferences.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import Field, field_validator

from docetl.operations.base import BaseOperation, Cardinality

# Whitespace as both Python and Rust read \s: Python also counts the
# separator controls \x1c-\x1f as whitespace, Rust does not.
WHITESPACE_CLASS = r"[^\S\x1c-\x1f]"
PARAGRAPH_BREAK = rf"\n{WHITESPACE_CLASS}*\n"
EMPTY_PARAGRAPH = rf"^{WHITESPACE_CLASS}*$"
SPLIT_MARK = "\x1e"
JOINER = "\n\n"

# Syntax Rust's regex crate refuses: lookaround and backreferences.
_UNPORTABLE = re.compile(r"\(\?<?[=!]|\\[1-9]|\(\?P=|\(\?P?<[A-Za-z_]")


def check_pattern(pattern: str) -> None:
    """Raise ValueError unless ``pattern`` compiles in Python and avoids syntax
    Rust's regex crate (Quail's SQL engine) refuses."""
    try:
        re.compile(pattern)
    except re.error as error:
        raise ValueError(f"pattern {pattern!r} does not compile: {error}") from error
    if _UNPORTABLE.search(pattern):
        raise ValueError(
            f"pattern {pattern!r} uses lookaround, a named group, or a backreference, "
            "which Quail's SQL regex engine does not support"
        )


def compress(text: str | None, config: dict[str, Any]) -> str | None:
    """Return ``text`` compressed as the module docstring describes."""
    if text is None:
        return None
    for pattern in config.get("remove_patterns") or []:
        text = re.sub(pattern, "", text)
    pieces = re.sub(PARAGRAPH_BREAK, SPLIT_MARK, text).split(SPLIT_MARK)
    paragraphs = [p for p in pieces if not re.search(EMPTY_PARAGRAPH, p)]
    keep_patterns = config.get("keep_patterns") or []
    drop_patterns = config.get("drop_patterns") or []
    first = int(config.get("keep_first_chars") or 0)
    kept: list[tuple[int, str]] = []
    before = 0
    for k, paragraph in enumerate(paragraphs):
        dropped = any(re.search(p, paragraph) for p in drop_patterns)
        if not dropped and (before < first or any(re.search(p, paragraph) for p in keep_patterns)):
            kept.append((k, paragraph))
        before += len(paragraph)
    if not kept:
        return text
    marker = config.get("gap_marker", "[...]")
    parts = []
    previous = None
    for k, paragraph in kept:
        if previous is not None and k > previous + 1 and marker:
            parts.append(marker + JOINER + paragraph)
        else:
            parts.append(paragraph)
        previous = k
    return JOINER.join(parts)


class RegexCompressOperation(BaseOperation):
    class schema(BaseOperation.schema):
        type: str = "regex_compress"
        document_key: str
        output_key: str | None = None
        keep_patterns: list[str] = Field(default_factory=list)
        drop_patterns: list[str] = Field(default_factory=list)
        remove_patterns: list[str] = Field(default_factory=list)
        keep_first_chars: int = Field(default=0, ge=0)
        gap_marker: str = "[...]"

        @field_validator("keep_patterns", "drop_patterns", "remove_patterns")
        def _portable(cls, patterns: list[str]) -> list[str]:
            for pattern in patterns:
                check_pattern(pattern)
            return patterns

    @classmethod
    def _output_key(cls, config: dict[str, Any]) -> str | None:
        return config.get("output_key") or config.get("document_key")

    @classmethod
    def transform_schema(cls, schema, config):
        result = super().transform_schema(schema, config)
        if cls._output_key(config):
            result[cls._output_key(config)] = "string"
        return result

    @classmethod
    def cardinality(cls, config: dict[str, Any]) -> Cardinality:
        return Cardinality.ONE_TO_ONE

    @classmethod
    def fields_read(cls, config: dict[str, Any]) -> frozenset[str] | None:
        key = config.get("document_key")
        return frozenset({key}) if key else None

    @classmethod
    def fields_written(cls, config: dict[str, Any]) -> frozenset[str] | None:
        key = cls._output_key(config)
        return frozenset({key}) if key else None

    @classmethod
    def is_row_local(cls, config: dict[str, Any]) -> bool:
        return True

    @classmethod
    def preserves_order(cls, config: dict[str, Any]) -> bool:
        return True

    def syntax_check(self, context: dict[str, Any] | None = None) -> None:
        super().syntax_check(context)

    def execute(self, input_data: list[dict]) -> tuple[list[dict], float]:
        key = self.config["document_key"]
        out = self._output_key(self.config)
        results = []
        for row in input_data:
            row = dict(row)
            value = row.get(key)
            row[out] = compress(value if value is None else str(value), self.config)
            results.append(row)
        return results, 0.0
