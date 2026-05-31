"""Operator-specific presentation for the progress TUI.

The dashboard renders every operation through a *default* profile (status +
output fields + prompt). Specific operator types layer extra context on top via
a small registry:

- ``unit`` / ``doc_unit`` — the nouns for the live work-unit counter and for
  finished output documents (docs / groups / comparisons / chunks …).
- ``provenance`` — a short human-readable line about where a document came from,
  derived from metadata the operators already emit (no global document-id
  system): reduce's ``_counts_prereduce_*`` source counts, split's
  ``<name>_chunk_num`` / ``<name>_id``.
- ``consumed_keys`` — internal metadata keys the profile turns into provenance,
  so the detail pane hides them from the raw field list.
- ``summary`` — extra operation-level rows (e.g. filter's dropped count).

Everything falls back to :data:`_DEFAULT`, so an unknown/new operator still
renders correctly; a profile only overrides the parts it cares about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from rich.text import Text

from docetl.progress.events import OpState


@dataclass(frozen=True)
class OpProfile:
    unit: str = "docs"
    doc_unit: str = "docs"
    provenance: Callable[[OpState, dict], str | None] | None = None
    consumed_keys: Callable[[dict], set[str]] | None = None
    summary: Callable[[OpState], list[Text]] | None = None


# -- reduce: how many input documents were combined into this group ----------
def _reduce_provenance(op: OpState, doc: dict) -> str | None:
    for k, v in doc.items():
        if k.startswith("_counts_prereduce_") and isinstance(v, (int, float)):
            n = int(v)
            return f"combined from {n} input document{'' if n == 1 else 's'}"
    return None


# -- split: which chunk of its source document this is -----------------------
def _split_chunk_key(doc: dict) -> str | None:
    for k in doc:
        if k.endswith("_chunk_num"):
            return k
    return None


def _split_provenance(op: OpState, doc: dict) -> str | None:
    chunk_key = _split_chunk_key(doc)
    if chunk_key is None:
        return None
    chunk = doc[chunk_key]
    parent_key = chunk_key[: -len("_chunk_num")] + "_id"
    parent = doc.get(parent_key)
    # Count this parent's chunks so the user sees "chunk 2 of 5", not a raw uuid.
    total = sum(1 for d in op.outputs if d.get(parent_key) == parent) if parent else 0
    return f"chunk {chunk} of {total}" if total else f"chunk {chunk}"


def _split_consumed(doc: dict) -> set[str]:
    chunk_key = _split_chunk_key(doc)
    if chunk_key is None:
        return set()
    prefix = chunk_key[: -len("_chunk_num")]
    consumed = {chunk_key, prefix + "_id"}
    # Hide the original (full) split_key field — it's the entire source document
    # copied onto every chunk, and the ``<split_key>_chunk`` field supersedes it.
    for k in doc:
        if k.endswith("_chunk") and not k.endswith("_chunk_num"):
            split_key = k[: -len("_chunk")]
            if split_key in doc:
                consumed.add(split_key)
            break
    return consumed


# -- filter: the dropped count the default rendering can't derive ------------
def _filter_summary(op: OpState) -> list[Text]:
    if op.total is None or op.out_count is None:
        return []
    dropped = max(0, op.total - op.out_count)
    return [Text(f"dropped: {dropped:,}\n", style="grey70")]


# -- registry -----------------------------------------------------------------
# Only operators whose presentation differs from the default appear here;
# everything else (map, parallel_map, unnest, …) resolves to ``_DEFAULT``.
_DEFAULT = OpProfile()

_PROFILES: dict[str, OpProfile] = {
    "filter": OpProfile(doc_unit="kept docs", summary=_filter_summary),
    "reduce": OpProfile(unit="groups", doc_unit="groups", provenance=_reduce_provenance),
    "resolve": OpProfile(unit="comparisons", doc_unit="records"),
    "equijoin": OpProfile(unit="comparisons", doc_unit="pairs"),
    "split": OpProfile(
        doc_unit="chunks", provenance=_split_provenance, consumed_keys=_split_consumed
    ),
    "gather": OpProfile(doc_unit="chunks"),
}


def get_profile(op_type: str | None) -> OpProfile:
    return _PROFILES.get(op_type or "", _DEFAULT)
