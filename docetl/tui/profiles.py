"""Operator-specific presentation for the progress TUI.

The dashboard renders every operation through a *default* profile (status,
output JSON, prompt). Specific operator types layer extra context on top via a
small registry: the progress *unit* word (what the live denominator counts) and
a per-document *provenance* line derived from metadata the operators already
emit — no global document-id system required.

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
    """How one operator type is presented in the progress view.

    ``unit`` / ``doc_unit`` are the nouns shown for the live work-unit counter
    and for finished output documents respectively. ``provenance`` and
    ``summary`` are optional hooks returning extra ``Text`` lines for the
    per-document detail and the operation-level summary; both default to "no
    extra lines" so the default rendering is used as-is.
    """

    unit: str = "docs"
    doc_unit: str = "docs"
    provenance: Callable[[dict], list[Text]] | None = None
    summary: Callable[[OpState], list[Text]] | None = None


# -- shared provenance helpers ------------------------------------------------
def _section(label: str, value: str) -> Text:
    """A titled detail section: an underlined header over a grey value line."""
    t = Text(f"\n{label}\n", style="bold underline")
    t.append(value, style="grey70")
    return t


def _reduce_provenance(doc: dict) -> list[Text]:
    # reduce writes ``_counts_prereduce_<name>`` = number of source docs merged.
    for k, v in doc.items():
        if k.startswith("_counts_prereduce_") and isinstance(v, (int, float)):
            n = int(v)
            return [_section(
                "provenance", f"merged {n} source document{'' if n == 1 else 's'}"
            )]
    return []


def _split_provenance(doc: dict) -> list[Text]:
    # split writes ``<name>_chunk_num`` and ``<name>_id`` (the parent doc id);
    # pair them by their shared ``<name>`` prefix so an unrelated ``*_id`` field
    # can't be mistaken for the parent.
    for k, chunk in doc.items():
        if not k.endswith("_chunk_num"):
            continue
        parent = doc.get(k[: -len("_chunk_num")] + "_id")
        value = f"chunk {chunk}"
        if isinstance(parent, str):
            value += f" of parent {parent[:8]}"
        return [_section("provenance", value)]
    return []


def _filter_summary(op: OpState) -> list[Text]:
    # The generic "output: N kept docs" line carries the kept count; add the
    # complementary dropped count that the default rendering can't derive.
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
    "split": OpProfile(doc_unit="chunks", provenance=_split_provenance),
    "gather": OpProfile(doc_unit="chunks"),
}


def get_profile(op_type: str | None) -> OpProfile:
    return _PROFILES.get(op_type or "", _DEFAULT)
