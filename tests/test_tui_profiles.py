"""Unit tests for operator coverage in the progress view (issue #487 phase 2).

Covers the model-level changes (``grid_count`` / done-state ``cell_status``)
and the operator-profile registry (per-type units + provenance) without any
LLM calls.
"""

from docetl.progress.events import OpState
from docetl.tui.profiles import get_profile


# -- model: grid_count + cell_status -----------------------------------------
def test_grid_count_uses_work_units_while_running_then_outputs():
    op = OpState("s", "s/resolve", "resolve")
    op.status = "running"
    op.total = 45  # comparisons in flight
    assert op.grid_count == 45  # one cell per work unit while running

    op.status = "done"
    op.out_count = 6  # collapsed to 6 output records
    assert op.grid_count == 6  # switches to one cell per output document


def test_grid_count_handles_fan_out_split():
    # split fans 1 input doc into many chunks; the grid must show the chunks.
    op = OpState("s", "s/split", "split")
    op.status = "done"
    op.total = 1
    op.completed = 1
    op.out_count = 5
    assert op.grid_count == 5
    # all output cells render done even though completed (work units) is 1.
    assert [op.cell_status(i, 0) for i in range(5)] == ["done"] * 5


def test_cell_status_done_marks_errors_first():
    op = OpState("s", "s/op", "map")
    op.status = "done"
    op.out_count = 4
    op.errors = 1
    assert op.cell_status(0, 0) == "error"
    assert op.cell_status(1, 0) == "done"


# -- profiles -----------------------------------------------------------------
def test_units_per_operator_type():
    assert get_profile("reduce").unit == "groups"
    assert get_profile("resolve").unit == "comparisons"
    assert get_profile("split").doc_unit == "chunks"
    assert get_profile("filter").doc_unit == "kept docs"
    # unknown / missing types fall back to the default profile.
    assert get_profile("totally_new_op").unit == "docs"
    assert get_profile(None).provenance is None


def test_reduce_provenance_reports_source_count():
    prof = get_profile("reduce")
    doc = {"summary": "x", "_counts_prereduce_my_reduce": 7}
    lines = prof.provenance(doc)
    assert lines and "merged 7 source documents" in str(lines[0])
    # a group of one is singular and never crashes on missing metadata.
    assert "1 source document" in str(prof.provenance({"_counts_prereduce_r": 1})[0])
    assert prof.provenance({"no": "meta"}) == []


def test_split_provenance_pairs_chunk_and_parent_by_prefix():
    prof = get_profile("split")
    # An unrelated *_id field must not be mistaken for the parent: only the id
    # sharing the chunk_num's prefix counts.
    doc = {
        "content_chunk": "...",
        "customer_id": "should-not-appear",
        "split_x_id": "abcd1234efgh",
        "split_x_chunk_num": 3,
    }
    line = str(prof.provenance(doc)[0])
    assert "chunk 3" in line and "abcd1234" in line
    assert "should-not-appear" not in line


def test_filter_summary_reports_dropped():
    f = OpState("s", "s/f", "filter")
    f.total, f.out_count = 20, 12
    assert "dropped: 8" in "".join(str(x) for x in get_profile("filter").summary(f))
