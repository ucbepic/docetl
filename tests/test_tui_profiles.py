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
    op = OpState("s", "s/r", "reduce")
    assert prof.provenance(op, {"_counts_prereduce_r": 7}) == "combined from 7 input documents"
    # a group of one is singular; missing metadata yields no line (not a crash).
    assert prof.provenance(op, {"_counts_prereduce_r": 1}) == "combined from 1 input document"
    assert prof.provenance(op, {"no": "meta"}) is None


def test_split_provenance_reads_as_chunk_n_of_m_without_uuid():
    prof = get_profile("split")
    op = OpState("s", "s/split", "split")
    op.outputs = [
        {"split_x_id": "be59-uuid", "split_x_chunk_num": 1},
        {"split_x_id": "be59-uuid", "split_x_chunk_num": 2},
        {"split_x_id": "be59-uuid", "split_x_chunk_num": 3},
    ]
    line = prof.provenance(op, op.outputs[1])
    assert line == "chunk 2 of 3"
    assert "be59-uuid" not in line  # the raw parent id is never shown
    # split's internal chunk-bookkeeping keys are hidden from the field list.
    assert prof.consumed_keys(op.outputs[1]) == {"split_x_id", "split_x_chunk_num"}


def test_filter_summary_reports_dropped():
    f = OpState("s", "s/f", "filter")
    f.total, f.out_count = 20, 12
    assert "dropped: 8" in "".join(str(x) for x in get_profile("filter").summary(f))
