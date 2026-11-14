"""Gather header context follows the active section within each document."""

import copy

import pytest

import docetl.operations.gather as gather_module
from docetl.frame import from_list
from docetl.operations.gather import GatherOperation

PERIPHERAL = {"previous": {"tail": {"count": 0}}}


def chunk(order, headers=(), doc="first"):
    return {
        "doc": doc,
        "order": order,
        "text": f"content-{doc}-{order}",
        "headers": [{"level": level, "header": text} for level, text in headers],
    }


def operation(runner, **options):
    return GatherOperation(
        runner,
        {
            "name": "gather_context",
            "type": "gather",
            "content_key": "text",
            "doc_id_key": "doc",
            "order_key": "order",
            "doc_header_key": "headers",
            "peripheral_chunks": PERIPHERAL,
            **options,
        },
        "gpt-4o-mini",
        1,
    )


def section(text):
    return next(
        (line for line in text.splitlines() if line.startswith("_Current Section:_")),
        "",
    )


def test_headingless_continuation_inherits_active_section(runner):
    rows = [chunk(0, [(1, "Document")]), chunk(1, [(2, "Section")]), chunk(2)]
    original = copy.deepcopy(rows)
    output, cost = operation(runner).execute(rows)

    assert [section(row["text_rendered"]) for row in output] == [
        "",
        "_Current Section:_ # Document",
        "_Current Section:_ # Document > ## Section",
    ]
    assert rows == original
    assert cost == 0


def test_sparse_parent_invalidates_all_stale_descendants(runner):
    rows = [
        chunk(0, [(1, "Document")]),
        chunk(1, [(4, "Old branch")]),
        chunk(2, [(2, "New branch")]),
        chunk(3, [(6, "Deep section")]),
        chunk(4),
    ]
    output, _ = operation(runner).execute(rows)
    assert section(output[3]["text_rendered"]) == (
        "_Current Section:_ # Document > ## New branch"
    )
    assert section(output[4]["text_rendered"]) == (
        "_Current Section:_ # Document > ## New branch > ###### Deep section"
    )


def test_sibling_and_root_transitions_replace_the_active_branch(runner):
    rows = [
        chunk(0, [(1, "First root"), (2, "First child")]),
        chunk(1, [(3, "Old subsection")]),
        chunk(2, [(2, "Sibling child")]),
        chunk(3),
        chunk(4, [(1, "Second root")]),
        chunk(5),
    ]
    output, _ = operation(runner).execute(rows)
    assert section(output[3]["text_rendered"]) == (
        "_Current Section:_ # First root > ## Sibling child"
    )
    assert section(output[4]["text_rendered"]) == ""
    assert section(output[5]["text_rendered"]) == "_Current Section:_ # Second root"


def test_sorted_document_headers_do_not_leak_between_documents_or_executions(runner):
    rows = [
        chunk(2),
        chunk(1, [(2, "Second child")], doc="second"),
        chunk(0, [(1, "First root")]),
        chunk(2, doc="second"),
        chunk(1, [(2, "First child")]),
        chunk(0, doc="second"),
    ]
    original = copy.deepcopy(rows)
    op = operation(runner)
    output, _ = op.execute(rows)
    contexts = {
        (row["doc"], row["order"]): section(row["text_rendered"]) for row in output
    }
    assert contexts[("first", 2)] == "_Current Section:_ # First root > ## First child"
    assert contexts[("second", 0)] == ""
    assert contexts[("second", 1)] == ""
    assert contexts[("second", 2)] == "_Current Section:_ ## Second child"
    assert rows == original
    assert [(row["doc"], row["order"]) for row in output] == [
        ("first", 0),
        ("first", 1),
        ("first", 2),
        ("second", 0),
        ("second", 1),
        ("second", 2),
    ]
    repeated, _ = op.execute([chunk(0), chunk(1)])
    assert all(section(row["text_rendered"]) == "" for row in repeated)


@pytest.mark.parametrize(
    "options",
    [{"doc_header_key": None}, {"doc_header_key": None, "peripheral_chunks": {}}],
)
def test_existing_disabled_header_behavior_is_preserved(runner, options):
    rows = [chunk(0, [(1, "Document")]), chunk(1)]
    output, _ = operation(runner, **options).execute(rows)
    assert all(section(row["text_rendered"]) == "" for row in output)
    if options.get("peripheral_chunks") == {}:
        assert [row["text_rendered"] for row in output] == [row["text"] for row in rows]


def test_header_only_configuration_preserves_context_without_peripheral_markers(runner):
    rows = [chunk(0, [(1, "Document")]), chunk(1)]
    output, _ = operation(runner, peripheral_chunks={}).execute(rows)
    assert output[0]["text_rendered"] == rows[0]["text"]
    assert output[1]["text_rendered"] == (
        "_Current Section:_ # Document\ncontent-first-1"
    )


def test_frame_collect_supports_header_only_configuration():
    rows = [chunk(0, [(1, "Document")]), chunk(1)]
    output = (
        from_list(rows)
        .gather(
            content_key="text",
            doc_id_key="doc",
            order_key="order",
            doc_header_key="headers",
            peripheral_chunks={},
        )
        .collect()
    )
    assert output[0]["text_rendered"] == rows[0]["text"]
    assert output[1]["text_rendered"] == (
        "_Current Section:_ # Document\ncontent-first-1"
    )


def test_direct_render_helpers_keep_their_existing_call_signature(runner):
    rows = [
        chunk(0, [(1, "Document")]),
        chunk(1, [(4, "Stale")]),
        chunk(2, [(2, "Branch")]),
        chunk(3, [(6, "Leaf")]),
        chunk(4),
    ]
    op = operation(runner)
    expected = "_Current Section:_ # Document > ## Branch"
    assert op.render_hierarchy_headers(rows[3], rows[:4], "headers") == expected
    rendered = op.render_chunk_with_context(
        rows, 4, PERIPHERAL, "text", "order", "BEGIN", "END", "headers"
    )
    assert section(rendered) == expected + " > ###### Leaf"
    assert "BEGIN\ncontent-first-4\nEND" in rendered


def test_frame_collect_inherits_header_path_for_continuation():
    rows = [chunk(0, [(1, "Document")]), chunk(1, [(2, "Section")]), chunk(2)]
    output = (
        from_list(rows)
        .gather(
            content_key="text",
            doc_id_key="doc",
            order_key="order",
            doc_header_key="headers",
            peripheral_chunks=PERIPHERAL,
        )
        .collect()
    )
    assert section(output[2]["text_rendered"]) == (
        "_Current Section:_ # Document > ## Section"
    )


@pytest.mark.parametrize("empty_headers", [[], None])
def test_empty_or_none_headers_continue_the_active_section(runner, empty_headers):
    rows = [chunk(0, [(1, "Document")]), chunk(1)]
    rows[1]["headers"] = empty_headers
    output, _ = operation(runner).execute(rows)
    assert section(output[1]["text_rendered"]) == "_Current Section:_ # Document"


@pytest.mark.parametrize("invalid_headers", [False, 0, "", {}])
def test_malformed_header_container_clears_active_section(
    runner, monkeypatch, invalid_headers
):
    rows = [chunk(0, [(1, "Old root")]), chunk(1), chunk(2)]
    rows[1]["headers"] = invalid_headers
    original = copy.deepcopy(rows)
    op = operation(runner)
    errors = []
    monkeypatch.setattr(op.console, "log", lambda message: errors.append(str(message)))
    output, _ = op.execute(rows)
    assert [section(row["text_rendered"]) for row in output] == ["", "", ""]
    assert any("Error processing headers" in message for message in errors)
    assert rows == original


@pytest.mark.parametrize(
    "invalid_entry",
    [
        {"level": 0, "header": "Invalid"},
        {"level": -1, "header": "Invalid"},
        {"level": True, "header": "Invalid"},
        {"level": 1.5, "header": "Invalid"},
        {"level": "2", "header": "Invalid"},
        {"level": 2, "header": 123},
        None,
        "unsupported entry",
    ],
)
def test_malformed_chunk_clears_ancestry_atomically_and_recovers(
    runner, monkeypatch, invalid_entry
):
    rows = [
        chunk(0, [(1, "Old root")]),
        chunk(1, [(2, "Partial heading")]),
        chunk(2),
        chunk(3, [(3, "Recovered")]),
        chunk(4),
    ]
    rows[1]["headers"].append(invalid_entry)
    original = copy.deepcopy(rows)
    op = operation(runner)
    errors = []
    monkeypatch.setattr(op.console, "log", lambda message: errors.append(str(message)))
    output, _ = op.execute(rows)

    assert [section(row["text_rendered"]) for row in output] == [
        "",
        "",
        "",
        "",
        "_Current Section:_ ### Recovered",
    ]
    assert any("Error processing headers" in message for message in errors)
    assert rows == original


def test_multiple_headings_in_one_chunk_apply_in_document_order(runner):
    rows = [
        chunk(0, [(1, "Root"), (4, "Discarded"), (2, "Branch"), (5, "Leaf")]),
        chunk(1),
    ]
    original = copy.deepcopy(rows)
    output, _ = operation(runner).execute(rows)
    assert section(output[1]["text_rendered"]) == (
        "_Current Section:_ # Root > ## Branch > ##### Leaf"
    )
    assert rows == original


def test_execute_reads_each_chunks_headers_once_across_documents(runner, monkeypatch):
    rows = [
        chunk(1),
        chunk(0, [(1, "First")]),
        chunk(1, doc="second"),
        chunk(0, [(1, "Second")], doc="second"),
    ]
    header_reads = []
    lookup_field = gather_module.lookup_field

    def observe_lookup(item, key):
        if key == "headers":
            header_reads.append(item)
        return lookup_field(item, key)

    monkeypatch.setattr(gather_module, "lookup_field", observe_lookup)
    output, _ = operation(runner).execute(rows)
    assert len(header_reads) == len(rows)
    assert {id(item) for item in header_reads} == {id(item) for item in rows}
    assert section(output[1]["text_rendered"]) == "_Current Section:_ # First"
    assert section(output[3]["text_rendered"]) == "_Current Section:_ # Second"


def test_standalone_header_helper_accepts_a_copy_of_the_current_chunk(runner):
    rows = [chunk(0, [(1, "Root")]), chunk(1, [(2, "Branch")]), chunk(2)]
    assert (
        operation(runner).render_hierarchy_headers(
            copy.deepcopy(rows[2]), rows, "headers"
        )
        == "_Current Section:_ # Root > ## Branch"
    )
