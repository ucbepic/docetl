"""Nested unnest targets must be replaced in the document being emitted."""

import copy

import pytest

from docetl.frame import from_list
from docetl.operations.unnest import UnnestOperation


def execute(runner, rows, path="user.tags", **options):
    operation = UnnestOperation(
        runner,
        {"name": "nested_unnest", "type": "unnest", "unnest_key": path, **options},
        "gpt-4o-mini",
        1,
    )
    result, cost = operation.execute(rows)
    assert cost == 0
    return result


@pytest.mark.parametrize(
    "path,container",
    [
        ("user.tags", {"tags": ["a", "b"], "name": "Alice"}),
        ("user['tags']", {"tags": ["a", "b"], "name": "Alice"}),
        ("user['items']", {"items": ["a", "b"], "name": "Alice"}),
        ("user['tag.list']", {"tag.list": ["a", "b"], "name": "Alice"}),
        ("user.0", [["a", "b"], "sibling"]),
        ("user[0]", [["a", "b"], "sibling"]),
        ("user[-1]", ["sibling", ["a", "b"]]),
        ("user[0]", (["a", "b"], "sibling")),
    ],
)
def test_nested_list_replaces_target_and_preserves_input(runner, path, container):
    row = {"user": container, "__unnest_value__": "user data", "id": 7}
    original = copy.deepcopy(row)
    result = execute(runner, [row], path)

    expected = []
    for value in ["a", "b"]:
        item = copy.deepcopy(original)
        if isinstance(container, dict):
            field = next(
                key for key, entry in container.items() if isinstance(entry, list)
            )
            item["user"][field] = value
        elif isinstance(container, tuple):
            item["user"] = (value, "sibling")
        else:
            item["user"][-1 if "-1" in path else 0] = value
        expected.append(item)
    assert result == expected
    assert row == original
    if isinstance(container, dict):
        result[0]["user"]["name"] = "changed"
    else:
        result[0]["user"] = "changed"
    assert row == original
    assert result[1] == expected[1]


def test_nested_keep_empty_replaces_array_with_none(runner):
    row = {"user": {"tags": [], "name": "Alice"}}
    assert execute(runner, [row], keep_empty=True) == [
        {"user": {"tags": None, "name": "Alice"}}
    ]
    assert execute(runner, [row], keep_empty=False) == []
    assert row["user"]["tags"] == []


def test_replacing_path_preserves_aliased_sibling(runner):
    shared = {"tags": ["a", "b"], "name": "Alice"}
    row = {"user": shared, "backup": shared}
    original = copy.deepcopy(row)
    assert execute(runner, [row]) == [
        {"user": {"tags": value, "name": "Alice"}, "backup": original["backup"]}
        for value in ["a", "b"]
    ]
    assert row == original


def test_replacing_path_preserves_aliased_list_sibling(runner):
    shared = [{"tags": ["a", "b"], "name": "Alice"}]
    row = {"user": shared, "backup": shared}
    original = copy.deepcopy(row)
    assert execute(runner, [row], "user[0].tags") == [
        {"user": [{"tags": value, "name": "Alice"}], "backup": original["backup"]}
        for value in ["a", "b"]
    ]
    assert row == original


def test_nested_recursive_unnest_and_depth(runner):
    row = {"user": {"tags": [["a", "b"], ["c"]], "name": "Alice"}}
    assert [
        item["user"]["tags"] for item in execute(runner, [row], recursive=True)
    ] == ["a", "b", "c"]
    assert [
        item["user"]["tags"] for item in execute(runner, [row], recursive=True, depth=1)
    ] == [["a", "b"], ["c"]]


def test_recursive_keep_empty_retains_nested_empty_array(runner):
    row = {"user": {"tags": [["a"], []]}}
    assert [
        item["user"]["tags"]
        for item in execute(runner, [row], recursive=True, keep_empty=True)
    ] == ["a", None]


@pytest.mark.parametrize("path", ["user.details", "user['details']"])
def test_nested_dictionary_expands_at_root_without_temporary_keys(runner, path):
    row = {"user": {"details": {"name": "Alice", "age": 30}, "id": 7}}
    original = copy.deepcopy(row)
    assert execute(runner, [row], path, expand_fields=["name", "missing"]) == [
        {**original, "name": "Alice", "missing": None}
    ]
    assert row == original


def test_nested_recursive_list_of_dictionaries_expands_at_root(runner):
    row = {"user": {"tags": [{"name": "Alice"}, {"name": "Bob"}], "id": 7}}
    assert execute(runner, [row], recursive=True, expand_fields=["name"]) == [
        {"user": {"tags": {"name": "Alice"}, "id": 7}, "name": "Alice"},
        {"user": {"tags": {"name": "Bob"}, "id": 7}, "name": "Bob"},
    ]


@pytest.mark.parametrize(
    "path,row",
    [("details", {"details": {}}), ("user.details", {"user": {"details": {}}})],
)
def test_empty_dictionary_emits_one_row(runner, path, row):
    assert execute(runner, [row], path, keep_empty=True, expand_fields=["missing"]) == [
        {**row, "missing": None}
    ]


@pytest.mark.parametrize(
    "path",
    ["user.tags | list", "user.tags + user.other", "user[key]", "user.tags.copy()"],
)
def test_nonassignable_expressions_are_rejected(runner, path):
    with pytest.raises(ValueError, match="path"):
        execute(
            runner, [{"user": {"tags": ["a"], "other": ["b"]}, "key": "tags"}], path
        )


def test_nested_path_metadata_tracks_root_and_preserves_schema():
    config = {"unnest_key": "user['tags']", "expand_fields": []}
    assert UnnestOperation.fields_read(config) == {"user"}
    assert UnnestOperation.fields_written(config) == {"user"}
    schema = {"user": "{tags: list[string], name: string}", "id": "integer"}
    assert UnnestOperation.transform_schema(schema, config) == schema
    assert UnnestOperation.fields_written({"unnest_key": "user.tags"}) is None


def test_nonassignable_path_metadata_fails_closed():
    config = {"unnest_key": "user[key]", "expand_fields": []}
    assert UnnestOperation.fields_read(config) is None
    assert UnnestOperation.fields_written(config) is None


def test_frame_collect_unnests_nested_path():
    assert from_list([{"user": {"tags": ["a", "b"], "id": 7}}]).unnest(
        unnest_key="user.tags"
    ).collect() == [
        {"user": {"tags": "a", "id": 7}},
        {"user": {"tags": "b", "id": 7}},
    ]
