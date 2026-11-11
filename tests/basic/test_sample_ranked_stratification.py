"""Proportional ranked sampling must resolve its budget before ranking groups."""

import copy
import json
from collections import Counter
from types import SimpleNamespace

import pytest

from docetl.frame import from_list
from docetl.operations.sample import SampleOperation


class EmbeddingResponse(dict):
    _completion_cost = 1.0


class OfflineEmbeddings:
    def __init__(self):
        self.calls = 0

    def gen_embedding(self, model, texts):
        self.calls += 1
        return EmbeddingResponse(
            data=[
                (
                    {"embedding": [1.0, float(text.split()[-1]) + 1]}
                    if text.split()[-1].isdigit()
                    else {"embedding": [1.0, 0.0]}
                )
                for text in json.loads(texts)
            ]
        )


def operation(method, samples, **options):
    api = OfflineEmbeddings()
    config = {
        "name": "ranked",
        "type": "sample",
        "method": method,
        "samples": samples,
        "stratify_key": "group",
        "method_kwargs": {"keys": ["text"], "query": "target"},
        **options,
    }
    return (
        SampleOperation(SimpleNamespace(config={}, api=api), config, "gpt-4o-mini", 1),
        api,
    )


def documents(groups="AAAABBBB"):
    return [
        {"id": index, "group": group, "text": f"target document {index}"}
        for index, group in enumerate(groups)
    ]


@pytest.mark.parametrize("method", ["top_fts", "top_embedding"])
@pytest.mark.parametrize(
    "samples,expected",
    [
        (0.5, {"A": 2, "B": 2}),
        (0.75, {"A": 3, "B": 3}),
        (4, {"A": 2, "B": 2}),
        (1, {"A": 1}),
        (0.01, {}),
        (20, {"A": 4, "B": 4}),
        (1.0, {"A": 4, "B": 4}),
    ],
)
def test_ranked_proportional_budget_and_groups(method, samples, expected):
    rows = documents()
    original = copy.deepcopy(rows)
    op, api = operation(method, samples)
    config = copy.deepcopy(op.config)
    result, cost = op.execute(rows)

    assert Counter(item["group"] for item in result) == expected
    assert len({item["id"] for item in result}) == sum(expected.values())
    assert rows == original
    assert op.config == config
    for group, count in expected.items():
        selected = [item for item in result if item["group"] == group]
        assert [item["_ranked_rank"] for item in selected] == list(range(1, count + 1))
        scores = [item["_ranked_score"] for item in selected]
        assert scores == sorted(scores, reverse=True)
    if method == "top_embedding":
        assert api.calls == 2 * len(expected)
        assert cost == api.calls
    else:
        assert cost == 0
    repeated, _ = op.execute(rows)
    assert repeated == result


@pytest.mark.parametrize("method", ["top_fts", "top_embedding"])
@pytest.mark.parametrize(
    "groups,samples,expected",
    [("AAABB", 3, {"A": 2, "B": 1}), ("ABCD", 2, {"A": 1, "B": 1})],
)
def test_ranked_proportional_fills_rounding_remainder(
    method, groups, samples, expected
):
    op, _ = operation(method, samples)
    result, _ = op.execute(documents(groups))
    assert Counter(item["group"] for item in result) == expected


@pytest.mark.parametrize("method", ["top_fts", "top_embedding"])
def test_ranked_proportional_multiple_stratification_keys(method):
    rows = documents()
    for row in rows:
        row["source"] = "left" if row["id"] % 2 == 0 else "right"
    op, _ = operation(method, 4, stratify_key=["group", "source"])
    result, _ = op.execute(rows)
    assert Counter((item["group"], item["source"]) for item in result) == {
        ("A", "left"): 1,
        ("A", "right"): 1,
        ("B", "left"): 1,
        ("B", "right"): 1,
    }


@pytest.mark.parametrize("method", ["top_fts", "top_embedding"])
@pytest.mark.parametrize("samples", [2, 0.5])
def test_ranked_per_group_mode_is_unchanged(method, samples):
    op, api = operation(method, samples, samples_per_group=True)
    result, cost = op.execute(documents())
    assert Counter(item["group"] for item in result) == {"A": 2, "B": 2}
    assert [item["_ranked_rank"] for item in result] == [1, 2, 1, 2]
    assert cost == (api.calls if method == "top_embedding" else 0)


def test_ranked_embedding_reports_all_incurred_cost():
    op, api = operation("top_embedding", 4)
    _, cost = op.execute(documents())
    assert cost == api.calls
    assert api.calls == 4


def test_ranked_embedding_selects_most_similar_in_each_group():
    op, _ = operation("top_embedding", 4)
    result, _ = op.execute(documents())
    assert [item["id"] for item in result] == [0, 1, 4, 5]


def test_ranked_fts_renders_query_in_each_group_context():
    rows = documents()
    for row in rows:
        query = "apple" if row["group"] == "A" else "banana"
        row["query"] = query
        row["text"] = query if row["id"] in (0, 4) else "unrelated filler"
    op, _ = operation(
        "top_fts", 2, method_kwargs={"keys": ["text"], "query": "{{ input.query }}"}
    )
    result, _ = op.execute(rows)
    assert [item["id"] for item in result] == [0, 4]


def test_frame_collect_honors_proportional_ranked_budget():
    result = (
        from_list(documents())
        .sample(
            method="top_fts",
            samples=0.5,
            stratify_key="group",
            method_kwargs={"keys": ["text"], "query": "target"},
        )
        .collect()
    )
    assert Counter(item["group"] for item in result) == {"A": 2, "B": 2}
