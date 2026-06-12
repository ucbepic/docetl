"""Regression tests for operation edge cases found by an all-ops smoke run."""

import json
from types import SimpleNamespace

from docetl.operations.equijoin import EquijoinOperation
from docetl.operations.link_resolve import LinkResolveOperation


def test_link_resolve_noop_when_all_links_canonical():
    """Every link already points at an existing id: the op must return the
    data unchanged instead of feeding empty arrays to the similarity code."""
    runner = SimpleNamespace(config={}, api=None, is_cancelled=False)
    op = LinkResolveOperation(
        runner,
        {
            "name": "lr",
            "type": "link_resolve",
            "id_key": "title",
            "link_key": "related_to",
            "comparison_prompt": "Does {{ link_value }} refer to {{ id_value }}? {{ item }}",
            "blocking_threshold": 0.7,
        },
        "gpt-4o-mini",
        max_threads=2,
    )
    data = [
        {"title": "Apple", "related_to": ["Apple Inc"]},
        {"title": "Apple Inc", "related_to": []},
        {"title": "Banana", "related_to": ["Apple"]},
    ]
    out, cost = op.execute(data)
    assert out == data
    assert cost == 0


def test_equijoin_embedding_fallback_is_an_embedding_model():
    """The auto-blocking path must never fall back to the chat default_model
    for embeddings (embedding endpoints reject chat models)."""

    class CaptureAPI:
        def __init__(self):
            self.embed_models = []

        def gen_embedding(self, model, input):
            self.embed_models.append(model)
            texts = json.loads(input) if isinstance(input, str) else input
            return {"data": [{"embedding": [1.0, 0.0]} for _ in texts]}

        def call_llm(self, model, op_type, messages, schema, **kwargs):
            key = next(iter(schema.keys()))
            return SimpleNamespace(response={key: True}, total_cost=0.0)

    api = CaptureAPI()
    runner = SimpleNamespace(config={}, api=api, is_cancelled=False)
    op = EquijoinOperation(
        runner,
        {
            "name": "j",
            "type": "equijoin",
            "comparison_prompt": "Related? {{ left.a }} {{ right.b }}",
            "blocking_threshold": 0.1,
        },
        "gpt-4o-mini",  # chat default_model — must NOT be used for embeddings
        max_threads=2,
    )
    out, _ = op.execute([{"a": "x"}], [{"b": "y"}])
    assert api.embed_models, "embedding blocking path did not run"
    assert all(m == "text-embedding-3-small" for m in api.embed_models), api.embed_models
