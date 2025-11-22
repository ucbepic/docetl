import os
import pytest

from docetl.runner import DSLRunner


pytest.importorskip("lancedb")


def test_lancedb_retriever_fts_only(tmp_path):
    kb = [
        {"id": 1, "text": "alpha beta"},
        {"id": 2, "text": "gamma delta"},
        {"id": 3, "text": "epsilon zeta"},
    ]

    config = {
        "default_model": "gpt-4o-mini",
        "datasets": {
            "kb": {
                "type": "memory",
                "path": kb,
            },
        },
        "retrievers": {
            "kb_r": {
                "type": "lancedb",
                "dataset": "kb",
                "index_dir": str(tmp_path / "idx"),
                "build_index": "always",
                "index_types": ["fts"],
                "fts": {
                    "index_phrase": "{{ input.text }}",
                    "query_phrase": "{{ input.q }}",
                },
                "query": {"top_k": 2, "mode": "fts"},
            }
        },
        "operations": [],
        "pipeline": {"steps": [], "output": {"path": str(tmp_path / "out.json")}},
    }

    runner = DSLRunner(config, max_threads=8)
    # Ensure datasets loaded for retriever indexing
    runner.load()
    r = runner.retrievers["kb_r"]
    r.ensure_index()

    res = r.retrieve({"input": {"q": "alpha"}})
    assert res.docs, "Expected at least one FTS hit"
    ctx = res.rendered_context
    assert "alpha" in ctx, f"retrieval_context should include indexed text, got: {ctx}"

@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Requires OpenAI embeddings (set OPENAI_API_KEY)",
)
def test_lancedb_retriever_embedding_only(tmp_path):
    kb = [
        {"id": 1, "text": "alpha beta"},
        {"id": 2, "text": "gamma delta"},
        {"id": 3, "text": "epsilon zeta"},
    ]

    config = {
        "default_model": "gpt-4o-mini",
        "datasets": {
            "kb": {
                "type": "memory",
                "path": kb,
            },
        },
        "retrievers": {
            "kb_r": {
                "type": "lancedb",
                "dataset": "kb",
                "index_dir": str(tmp_path / "idx2"),
                "build_index": "always",
                "index_types": ["embedding"],
                "embedding": {
                    "model": "text-embedding-3-small",
                    "index_phrase": "{{ input.text }}",
                    "query_phrase": "{{ input.q }}",
                },
                "query": {"top_k": 2, "mode": "embedding"},
            }
        },
        "operations": [],
        "pipeline": {"steps": [], "output": {"path": str(tmp_path / "out2.json")}},
    }

    runner = DSLRunner(config, max_threads=8)
    runner.load()
    r = runner.retrievers["kb_r"]
    r.ensure_index()

    res = r.retrieve({"input": {"q": "alpha"}})
    assert res.docs, "Expected at least one embedding hit"
    assert "alpha" in res.rendered_context or "beta" in res.rendered_context


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Requires OpenAI embeddings (set OPENAI_API_KEY)",
)
def test_lancedb_retriever_hybrid(tmp_path):
    kb = [
        {"id": 1, "text": "alpha beta"},
        {"id": 2, "text": "gamma delta"},
        {"id": 3, "text": "epsilon zeta"},
    ]

    config = {
        "default_model": "gpt-4o-mini",
        "datasets": {
            "kb": {
                "type": "memory",
                "path": kb,
            },
        },
        "retrievers": {
            "kb_r": {
                "type": "lancedb",
                "dataset": "kb",
                "index_dir": str(tmp_path / "idx3"),
                "build_index": "always",
                "index_types": ["fts", "embedding"],
                "fts": {
                    "index_phrase": "{{ input.text }}",
                    "query_phrase": "{{ input.q }}",
                },
                "embedding": {
                    "model": "text-embedding-3-small",
                    "index_phrase": "{{ input.text }}",
                    "query_phrase": "{{ input.q }}",
                },
                "query": {"top_k": 2, "mode": "hybrid"},
            }
        },
        "operations": [],
        "pipeline": {"steps": [], "output": {"path": str(tmp_path / "out3.json")}},
    }

    runner = DSLRunner(config, max_threads=8)
    runner.load()
    r = runner.retrievers["kb_r"]
    r.ensure_index()

    res = r.retrieve({"input": {"q": "alpha"}})
    assert res.docs, "Expected at least one hybrid hit"
    assert "alpha" in res.rendered_context

