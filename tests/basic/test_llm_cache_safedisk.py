import json

import pytest
from diskcache_community import Cache, Disk
from litellm import ModelResponse
from litellm.types.utils import EmbeddingResponse

import docetl.operations.utils.api as api_mod
from docetl.operations.map import MapOperation
from docetl.operations.utils.cache import cache
from docetl.operations.utils.cascade import CascadeResult, CascadeStats
from docetl.operations.utils.openai_agents_runner import AgentResult

planted_calls = []


def tool_reply(output):
    call = {"name": "send_output", "arguments": json.dumps(output)}
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call_1", "type": "function", "function": call}],
    }
    return ModelResponse(
        model="gpt-4o-mini",
        choices=[{"index": 0, "finish_reason": "tool_calls", "message": message}],
    )


def planted():
    planted_calls.append(1)
    return tool_reply({"sentiment": "planted"})


class Planted:
    def __reduce__(self):
        return (planted, ())


@pytest.fixture
def llm_cache(tmp_path, monkeypatch):
    temp = Cache(str(tmp_path / "llm"), disk=type(cache.disk))
    monkeypatch.setattr(api_mod, "cache", temp)
    yield temp
    temp.close()


def test_planted_pickle_is_a_cache_miss(llm_cache, runner, monkeypatch):
    llm_calls = []

    def fake_completion(model, messages, **kwargs):
        llm_calls.append(model)
        return tool_reply({"sentiment": "positive"})

    monkeypatch.setattr(api_mod, "completion", fake_completion)
    config = {
        "name": "sentiment",
        "type": "map",
        "prompt": "Classify the sentiment of: {{ input.text }}",
        "output": {"schema": {"sentiment": "string"}},
    }
    data = [{"text": "What a good day."}]

    MapOperation(runner, dict(config), "gpt-4o-mini", 1).execute(data)
    (key,) = list(llm_cache)
    with Cache(llm_cache.directory, disk=Disk) as other_writer:
        other_writer.set(key, Planted())

    results, _ = MapOperation(runner, dict(config), "gpt-4o-mini", 1).execute(data)

    assert planted_calls == []
    assert results[0]["sentiment"] == "positive"
    assert len(llm_calls) == 2


def test_cached_value_types_round_trip(llm_cache):
    stats = CascadeStats(40, 40, 6, 0.15, "recall", 0.9, 0.05, 400, 0.72)
    cascade = CascadeResult([True], [False], stats, [0])
    values = [
        tool_reply({"sentiment": "positive"}),
        EmbeddingResponse(model="text-embedding-3-small", data=[]),
        AgentResult({"sentiment": "positive"}, 0.01),
        (cascade, 0.2, 0.1, 0.1, [0.9], [True], True),
    ]
    for i, value in enumerate(values):
        llm_cache.set(i, value)
        assert llm_cache.get(i) == value
