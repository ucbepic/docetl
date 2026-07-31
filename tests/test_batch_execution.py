import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from docetl.execution import (
    BatchExecutionError,
    BatchRequest,
    BatchResult,
    execute_batch,
    partition_batch_requests,
)
from docetl.operations.map import MapOperation
from docetl.runner import DSLRunner


def _response_row(custom_id: str, value: str) -> dict:
    return {
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "body": {
                "id": f"completion-{custom_id}",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call-{custom_id}",
                                    "type": "function",
                                    "function": {
                                        "name": "send_output",
                                        "arguments": json.dumps({"label": value}),
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            },
        },
        "error": None,
    }


def test_openai_batch_persists_and_orders_results(tmp_path, monkeypatch):
    import litellm

    calls = {"upload": 0, "create": 0, "retrieve": 0, "content": 0}

    def create_file(**kwargs):
        calls["upload"] += 1
        assert kwargs["purpose"] == "batch"
        return SimpleNamespace(id="file-input")

    def create_batch(**kwargs):
        calls["create"] += 1
        assert kwargs["endpoint"] == "/v1/chat/completions"
        return SimpleNamespace(id="batch-1")

    def retrieve_batch(**kwargs):
        calls["retrieve"] += 1
        return SimpleNamespace(
            status="completed", output_file_id="file-output", error_file_id=None
        )

    def file_content(**kwargs):
        calls["content"] += 1
        # Providers do not guarantee output order.
        return SimpleNamespace(
            text="".join(
                json.dumps(row) + "\n"
                for row in [
                    _response_row("second", "B"),
                    _response_row("first", "A"),
                ]
            )
        )

    monkeypatch.setattr(litellm, "create_file", create_file)
    monkeypatch.setattr(litellm, "create_batch", create_batch)
    monkeypatch.setattr(litellm, "retrieve_batch", retrieve_batch)
    monkeypatch.setattr(litellm, "file_content", file_content)

    requests = [
        BatchRequest("first", {"model": "gpt-4o-mini", "messages": []}),
        BatchRequest("second", {"model": "gpt-4o-mini", "messages": []}),
    ]
    config = {"backend": "litellm", "work_dir": str(tmp_path)}
    results = execute_batch(requests, config)

    assert [result.custom_id for result in results] == ["first", "second"]
    assert calls == {"upload": 1, "create": 1, "retrieve": 1, "content": 1}

    # Completed output is a durable checkpoint; a rerun must not resubmit.
    assert execute_batch(requests, config) == results
    assert calls == {"upload": 1, "create": 1, "retrieve": 1, "content": 1}


def test_openai_batch_resumes_existing_provider_job(tmp_path, monkeypatch):
    import litellm
    from docetl.execution import batch as batch_module

    requests = [BatchRequest("one", {"model": "gpt-4o-mini", "messages": []})]
    config = {"backend": "litellm", "work_dir": str(tmp_path)}
    job_dir = batch_module._job_directory(requests, config)
    job_dir.mkdir(parents=True)
    (job_dir / "manifest.json").write_text(
        json.dumps({"batch_id": "existing-batch"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        litellm,
        "create_file",
        lambda **kwargs: pytest.fail("resume must not upload a second file"),
    )
    monkeypatch.setattr(
        litellm,
        "create_batch",
        lambda **kwargs: pytest.fail("resume must not create a second batch"),
    )
    monkeypatch.setattr(
        litellm,
        "retrieve_batch",
        lambda **kwargs: SimpleNamespace(
            status="completed", output_file_id="output", error_file_id=None
        ),
    )
    monkeypatch.setattr(
        litellm,
        "file_content",
        lambda **kwargs: SimpleNamespace(
            text=json.dumps(_response_row("one", "done")) + "\n"
        ),
    )

    assert execute_batch(requests, config)[0].custom_id == "one"


def test_batch_rejects_duplicate_output_ids(tmp_path):
    from docetl.execution import batch as batch_module

    requests = [
        BatchRequest("one", {"model": "m", "messages": []}),
        BatchRequest("two", {"model": "m", "messages": []}),
    ]
    config = {"backend": "vllm", "work_dir": str(tmp_path), "model": "m"}
    job_dir = batch_module._job_directory(requests, config)
    job_dir.mkdir(parents=True)
    duplicate = _response_row("one", "x")
    duplicate["response"] = duplicate["response"]["body"]
    (job_dir / "output.jsonl").write_text(
        json.dumps(duplicate) + "\n" + json.dumps(duplicate) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(BatchExecutionError, match="duplicate custom_id"):
        execute_batch(requests, config)


def test_vllm_batch_uses_safe_argv_and_openai_jsonl(tmp_path, monkeypatch):
    from docetl.execution import batch as batch_module

    captured = {}

    def fake_run(argv, check):
        captured["argv"] = argv
        assert check is True
        input_path = Path(argv[argv.index("-i") + 1])
        output_path = Path(argv[argv.index("-o") + 1])
        request = json.loads(input_path.read_text(encoding="utf-8"))
        assert request["body"]["model"] == "Qwen/Qwen3-8B"
        assert request["body"]["max_tokens"] == 512
        response = _response_row("one", "local")
        # vLLM writes the endpoint body directly rather than OpenAI's wrapper.
        response["response"] = response["response"]["body"]
        output_path.write_text(json.dumps(response) + "\n", encoding="utf-8")

    monkeypatch.setattr(batch_module.subprocess, "run", fake_run)
    results = execute_batch(
        [BatchRequest("one", {"model": "ignored", "messages": []})],
        {
            "backend": "vllm",
            "work_dir": str(tmp_path),
            "model": "Qwen/Qwen3-8B",
            "request_defaults": {"max_tokens": 512},
            "engine_args": {
                "tensor_parallel_size": 2,
                "gpu_memory_utilization": 0.9,
                "enforce_eager": True,
            },
        },
    )

    assert results[0].body["model"] == "gpt-4o-mini"
    assert captured["argv"][:2] == ["vllm", "run-batch"]
    assert "--tensor-parallel-size" in captured["argv"]
    assert "--enforce-eager" in captured["argv"]


def test_batch_partition_respects_count_and_encoded_bytes():
    requests = [
        BatchRequest(str(index), {"model": "m", "messages": [], "value": "x" * 20})
        for index in range(3)
    ]
    one_size = len(
        (json.dumps(requests[0].as_dict(), sort_keys=True) + "\n").encode("utf-8")
    )

    assert [
        len(chunk) for chunk in partition_batch_requests(requests, max_requests=2)
    ] == [2, 1]
    assert [
        len(chunk)
        for chunk in partition_batch_requests(
            requests, max_requests=10, max_bytes=one_size * 2 - 1
        )
    ] == [1, 1, 1]


def test_map_materializes_one_request_per_row_and_preserves_order(monkeypatch):
    import docetl.operations.utils.api as api_module

    seen = {}

    def fake_execute_batch(requests, config):
        seen["requests"] = requests
        seen["config"] = config
        return [
            BatchResult(
                request.custom_id,
                _response_row(request.custom_id, value)["response"]["body"],
            )
            for request, value in zip(requests, ["A", "B"])
        ]

    monkeypatch.setattr(api_module, "execute_batch", fake_execute_batch)
    runner = DSLRunner(
        {
            "default_model": "gpt-4o-mini",
            "operations": [],
            "pipeline": {"steps": [], "output": {"path": "/tmp/out.json"}},
        },
        max_threads=2,
    )
    operation = MapOperation(
        runner,
        {
            "name": "classify",
            "type": "map",
            "prompt": "Classify {{ input.text }}",
            "output": {"schema": {"label": "string"}},
            "execution": {"mode": "batch", "backend": "litellm"},
            "bypass_cache": True,
        },
        "gpt-4o-mini",
        2,
    )

    output, _ = operation.execute([{"text": "one"}, {"text": "two"}])

    assert output == [
        {"text": "one", "label": "A"},
        {"text": "two", "label": "B"},
    ]
    assert len(seen["requests"]) == 2
    assert seen["requests"][0].body["tools"][0]["function"]["name"] == "send_output"


def test_map_splits_provider_request_limit(monkeypatch):
    import docetl.operations.utils.api as api_module

    batch_sizes = []

    def fake_execute_batch(requests, config):
        batch_sizes.append(len(requests))
        return [
            BatchResult(
                request.custom_id,
                _response_row(request.custom_id, "ok")["response"]["body"],
            )
            for request in requests
        ]

    monkeypatch.setattr(api_module, "execute_batch", fake_execute_batch)
    runner = DSLRunner(
        {
            "default_model": "gpt-4o-mini",
            "operations": [],
            "pipeline": {"steps": [], "output": {"path": "/tmp/out.json"}},
        }
    )
    operation = MapOperation(
        runner,
        {
            "name": "classify",
            "type": "map",
            "prompt": "Classify {{ input.text }}",
            "output": {"schema": {"label": "string"}},
            "execution": {
                "mode": "batch",
                "backend": "litellm",
                "max_batch_requests": 2,
            },
            "bypass_cache": True,
        },
        "gpt-4o-mini",
        2,
    )

    output, _ = operation.execute([{"text": str(i)} for i in range(5)])

    assert len(output) == 5
    assert batch_sizes == [2, 2, 1]


def test_litellm_provider_is_inferred_from_model(monkeypatch):
    import docetl.operations.utils.api as api_module

    captured = {}

    def fake_execute_batch(requests, config):
        captured["provider"] = config["provider"]
        captured["model"] = requests[0].body["model"]
        return [
            BatchResult(
                requests[0].custom_id,
                _response_row(requests[0].custom_id, "ok")["response"]["body"],
            )
        ]

    monkeypatch.setattr(api_module, "execute_batch", fake_execute_batch)
    runner = DSLRunner(
        {
            "default_model": "vertex_ai/gemini-2.5-flash",
            "operations": [],
            "pipeline": {"steps": [], "output": {"path": "/tmp/out.json"}},
        }
    )
    operation = MapOperation(
        runner,
        {
            "name": "classify",
            "type": "map",
            "model": "vertex_ai/gemini-2.5-flash",
            "prompt": "Classify {{ input.text }}",
            "output": {"schema": {"label": "string"}},
            "execution": {"mode": "batch", "backend": "litellm"},
            "bypass_cache": True,
        },
        "vertex_ai/gemini-2.5-flash",
        1,
    )

    operation.execute([{"text": "one"}])

    assert captured == {"provider": "vertex_ai", "model": "gemini-2.5-flash"}


def test_vllm_deepseek_string_request_preserves_no_tool_call_path():
    runner = DSLRunner(
        {
            "default_model": "deepseek-r1",
            "operations": [],
            "pipeline": {"steps": [], "output": {"path": "/tmp/out.json"}},
        }
    )

    body = runner.api._prepare_materialized_completion(
        model="deepseek-r1",
        op_type="map",
        messages=[{"role": "user", "content": "Summarize"}],
        output_schema={"summary": "string"},
        litellm_completion_kwargs={},
        op_config={"output": {"schema": {"summary": "string"}}},
        execution_config={
            "mode": "batch",
            "backend": "vllm",
            "model": "deepseek-ai/DeepSeek-R1",
        },
    )

    assert body["model"] == "deepseek-ai/DeepSeek-R1"
    assert "tools" not in body
    assert "tool_choice" not in body


def test_inferred_unsupported_litellm_batch_provider_is_rejected():
    runner = DSLRunner(
        {
            "default_model": "anthropic/claude-sonnet-4-5",
            "operations": [],
            "pipeline": {"steps": [], "output": {"path": "/tmp/out.json"}},
        }
    )

    with pytest.raises(ValueError, match="currently supports provider values"):
        runner.api.call_llm_materialized_batch(
            [
                {
                    "model": "anthropic/claude-sonnet-4-5",
                    "op_type": "map",
                    "messages": [{"role": "user", "content": "Classify"}],
                    "output_schema": {"label": "string"},
                    "op_config": {"output": {"schema": {"label": "string"}}},
                    "bypass_cache": True,
                }
            ],
            {"mode": "batch", "backend": "litellm"},
        )


@pytest.mark.parametrize("field", ["agent", "gleaning", "validate", "calibrate"])
def test_batch_execution_rejects_multi_round_map_features(field):
    config = {
        "name": "classify",
        "type": "map",
        "prompt": "Classify {{ input.text }}",
        "output": {"schema": {"label": "string"}},
        "execution": {"mode": "batch", "backend": "litellm"},
    }
    config[field] = (
        {"num_rounds": 1, "validation_prompt": "check"}
        if field == "gleaning"
        else ["output['label']"]
        if field == "validate"
        else True
    )
    with pytest.raises(ValueError, match="cannot yet be combined"):
        MapOperation.schema.model_validate(config)
