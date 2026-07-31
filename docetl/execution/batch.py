"""Durable execution of OpenAI-compatible JSONL request batches.

Both OpenAI's Batch API and vLLM's ``run-batch`` command consume nearly the
same request envelope.  Keeping that envelope at the DocETL boundary lets an
operation materialize requests once and choose hosted or self-hosted execution
without changing its prompt or result parsing.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


TERMINAL_BATCH_STATES = {"completed", "failed", "expired", "cancelled"}
SUPPORTED_LITELLM_BATCH_PROVIDERS = frozenset(
    {"openai", "azure", "vertex_ai", "bedrock", "hosted_vllm"}
)


class BatchExecutionError(RuntimeError):
    """Raised when a materialized batch cannot produce all requested results."""


@dataclass(frozen=True)
class BatchRequest:
    """One request in the OpenAI Batch JSONL format."""

    custom_id: str
    body: dict[str, Any]
    url: str = "/v1/chat/completions"

    def as_dict(self) -> dict[str, Any]:
        return {
            "custom_id": self.custom_id,
            "method": "POST",
            "url": self.url,
            "body": self.body,
        }


@dataclass(frozen=True)
class BatchResult:
    """A provider-neutral response for one materialized request."""

    custom_id: str
    body: dict[str, Any] | None
    error: Any | None = None
    status_code: int | None = None


def _atomic_write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def _jsonl(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def partition_batch_requests(
    requests: list[BatchRequest],
    *,
    max_requests: int,
    max_bytes: int | None = None,
) -> list[list[BatchRequest]]:
    """Partition requests without exceeding provider count or file limits."""
    if max_requests <= 0:
        raise ValueError("max_requests must be greater than zero")
    if max_bytes is not None and max_bytes <= 0:
        raise ValueError("max_bytes must be greater than zero")

    chunks: list[list[BatchRequest]] = []
    current: list[BatchRequest] = []
    current_bytes = 0
    for request in requests:
        request_bytes = len(
            (json.dumps(request.as_dict(), sort_keys=True) + "\n").encode("utf-8")
        )
        if max_bytes is not None and request_bytes > max_bytes:
            raise BatchExecutionError(
                f"Batch request {request.custom_id!r} is {request_bytes} bytes, "
                f"exceeding max_batch_bytes={max_bytes}"
            )
        if current and (
            len(current) >= max_requests
            or (max_bytes is not None and current_bytes + request_bytes > max_bytes)
        ):
            chunks.append(current)
            current = []
            current_bytes = 0
        current.append(request)
        current_bytes += request_bytes
    if current:
        chunks.append(current)
    return chunks


def _job_directory(requests: list[BatchRequest], config: dict[str, Any]) -> Path:
    # Exclude polling/lifecycle settings: changing how often we check must not
    # create and bill a duplicate provider job for the same logical requests.
    identity_config = {
        key: value
        for key, value in config.items()
        if key not in {"poll_interval_seconds", "timeout_seconds", "work_dir"}
    }
    material = _jsonl(request.as_dict() for request in requests)
    digest = hashlib.sha256(
        (
            json.dumps(identity_config, sort_keys=True, default=str) + "\n" + material
        ).encode()
    ).hexdigest()[:20]
    return Path(config.get("work_dir", ".docetl/batches")) / digest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise BatchExecutionError(
                f"Invalid JSON on line {line_number} of {path}"
            ) from exc
    return rows


def _result_from_row(row: dict[str, Any]) -> BatchResult:
    response = row.get("response")
    if response is None:
        return BatchResult(row["custom_id"], None, row.get("error"))

    # OpenAI wraps the endpoint response in {status_code, body}; vLLM's
    # offline runner writes the endpoint response directly.
    if isinstance(response, dict) and "body" in response:
        return BatchResult(
            custom_id=row["custom_id"],
            body=response.get("body"),
            error=row.get("error"),
            status_code=response.get("status_code"),
        )
    return BatchResult(
        custom_id=row["custom_id"],
        body=response,
        error=row.get("error"),
        status_code=200,
    )


def _ordered_results(
    requests: list[BatchRequest], rows: list[dict[str, Any]]
) -> list[BatchResult]:
    parsed = [_result_from_row(row) for row in rows]
    if len({result.custom_id for result in parsed}) != len(parsed):
        raise BatchExecutionError("Batch output contains duplicate custom_id values")
    by_id = {result.custom_id: result for result in parsed}
    missing = [
        request.custom_id for request in requests if request.custom_id not in by_id
    ]
    if missing:
        preview = ", ".join(missing[:5])
        raise BatchExecutionError(
            f"Batch output is missing {len(missing)} request(s): {preview}"
        )
    return [by_id[request.custom_id] for request in requests]


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _content_text(content: Any) -> str:
    if isinstance(content, bytes):
        return content.decode("utf-8")
    if isinstance(content, str):
        return content
    text = getattr(content, "text", None)
    if callable(text):
        text = text()
    if isinstance(text, str):
        return text
    raw = getattr(content, "content", None)
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    raise BatchExecutionError("Batch file response did not contain text content")


def _provider_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(config.get("provider_kwargs", {}))
    kwargs["custom_llm_provider"] = config.get("provider", "openai")
    return kwargs


def _execute_provider_batch(
    requests: list[BatchRequest], config: dict[str, Any], job_dir: Path
) -> list[BatchResult]:
    from litellm import create_batch, create_file, file_content, retrieve_batch

    input_path = job_dir / "input.jsonl"
    output_path = job_dir / "output.jsonl"
    error_path = job_dir / "errors.jsonl"
    manifest_path = job_dir / "manifest.json"
    if output_path.exists():
        rows = _read_jsonl(output_path)
        if error_path.exists():
            rows.extend(_read_jsonl(error_path))
        return _ordered_results(requests, rows)

    _atomic_write(input_path, _jsonl(request.as_dict() for request in requests))
    provider_kwargs = _provider_kwargs(config)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        batch_id = manifest["batch_id"]
    else:
        if config.get("routing_model"):
            provider_kwargs["model"] = config["routing_model"]
        with input_path.open("rb") as input_file:
            uploaded = create_file(
                file=input_file,
                purpose="batch",
                **provider_kwargs,
            )
        batch = create_batch(
            completion_window=config.get("completion_window", "24h"),
            endpoint=requests[0].url,
            input_file_id=_get(uploaded, "id"),
            metadata=config.get("metadata"),
            **provider_kwargs,
        )
        batch_id = _get(batch, "id")
        _atomic_write(
            manifest_path,
            json.dumps(
                {
                    "backend": "litellm",
                    "batch_id": batch_id,
                    "input_file_id": _get(uploaded, "id"),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )

    started = time.monotonic()
    poll_interval = float(config.get("poll_interval_seconds", 30))
    timeout_seconds = config.get("timeout_seconds")
    while True:
        batch = retrieve_batch(batch_id=batch_id, **provider_kwargs)
        status = _get(batch, "status")
        if status in TERMINAL_BATCH_STATES:
            break
        if timeout_seconds is not None and time.monotonic() - started > float(
            timeout_seconds
        ):
            raise TimeoutError(
                f"Batch {batch_id} did not finish within {timeout_seconds} seconds; "
                f"rerun the pipeline to resume it"
            )
        time.sleep(poll_interval)

    output_file_id = _get(batch, "output_file_id")
    error_file_id = _get(batch, "error_file_id")
    if output_file_id:
        _atomic_write(
            output_path,
            _content_text(file_content(file_id=output_file_id, **provider_kwargs)),
        )
    if error_file_id:
        _atomic_write(
            error_path,
            _content_text(file_content(file_id=error_file_id, **provider_kwargs)),
        )
    if status != "completed" and not output_path.exists():
        raise BatchExecutionError(f"Batch {batch_id} ended with status {status!r}")

    rows = _read_jsonl(output_path) if output_path.exists() else []
    if error_path.exists():
        rows.extend(_read_jsonl(error_path))
    return _ordered_results(requests, rows)


def _cli_args(engine_args: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in engine_args.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(flag)
        elif isinstance(value, list):
            for item in value:
                args.extend([flag, str(item)])
        elif value is not None:
            args.extend([flag, str(value)])
    return args


def _execute_vllm_batch(
    requests: list[BatchRequest], config: dict[str, Any], job_dir: Path
) -> list[BatchResult]:
    input_path = job_dir / "input.jsonl"
    output_path = job_dir / "output.jsonl"
    if output_path.exists():
        return _ordered_results(requests, _read_jsonl(output_path))

    model = config.get("model")
    if not model:
        models = {request.body.get("model") for request in requests}
        if len(models) != 1:
            raise ValueError("vLLM batch execution requires exactly one model")
        model = models.pop()

    request_defaults = config.get("request_defaults", {})
    materialized = []
    for request in requests:
        body = {**request_defaults, **request.body, "model": model}
        materialized.append(BatchRequest(request.custom_id, body, request.url))
    _atomic_write(input_path, _jsonl(item.as_dict() for item in materialized))

    command = config.get("command", ["vllm", "run-batch"])
    if not isinstance(command, list) or not all(
        isinstance(part, str) for part in command
    ):
        raise ValueError("vLLM batch 'command' must be a list of strings")
    argv = [
        *command,
        "-i",
        str(input_path),
        "-o",
        str(output_path),
        "--model",
        str(model),
        *_cli_args(config.get("engine_args", {})),
    ]
    try:
        subprocess.run(argv, check=True)
    except FileNotFoundError as exc:
        raise BatchExecutionError(
            f"vLLM batch command was not found: {command[0]!r}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise BatchExecutionError(
            f"vLLM batch command exited with status {exc.returncode}; "
            f"input remains at {input_path}"
        ) from exc
    if not output_path.exists():
        raise BatchExecutionError("vLLM batch command did not create its output file")
    return _ordered_results(requests, _read_jsonl(output_path))


def execute_batch(
    requests: list[BatchRequest], config: dict[str, Any]
) -> list[BatchResult]:
    """Execute *requests* with the configured durable batch backend."""
    if not requests:
        return []
    endpoints = {request.url for request in requests}
    if len(endpoints) != 1:
        raise ValueError("A batch may target only one endpoint")
    ids = [request.custom_id for request in requests]
    if len(ids) != len(set(ids)):
        raise ValueError("Batch request custom_id values must be unique")

    job_dir = _job_directory(requests, config)
    backend = config.get("backend", "litellm")
    if backend == "litellm":
        return _execute_provider_batch(requests, config, job_dir)
    if backend == "vllm":
        return _execute_vllm_batch(requests, config, job_dir)
    raise ValueError(f"Unknown batch execution backend {backend!r}")
