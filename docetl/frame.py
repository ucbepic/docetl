"""PySpark-like API for docetl pipelines.

Usage::

    import docetl

    docetl.default_model = "gpt-4o-mini"

    results = (
        docetl.read_json("input.json")
        .map(prompt="Summarize: {{ input.text }}",
             output={"schema": {"summary": "string"}})
        .filter(prompt="Is this good? {{ input.summary }}",
                output={"schema": {"keep": "boolean"}})
        .collect()
    )
"""

from __future__ import annotations

import os
from typing import Any

from docetl import _config


class Frame:
    """A lazy pipeline frame—operations are recorded but not executed
    until a terminal action (``.collect()``, ``.write_json()``, etc.) is called."""

    def __init__(
        self,
        datasets: dict[str, dict[str, Any]],
        operations: list[dict[str, Any]] | None = None,
        steps: list[dict[str, Any]] | None = None,
        *,
        _last_step: str | None = None,
        _first_dataset: str | None = None,
        _op_counter: dict[str, int] | None = None,
        _extra_datasets: dict[str, dict[str, Any]] | None = None,
    ):
        self._datasets = datasets
        self._operations = operations or []
        self._steps = steps or []
        self._last_step = _last_step
        self._first_dataset = _first_dataset or next(iter(datasets), None)
        self._op_counter = _op_counter or {}
        self._extra_datasets = _extra_datasets or {}

    def _copy(self, **overrides) -> Frame:
        kw = dict(
            datasets=self._datasets,
            operations=list(self._operations),
            steps=list(self._steps),
            _last_step=self._last_step,
            _first_dataset=self._first_dataset,
            _op_counter=dict(self._op_counter),
            _extra_datasets=dict(self._extra_datasets),
        )
        kw.update(overrides)
        return Frame(**kw)

    # ── auto-naming ────────────────────────────────────────────────

    def _auto_name(self, op_type: str) -> tuple[str, dict[str, int]]:
        counter = dict(self._op_counter)
        counter[op_type] = counter.get(op_type, 0) + 1
        return f"{op_type}_{counter[op_type]}", counter

    def _append_op(self, op_type: str, name: str | None, config: dict[str, Any]) -> Frame:
        name_val, new_counter = self._auto_name(op_type) if name is None else (name, self._op_counter)
        op = {"name": name_val, "type": op_type, **{k: v for k, v in config.items() if v is not None}}

        step_input = self._last_step or self._first_dataset
        step_name = f"step_{name_val}"
        step: dict[str, Any] = {"name": step_name, "operations": [name_val]}
        if step_input:
            step["input"] = step_input

        new = self._copy(
            operations=self._operations + [op],
            steps=self._steps + [step],
            _last_step=step_name,
            _op_counter=new_counter if name is None else dict(self._op_counter),
        )
        return new

    def _append_equijoin(self, name: str, left: str, right: str, config: dict[str, Any]) -> Frame:
        op = {"name": name, "type": "equijoin", **{k: v for k, v in config.items() if v is not None}}
        step_name = f"step_{name}"
        step: dict[str, Any] = {
            "name": step_name,
            "operations": [{name: {"left": left, "right": right}}],
        }

        return self._copy(
            operations=self._operations + [op],
            steps=self._steps + [step],
            _last_step=step_name,
        )

    # ── LLM operations ─────────────────────────────────────────────

    def map(
        self,
        name: str | None = None,
        *,
        prompt: str | None = None,
        output: dict[str, Any] | None = None,
        model: str | None = None,
        optimize: bool | None = None,
        recursively_optimize: bool | None = None,
        sample_size: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        validation_rules: list[str] | None = None,
        num_retries_on_validate_failure: int | None = None,
        drop_keys: list[str] | None = None,
        timeout: int | None = None,
        enable_observability: bool | None = None,
        batch_size: int | None = None,
        clustering_method: str | None = None,
        batch_prompt: str | None = None,
        litellm_completion_kwargs: dict[str, Any] | None = None,
        pdf_url_key: str | None = None,
        flush_partial_result: bool | None = None,
        limit: int | None = None,
        calibrate: bool | None = None,
        num_calibration_docs: int | None = None,
        **kwargs: Any,
    ) -> Frame:
        return self._append_op("map", name, {
            "prompt": prompt, "output": output, "model": model,
            "optimize": optimize, "recursively_optimize": recursively_optimize,
            "sample_size": sample_size, "tools": tools,
            "validation_rules": validation_rules,
            "num_retries_on_validate_failure": num_retries_on_validate_failure,
            "drop_keys": drop_keys, "timeout": timeout,
            "enable_observability": enable_observability,
            "batch_size": batch_size, "clustering_method": clustering_method,
            "batch_prompt": batch_prompt,
            "litellm_completion_kwargs": litellm_completion_kwargs,
            "pdf_url_key": pdf_url_key,
            "flush_partial_result": flush_partial_result,
            "limit": limit, "calibrate": calibrate,
            "num_calibration_docs": num_calibration_docs,
            **kwargs,
        })

    def parallel_map(
        self,
        name: str | None = None,
        *,
        prompts: list[dict[str, Any]] | None = None,
        output: dict[str, Any] | None = None,
        drop_keys: list[str] | None = None,
        enable_observability: bool | None = None,
        pdf_url_key: str | None = None,
        **kwargs: Any,
    ) -> Frame:
        return self._append_op("parallel_map", name, {
            "prompts": prompts, "output": output, "drop_keys": drop_keys,
            "enable_observability": enable_observability,
            "pdf_url_key": pdf_url_key, **kwargs,
        })

    def filter(
        self,
        name: str | None = None,
        *,
        prompt: str | None = None,
        output: dict[str, Any] | None = None,
        model: str | None = None,
        optimize: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        validation_rules: list[str] | None = None,
        drop_keys: list[str] | None = None,
        timeout: int | None = None,
        batch_size: int | None = None,
        litellm_completion_kwargs: dict[str, Any] | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> Frame:
        return self._append_op("filter", name, {
            "prompt": prompt, "output": output, "model": model,
            "optimize": optimize, "tools": tools,
            "validation_rules": validation_rules, "drop_keys": drop_keys,
            "timeout": timeout, "batch_size": batch_size,
            "litellm_completion_kwargs": litellm_completion_kwargs,
            "limit": limit, **kwargs,
        })

    def reduce(
        self,
        name: str | None = None,
        *,
        reduce_key: str | list[str] | None = None,
        prompt: str | None = None,
        output: dict[str, Any] | None = None,
        model: str | None = None,
        input: dict[str, Any] | None = None,
        optimize: bool | None = None,
        synthesize_resolve: bool | None = None,
        pass_through: bool | None = None,
        associative: bool | None = None,
        fold_prompt: str | None = None,
        fold_batch_size: int | None = None,
        merge_prompt: str | None = None,
        merge_batch_size: int | None = None,
        value_sampling: dict[str, Any] | None = None,
        verbose: bool | None = None,
        timeout: int | None = None,
        litellm_completion_kwargs: dict[str, Any] | None = None,
        enable_observability: bool | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> Frame:
        return self._append_op("reduce", name, {
            "reduce_key": reduce_key, "prompt": prompt, "output": output,
            "model": model, "input": input, "optimize": optimize,
            "synthesize_resolve": synthesize_resolve,
            "pass_through": pass_through, "associative": associative,
            "fold_prompt": fold_prompt, "fold_batch_size": fold_batch_size,
            "merge_prompt": merge_prompt, "merge_batch_size": merge_batch_size,
            "value_sampling": value_sampling, "verbose": verbose,
            "timeout": timeout,
            "litellm_completion_kwargs": litellm_completion_kwargs,
            "enable_observability": enable_observability,
            "limit": limit, **kwargs,
        })

    def resolve(
        self,
        name: str | None = None,
        *,
        comparison_prompt: str | None = None,
        resolution_prompt: str | None = None,
        output: dict[str, Any] | None = None,
        embedding_model: str | None = None,
        resolution_model: str | None = None,
        comparison_model: str | None = None,
        blocking_keys: list[str] | None = None,
        blocking_threshold: float | None = None,
        blocking_target_recall: float | None = None,
        blocking_conditions: list[str] | None = None,
        input: dict[str, Any] | None = None,
        embedding_batch_size: int | None = None,
        compare_batch_size: int | None = None,
        limit_comparisons: int | None = None,
        optimize: bool | None = None,
        timeout: int | None = None,
        litellm_completion_kwargs: dict[str, Any] | None = None,
        enable_observability: bool | None = None,
        **kwargs: Any,
    ) -> Frame:
        return self._append_op("resolve", name, {
            "comparison_prompt": comparison_prompt,
            "resolution_prompt": resolution_prompt, "output": output,
            "embedding_model": embedding_model,
            "resolution_model": resolution_model,
            "comparison_model": comparison_model,
            "blocking_keys": blocking_keys,
            "blocking_threshold": blocking_threshold,
            "blocking_target_recall": blocking_target_recall,
            "blocking_conditions": blocking_conditions,
            "input": input, "embedding_batch_size": embedding_batch_size,
            "compare_batch_size": compare_batch_size,
            "limit_comparisons": limit_comparisons, "optimize": optimize,
            "timeout": timeout,
            "litellm_completion_kwargs": litellm_completion_kwargs,
            "enable_observability": enable_observability, **kwargs,
        })

    def equijoin(
        self,
        right: Frame,
        name: str | None = None,
        *,
        comparison_prompt: str | None = None,
        output: dict[str, Any] | None = None,
        blocking_threshold: float | None = None,
        blocking_target_recall: float | None = None,
        blocking_conditions: list[str] | None = None,
        limits: dict[str, int] | None = None,
        comparison_model: str | None = None,
        optimize: bool | None = None,
        embedding_model: str | None = None,
        embedding_batch_size: int | None = None,
        compare_batch_size: int | None = None,
        limit_comparisons: int | None = None,
        blocking_keys: dict[str, list[str]] | None = None,
        timeout: int | None = None,
        litellm_completion_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Frame:
        if name is None:
            name, _ = self._auto_name("equijoin")

        all_datasets = {**self._datasets, **right._datasets}
        left_ds = self._first_dataset
        right_ds = right._first_dataset

        config = {
            k: v for k, v in {
                "comparison_prompt": comparison_prompt, "output": output,
                "blocking_threshold": blocking_threshold,
                "blocking_target_recall": blocking_target_recall,
                "blocking_conditions": blocking_conditions, "limits": limits,
                "comparison_model": comparison_model, "optimize": optimize,
                "embedding_model": embedding_model,
                "embedding_batch_size": embedding_batch_size,
                "compare_batch_size": compare_batch_size,
                "limit_comparisons": limit_comparisons,
                "blocking_keys": blocking_keys, "timeout": timeout,
                "litellm_completion_kwargs": litellm_completion_kwargs,
                **kwargs,
            }.items() if v is not None
        }

        new = self._copy(datasets=all_datasets)
        return new._append_equijoin(name, left_ds, right_ds, config)

    def extract(
        self,
        name: str | None = None,
        *,
        prompt: str | None = None,
        document_keys: list[str] | None = None,
        model: str | None = None,
        format_extraction: bool | None = None,
        extraction_key_suffix: str | None = None,
        extraction_method: str | None = None,
        timeout: int | None = None,
        litellm_completion_kwargs: dict[str, Any] | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> Frame:
        return self._append_op("extract", name, {
            "prompt": prompt, "document_keys": document_keys, "model": model,
            "format_extraction": format_extraction,
            "extraction_key_suffix": extraction_key_suffix,
            "extraction_method": extraction_method, "timeout": timeout,
            "litellm_completion_kwargs": litellm_completion_kwargs,
            "limit": limit, **kwargs,
        })

    # ── structural operations ──────────────────────────────────────

    def split(
        self,
        name: str | None = None,
        *,
        split_key: str | None = None,
        method: str | None = None,
        method_kwargs: dict[str, Any] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> Frame:
        return self._append_op("split", name, {
            "split_key": split_key, "method": method,
            "method_kwargs": method_kwargs, "model": model, **kwargs,
        })

    def gather(
        self,
        name: str | None = None,
        *,
        content_key: str | None = None,
        doc_id_key: str | None = None,
        order_key: str | None = None,
        peripheral_chunks: dict[str, Any] | None = None,
        doc_header_key: str | None = None,
        main_chunk_start: str | None = None,
        main_chunk_end: str | None = None,
        **kwargs: Any,
    ) -> Frame:
        return self._append_op("gather", name, {
            "content_key": content_key, "doc_id_key": doc_id_key,
            "order_key": order_key, "peripheral_chunks": peripheral_chunks,
            "doc_header_key": doc_header_key,
            "main_chunk_start": main_chunk_start,
            "main_chunk_end": main_chunk_end, **kwargs,
        })

    def unnest(
        self,
        name: str | None = None,
        *,
        unnest_key: str | None = None,
        keep_empty: bool | None = None,
        expand_fields: list[str] | None = None,
        recursive: bool | None = None,
        depth: int | None = None,
        **kwargs: Any,
    ) -> Frame:
        return self._append_op("unnest", name, {
            "unnest_key": unnest_key, "keep_empty": keep_empty,
            "expand_fields": expand_fields, "recursive": recursive,
            "depth": depth, **kwargs,
        })

    def cluster(
        self,
        name: str | None = None,
        *,
        embedding_keys: list[str] | None = None,
        summary_schema: dict[str, Any] | None = None,
        summary_prompt: str | None = None,
        output_key: str | None = None,
        model: str | None = None,
        embedding_model: str | None = None,
        max_batch_size: int | None = None,
        validate: list[str] | None = None,
        **kwargs: Any,
    ) -> Frame:
        return self._append_op("cluster", name, {
            "embedding_keys": embedding_keys,
            "summary_schema": summary_schema,
            "summary_prompt": summary_prompt, "output_key": output_key,
            "model": model, "embedding_model": embedding_model,
            "max_batch_size": max_batch_size, "validate": validate,
            **kwargs,
        })

    def sample(
        self,
        name: str | None = None,
        *,
        method: str | None = None,
        samples: int | float | list | None = None,
        stratify_key: str | list[str] | None = None,
        samples_per_group: bool | None = None,
        method_kwargs: dict[str, Any] | None = None,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> Frame:
        return self._append_op("sample", name, {
            "method": method, "samples": samples,
            "stratify_key": stratify_key,
            "samples_per_group": samples_per_group,
            "method_kwargs": method_kwargs, "random_state": random_state,
            **kwargs,
        })

    def code_map(
        self,
        name: str | None = None,
        *,
        code: str | None = None,
        drop_keys: list[str] | None = None,
        concurrent_thread_count: int | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> Frame:
        return self._append_op("code_map", name, {
            "code": code, "drop_keys": drop_keys,
            "concurrent_thread_count": concurrent_thread_count,
            "limit": limit, **kwargs,
        })

    def code_reduce(
        self,
        name: str | None = None,
        *,
        code: str | None = None,
        concurrent_thread_count: int | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> Frame:
        return self._append_op("code_reduce", name, {
            "code": code, "concurrent_thread_count": concurrent_thread_count,
            "limit": limit, **kwargs,
        })

    def code_filter(
        self,
        name: str | None = None,
        *,
        code: str | None = None,
        concurrent_thread_count: int | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> Frame:
        return self._append_op("code_filter", name, {
            "code": code, "concurrent_thread_count": concurrent_thread_count,
            "limit": limit, **kwargs,
        })

    # ── terminal actions ───────────────────────────────────────────

    def _build_runner(self, output_path: str = "", max_threads: int | None = None):
        from docetl.runner import DSLRunner
        from docetl.schemas import Dataset as DatasetSchema

        config: dict[str, Any] = {
            "datasets": {},
            "operations": self._operations,
            "pipeline": {
                "steps": self._steps,
                "output": {"type": "file", "path": output_path},
            },
        }

        if _config.default_model:
            config["default_model"] = _config.default_model
        if _config.rate_limits:
            config["rate_limits"] = _config.rate_limits

        for ds_name, ds_cfg in self._datasets.items():
            config["datasets"][ds_name] = ds_cfg

        return DSLRunner(config, max_threads=max_threads)

    def collect(self, max_threads: int | None = None) -> "pd.DataFrame":
        """Execute the pipeline and return results as a DataFrame."""
        import pandas as pd
        data = self._execute(max_threads=max_threads)
        return pd.DataFrame(data)

    def to_list(self, max_threads: int | None = None) -> list[dict]:
        """Execute the pipeline and return results as a list of dicts."""
        return self._execute(max_threads=max_threads)

    def _execute(self, max_threads: int | None = None) -> list[dict]:
        runner = self._build_runner(max_threads=max_threads)
        runner.load()
        runner.console.rule("[bold]Pipeline Execution[/bold]")
        output, _, _ = runner.last_op_container.next()
        return output

    def write_json(self, path: str, max_threads: int | None = None) -> None:
        """Execute the pipeline and write results to a JSON file."""
        runner = self._build_runner(output_path=path, max_threads=max_threads)
        runner.load_run_save()

    def write_csv(self, path: str, max_threads: int | None = None) -> None:
        """Execute the pipeline and write results to a CSV file."""
        runner = self._build_runner(output_path=path, max_threads=max_threads)
        runner.load_run_save()

    def write_parquet(self, path: str, max_threads: int | None = None) -> None:
        """Execute the pipeline and write results to a Parquet file."""
        runner = self._build_runner(output_path=path, max_threads=max_threads)
        runner.load_run_save()


# ── reader entry points ────────────────────────────────────────────

def read_json(path: str, *, parsing: list[dict[str, str]] | None = None) -> Frame:
    """Read a JSON file and return a Frame."""
    name = os.path.splitext(os.path.basename(path))[0]
    ds: dict[str, Any] = {"type": "file", "path": path}
    if parsing:
        ds["parsing"] = parsing
    return Frame({name: ds}, _first_dataset=name)


def read_csv(path: str, *, parsing: list[dict[str, str]] | None = None) -> Frame:
    """Read a CSV file and return a Frame."""
    name = os.path.splitext(os.path.basename(path))[0]
    ds: dict[str, Any] = {"type": "file", "path": path}
    if parsing:
        ds["parsing"] = parsing
    return Frame({name: ds}, _first_dataset=name)


def read_parquet(path: str, *, parsing: list[dict[str, str]] | None = None) -> Frame:
    """Read a Parquet file and return a Frame."""
    name = os.path.splitext(os.path.basename(path))[0]
    ds: dict[str, Any] = {"type": "file", "path": path}
    if parsing:
        ds["parsing"] = parsing
    return Frame({name: ds}, _first_dataset=name)


def from_list(data: list[dict], name: str = "data") -> Frame:
    """Create a Frame from an in-memory list of dicts."""
    return Frame({name: {"type": "memory", "path": data}}, _first_dataset=name)
