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


class Retriever:
    """A LanceDB retriever configuration. Pass to operations via ``retriever=``."""

    _counter = 0

    def __init__(
        self,
        dataset: str,
        index_dir: str,
        index_types: list[str],
        *,
        fts: dict[str, str] | None = None,
        embedding: dict[str, str] | None = None,
        query: dict[str, Any] | None = None,
        build_index: str = "if_missing",
    ):
        Retriever._counter += 1
        self._name = f"retriever_{Retriever._counter}"
        self._config: dict[str, Any] = {
            "type": "lancedb",
            "dataset": dataset,
            "index_dir": index_dir,
            "index_types": index_types,
            "build_index": build_index,
        }
        if fts is not None:
            self._config["fts"] = fts
        if embedding is not None:
            self._config["embedding"] = embedding
        if query is not None:
            self._config["query"] = query


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
        _retrievers: dict[str, dict[str, Any]] | None = None,
    ):
        self._datasets = datasets
        self._operations = operations or []
        self._steps = steps or []
        self._last_step = _last_step
        self._first_dataset = _first_dataset or next(iter(datasets), None)
        self._op_counter = _op_counter or {}
        self._extra_datasets = _extra_datasets or {}
        self._retrievers = _retrievers or {}
        self._total_cost: float = 0.0
        self._token_usage: dict[str, dict[str, int]] = {}

    def _copy(self, **overrides) -> Frame:
        kw = dict(
            datasets=self._datasets,
            operations=list(self._operations),
            steps=list(self._steps),
            _last_step=self._last_step,
            _first_dataset=self._first_dataset,
            _op_counter=dict(self._op_counter),
            _extra_datasets=dict(self._extra_datasets),
            _retrievers=dict(self._retrievers),
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

        new_retrievers = dict(self._retrievers)
        retriever = config.get("retriever")
        if isinstance(retriever, Retriever):
            new_retrievers[retriever._name] = retriever._config
            config = {**config, "retriever": retriever._name}

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
            _retrievers=new_retrievers,
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
        sample: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        validate: list[str] | None = None,
        num_retries_on_validate_failure: int | None = None,
        drop_keys: list[str] | None = None,
        timeout: int | None = None,
        enable_observability: bool | None = None,
        max_batch_size: int | None = None,
        clustering_method: str | None = None,
        batch_prompt: str | None = None,
        litellm_completion_kwargs: dict[str, Any] | None = None,
        pdf_url_key: str | None = None,
        flush_partial_results: bool | None = None,
        limit: int | None = None,
        calibrate: bool | None = None,
        num_calibration_docs: int | None = None,
        retriever: Retriever | str | None = None,
        **kwargs: Any,
    ) -> Frame:
        return self._append_op("map", name, {
            "prompt": prompt, "output": output, "model": model,
            "optimize": optimize, "recursively_optimize": recursively_optimize,
            "sample": sample, "tools": tools,
            "validate": validate,
            "num_retries_on_validate_failure": num_retries_on_validate_failure,
            "drop_keys": drop_keys, "timeout": timeout,
            "enable_observability": enable_observability,
            "max_batch_size": max_batch_size, "clustering_method": clustering_method,
            "batch_prompt": batch_prompt,
            "litellm_completion_kwargs": litellm_completion_kwargs,
            "pdf_url_key": pdf_url_key,
            "flush_partial_results": flush_partial_results,
            "limit": limit, "calibrate": calibrate,
            "num_calibration_docs": num_calibration_docs,
            "retriever": retriever,
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
        validate: list[str] | None = None,
        drop_keys: list[str] | None = None,
        timeout: int | None = None,
        max_batch_size: int | None = None,
        litellm_completion_kwargs: dict[str, Any] | None = None,
        limit: int | None = None,
        retriever: Retriever | str | None = None,
        **kwargs: Any,
    ) -> Frame:
        return self._append_op("filter", name, {
            "prompt": prompt, "output": output, "model": model,
            "optimize": optimize, "tools": tools,
            "validate": validate, "drop_keys": drop_keys,
            "timeout": timeout, "max_batch_size": max_batch_size,
            "litellm_completion_kwargs": litellm_completion_kwargs,
            "limit": limit, "retriever": retriever, **kwargs,
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
        retriever: Retriever | str | None = None,
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
            "limit": limit, "retriever": retriever, **kwargs,
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
        retriever: Retriever | str | None = None,
        **kwargs: Any,
    ) -> Frame:
        return self._append_op("extract", name, {
            "prompt": prompt, "document_keys": document_keys, "model": model,
            "format_extraction": format_extraction,
            "extraction_key_suffix": extraction_key_suffix,
            "extraction_method": extraction_method, "timeout": timeout,
            "litellm_completion_kwargs": litellm_completion_kwargs,
            "limit": limit, "retriever": retriever, **kwargs,
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
        reduce_key: str | list[str] | None = None,
        pass_through: bool | None = None,
        concurrent_thread_count: int | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> Frame:
        return self._append_op("code_reduce", name, {
            "code": code, "reduce_key": reduce_key,
            "pass_through": pass_through,
            "concurrent_thread_count": concurrent_thread_count,
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

    # ── inspection ─────────────────────────────────────────────────

    def schema(self) -> dict[str, str]:
        """Return the output schema of this pipeline.

        Walks the operation chain and returns the schema that the last
        operation will produce.  Does not execute anything.
        """
        result: dict[str, str] = {}
        for op in self._operations:
            output = op.get("output", {})
            if isinstance(output, dict) and "schema" in output:
                result.update(output["schema"])
            if op.get("drop_keys"):
                for k in op["drop_keys"]:
                    result.pop(k, None)
        return result

    # ── terminal actions ───────────────────────────────────────────

    def _build_config(self, output_path: str = "") -> dict[str, Any]:
        output_cfg: dict[str, Any] = {"type": "file", "path": output_path}
        if _config.intermediate_dir:
            output_cfg["intermediate_dir"] = _config.intermediate_dir

        all_datasets = {**self._datasets, **self._extra_datasets}

        config: dict[str, Any] = {
            "datasets": all_datasets,
            "operations": self._operations,
            "pipeline": {
                "steps": self._steps,
                "output": output_cfg,
            },
        }
        if self._retrievers:
            config["retrievers"] = self._retrievers
        if _config.default_model:
            config["default_model"] = _config.default_model
        if _config.rate_limits:
            config["rate_limits"] = _config.rate_limits
        if _config.bypass_cache:
            config["bypass_cache"] = True
        if _config.fallback_models:
            config["fallback_models"] = _config.fallback_models
        if _config.fallback_embedding_models:
            config["fallback_embedding_models"] = _config.fallback_embedding_models
        return config

    def _build_runner(self, output_path: str = "", max_threads: int | None = None):
        from docetl.runner import DSLRunner
        threads = max_threads or _config.max_threads
        return DSLRunner(self._build_config(output_path), max_threads=threads)

    def show(self, max: int = 5, max_threads: int | None = None) -> "pd.DataFrame":
        """Run the pipeline on the first *max* input documents and print results.

        If no operations have been added yet, just shows the raw input data.
        Returns the resulting DataFrame.
        """
        import pandas as pd

        if not self._operations:
            data = self._load_input_data()[:max]
            df = pd.DataFrame(data)
            print(df.to_string())
            return df

        sampled = self._copy(datasets=self._sample_datasets(max))
        data, cost = sampled._execute(max_threads=max_threads)
        df = pd.DataFrame(data)
        df.attrs["_total_cost"] = cost
        df.attrs["_token_usage"] = sampled._token_usage
        print(df.to_string())
        return df

    def count(self, max_threads: int | None = None) -> int:
        """Return the number of output rows.

        On a bare dataset (no operations), returns the input count without
        executing anything.  With operations, executes the pipeline and
        counts the results.
        """
        if not self._operations:
            return len(self._load_input_data())
        data, _ = self._execute(max_threads=max_threads)
        return len(data)

    def _load_input_data(self) -> list[dict]:
        """Load the primary input dataset without executing operations."""
        import json

        ds = self._datasets.get(self._first_dataset, {})
        if ds.get("type") == "memory":
            return ds.get("path", [])
        elif ds.get("type") == "file":
            path = ds.get("path", "")
            if path.lower().endswith(".json"):
                with open(path) as f:
                    return json.load(f)
            elif path.lower().endswith(".csv"):
                import csv as csv_mod
                with open(path, newline="") as f:
                    return list(csv_mod.DictReader(f))
        return []

    def _sample_datasets(self, n: int) -> dict[str, dict[str, Any]]:
        """Return a copy of the datasets dict with each dataset truncated to *n* rows."""
        import json

        sampled = {}
        for name, ds in self._datasets.items():
            if ds.get("type") == "memory":
                data = ds["path"][:n] if isinstance(ds.get("path"), list) else ds["path"]
                sampled[name] = {**ds, "path": data}
            elif ds.get("type") == "file":
                path = ds.get("path", "")
                if path.lower().endswith(".json"):
                    with open(path) as f:
                        data = json.load(f)
                    sampled[name] = {"type": "memory", "path": data[:n]}
                elif path.lower().endswith(".csv"):
                    import csv as csv_mod
                    with open(path, newline="") as f:
                        reader = csv_mod.DictReader(f)
                        data = [row for _, row in zip(range(n), reader)]
                    sampled[name] = {"type": "memory", "path": data}
                else:
                    sampled[name] = ds
            else:
                sampled[name] = ds
        return sampled

    def collect(self, max_threads: int | None = None) -> "pd.DataFrame":
        """Execute the pipeline and return results as a DataFrame."""
        import pandas as pd
        data, cost = self._execute(max_threads=max_threads)
        df = pd.DataFrame(data)
        df.attrs["_total_cost"] = cost
        df.attrs["_token_usage"] = self._token_usage
        return df

    def to_list(self, max_threads: int | None = None) -> list[dict]:
        """Execute the pipeline and return results as a list of dicts."""
        data, _ = self._execute(max_threads=max_threads)
        return data

    def _execute(self, max_threads: int | None = None) -> tuple[list[dict], float]:
        import time
        from docetl.display import format_execution_summary

        runner = self._build_runner(max_threads=max_threads)
        runner.load()
        if runner.last_op_container is None:
            raise ValueError("Pipeline has no operations to execute.")
        runner.console.rule("[bold]Pipeline Execution[/bold]")
        start = time.time()
        output, _, _ = runner.last_op_container.next()
        elapsed = time.time() - start

        self._total_cost = runner.total_cost
        self._token_usage = dict(runner.total_token_usage)

        from rich.panel import Panel
        summary = format_execution_summary(
            runner.total_cost, elapsed, runner.total_token_usage,
            runner.intermediate_dir, "",
        )
        runner.console.log(Panel(summary, title="Execution Summary"))
        return output, runner.total_cost

    @property
    def total_cost(self) -> float:
        """Total cost of the last execution or optimization."""
        return self._total_cost

    @property
    def token_usage(self) -> dict[str, dict[str, int]]:
        """Token usage per model from the last execution."""
        return self._token_usage

    def write_json(self, path: str, max_threads: int | None = None) -> float:
        """Execute the pipeline and write results to a JSON file. Returns total cost."""
        runner = self._build_runner(output_path=path, max_threads=max_threads)
        cost = runner.load_run_save()
        self._total_cost = cost
        self._token_usage = dict(runner.total_token_usage)
        return cost

    def write_csv(self, path: str, max_threads: int | None = None) -> float:
        """Execute the pipeline and write results to a CSV file. Returns total cost."""
        runner = self._build_runner(output_path=path, max_threads=max_threads)
        cost = runner.load_run_save()
        self._total_cost = cost
        self._token_usage = dict(runner.total_token_usage)
        return cost

    def write_parquet(self, path: str, max_threads: int | None = None) -> float:
        """Execute the pipeline and write results to a Parquet file. Returns total cost."""
        runner = self._build_runner(output_path=path, max_threads=max_threads)
        cost = runner.load_run_save()
        self._total_cost = cost
        self._token_usage = dict(runner.total_token_usage)
        return cost

    # ── optimization ───────────────────────────────────────────────

    def optimize(
        self,
        *,
        eval_fn: Any = None,
        metric_key: str | None = None,
        models: list[str] | None = None,
        agent_model: str | None = None,
        max_iterations: int = 20,
        save_dir: str | None = None,
        exploration_weight: float = 1.414,
        dataset_path: str | None = None,
        max_threads: int | None = None,
        max_concurrent_agents: int = 3,
    ) -> Frame:
        """Optimize this pipeline and return an optimized Frame.

        Uses MOAR (Multi-Objective Agentic Rewrites) to search over
        model choices and prompt strategies. The best result from the
        Pareto frontier is returned as a new Frame you can ``.collect()``
        or ``.write_json()`` as usual.

        Access the full search results (Pareto frontier, costs, all plans)
        via ``frame.search_results`` after optimization.
        """
        from docetl.moar.optimizer import MOAROptimizer

        if eval_fn is None:
            raise ValueError("eval_fn is required for optimization.")
        if metric_key is None:
            raise ValueError("metric_key is required for optimization.")

        result = MOAROptimizer(
            pipeline=self._build_config(),
            eval_fn=eval_fn, metric_key=metric_key,
            models=models,
            agent_model=agent_model or _config.agent_model,
            max_iterations=max_iterations, save_dir=save_dir,
            exploration_weight=exploration_weight,
            dataset_path=dataset_path,
            max_threads=max_threads or _config.max_threads,
            max_concurrent_agents=max_concurrent_agents,
        ).optimize()

        self._total_cost = result.total_search_cost

        best = result.best()
        if best is None:
            raise RuntimeError("Optimization found no valid pipelines.")

        optimized = Frame.from_yaml(best.yaml_path)
        optimized._search_results = result
        optimized._total_cost = result.total_search_cost
        return optimized

    @property
    def search_results(self):
        """Access MOAR search results after calling ``.optimize()``.

        Returns ``None`` if this Frame was not created by optimization.
        """
        return getattr(self, "_search_results", None)

    # ── YAML loading ───────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> Frame:
        """Load a YAML pipeline config and return a Frame."""
        import yaml

        with open(path) as f:
            config = yaml.safe_load(f)

        _yaml_to_config = {
            "default_model": "default_model",
            "rate_limits": "rate_limits",
            "bypass_cache": "bypass_cache",
            "fallback_models": "fallback_models",
            "fallback_embedding_models": "fallback_embedding_models",
        }
        for yaml_key, config_attr in _yaml_to_config.items():
            if config.get(yaml_key) is not None:
                setattr(_config, config_attr, config[yaml_key])

        datasets: dict[str, dict[str, Any]] = {}
        first_ds: str | None = None
        for name, ds in config.get("datasets", {}).items():
            datasets[name] = ds
            if first_ds is None:
                first_ds = name

        ops_by_name = {op["name"]: op for op in config.get("operations", [])}

        operations: list[dict[str, Any]] = []
        steps: list[dict[str, Any]] = []
        last_step: str | None = None

        for step_cfg in config.get("pipeline", {}).get("steps", []):
            step: dict[str, Any] = {"name": step_cfg["name"], "operations": []}
            if "input" in step_cfg and step_cfg["input"] is not None:
                step["input"] = step_cfg["input"]

            for op_ref in step_cfg.get("operations", []):
                if isinstance(op_ref, str):
                    if op_ref in ops_by_name:
                        operations.append(ops_by_name[op_ref])
                    step["operations"].append(op_ref)
                elif isinstance(op_ref, dict):
                    op_name = list(op_ref.keys())[0]
                    if op_name in ops_by_name:
                        operations.append(ops_by_name[op_name])
                    step["operations"].append(op_ref)

            steps.append(step)
            last_step = step["name"]

        return cls(
            datasets, operations, steps,
            _last_step=last_step, _first_dataset=first_ds,
        )

    def to_yaml(self, path: str | None = None) -> str:
        """Export this pipeline as a YAML config string.

        If *path* is given, also writes the YAML to that file.
        """
        import yaml
        config = self._build_config()
        config.pop("pipeline", None)

        pipeline_cfg: dict[str, Any] = {
            "steps": self._steps,
            "output": {"type": "file", "path": "output.json"},
        }
        config["pipeline"] = pipeline_cfg

        out = yaml.dump(config, default_flow_style=False, sort_keys=False)
        if path:
            with open(path, "w") as f:
                f.write(out)
        return out

    # ── code generation ────────────────────────────────────────────

    def to_python(self) -> str:
        """Generate Python source that recreates this pipeline using the Frame API."""
        ops_by_name = {op["name"]: op for op in self._operations}
        lines: list[str] = ["import docetl", ""]

        for attr in ("default_model", "agent_model", "max_threads", "bypass_cache",
                      "rate_limits", "fallback_models", "fallback_embedding_models",
                      "intermediate_dir"):
            val = getattr(_config, attr, None)
            if val is not None and val is not False:
                lines.append(f"docetl.{attr} = {repr(val)}")
        if lines[-1] != "":
            lines.append("")

        # Determine the first dataset reader call
        ds_items = list(self._datasets.items())
        if not ds_items:
            lines.append("frame = docetl.from_list([])")
        elif len(ds_items) == 1:
            ds_name, ds_cfg = ds_items[0]
            lines.append(f"frame = (")
            lines.append(f"    {_format_reader(ds_name, ds_cfg)}")
        else:
            ds_name, ds_cfg = ds_items[0]
            lines.append(f"frame = (")
            lines.append(f"    {_format_reader(ds_name, ds_cfg)}")

        # Operations from steps
        for step in self._steps:
            step_ops = step.get("operations", [])
            for op_ref in step_ops:
                if isinstance(op_ref, str):
                    op = ops_by_name.get(op_ref)
                    if op is None:
                        continue
                    lines.append(_format_op_call(op))
                elif isinstance(op_ref, dict):
                    op_name = list(op_ref.keys())[0]
                    join_cfg = op_ref[op_name]
                    op = ops_by_name.get(op_name)
                    if op is None:
                        continue
                    # For equijoin, we need the right dataset
                    right_ds = join_cfg.get("right", "")
                    if right_ds and right_ds in self._datasets:
                        right_cfg = self._datasets[right_ds]
                        lines.append(_format_equijoin_call(op, right_ds, right_cfg))
                    else:
                        lines.append(_format_equijoin_call(op, right_ds, None))

        lines.append("    .collect()")
        lines.append(")")
        lines.append("")
        return "\n".join(lines)


# ── YAML to Python (standalone) ────────────────────────────────────

def yaml_to_python(yaml_path: str) -> str:
    """Convert a YAML pipeline config file to equivalent Python Frame code."""
    return Frame.from_yaml(yaml_path).to_python()


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


# ── codegen formatting helpers ─────────────────────────────────────

_SKIP_KEYS = {"name", "type"}


def _fmt(value: Any) -> str:
    if isinstance(value, str) and "\n" in value:
        return f'"""{value}"""'
    return repr(value)


def _format_reader(ds_name: str, ds_cfg: dict[str, Any]) -> str:
    is_memory = ds_cfg.get("type") == "memory"
    path = ds_cfg.get("path", "")

    if is_memory:
        return f"docetl.from_list({_fmt(path)}, name={repr(ds_name)})"

    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".csv":
        reader = "docetl.read_csv"
    elif ext == ".parquet":
        reader = "docetl.read_parquet"
    else:
        reader = "docetl.read_json"

    parts = [repr(path)]
    parsing = ds_cfg.get("parsing")
    if parsing:
        parts.append(f"parsing={_fmt(parsing)}")

    return f"{reader}({', '.join(parts)})"


def _format_op_call(op: dict[str, Any]) -> str:
    op_type = op.get("type", "map")
    parts: list[str] = [repr(op["name"])]
    for k, v in op.items():
        if k in _SKIP_KEYS or v is None:
            continue
        parts.append(f"{k}={_fmt(v)}")

    joined = ", ".join(parts)
    prefix = f"    .{op_type}("

    if len(prefix) + len(joined) + 1 <= 100:
        return f"{prefix}{joined})"

    indent = " " * len(prefix)
    formatted = f",\n{indent}".join(parts)
    return f"{prefix}{formatted})"


def _format_equijoin_call(
    op: dict[str, Any], right_ds: str, right_cfg: dict[str, Any] | None,
) -> str:
    parts: list[str] = []
    if right_cfg:
        parts.append(_format_reader(right_ds, right_cfg))
    parts.append(repr(op["name"]))
    for k, v in op.items():
        if k in _SKIP_KEYS or v is None:
            continue
        parts.append(f"{k}={_fmt(v)}")

    joined = ", ".join(parts)
    prefix = "    .equijoin("

    if len(prefix) + len(joined) + 1 <= 100:
        return f"{prefix}{joined})"

    indent = " " * len(prefix)
    formatted = f",\n{indent}".join(parts)
    return f"{prefix}{formatted})"
