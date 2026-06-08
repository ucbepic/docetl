"""Fluent builder for docetl pipelines and YAML-to-Python conversion."""

from __future__ import annotations

import os
from typing import Any

import yaml


class PipelineBuilder:
    """Fluent builder that eliminates the boilerplate of wiring operations,
    steps, and datasets by hand.

    Quick start::

        from docetl.builder import PipelineBuilder

        pipe = (
            PipelineBuilder("my_pipeline", default_model="gpt-4o-mini")
            .dataset("docs", type="file", path="input.json")
            .output(type="file", path="output.json")
            .map("summarize",
                 prompt="Summarize: {{ input.text }}",
                 output={"schema": {"summary": "string"}})
            .reduce("aggregate",
                    reduce_key="category",
                    prompt="Group these: {{ inputs }}",
                    output={"schema": {"grouped": "string"}})
        )
        pipe.run()
    """

    def __init__(
        self,
        name: str = "pipeline",
        *,
        default_model: str | None = None,
        rate_limits: dict[str, int] | None = None,
        optimizer_config: dict[str, Any] | None = None,
        parsing_tools: list[dict] | None = None,
        **kwargs: Any,
    ):
        self._name = name
        self._default_model = default_model
        self._rate_limits = rate_limits
        self._optimizer_config = optimizer_config
        self._parsing_tools = parsing_tools or []
        self._extra_config: dict[str, Any] = kwargs

        self._datasets: dict[str, dict[str, Any]] = {}
        self._output_config: dict[str, Any] | None = None
        self._operations: list[dict[str, Any]] = []
        self._steps: list[dict[str, Any]] = []

        self._op_counter: dict[str, int] = {}
        self._last_step_name: str | None = None
        self._first_dataset: str | None = None
        self._explicit_step: dict[str, Any] | None = None

    # ── datasets & output ───────────────────────────────────────────

    def dataset(
        self,
        name: str,
        *,
        type: str = "file",
        path: str | None = None,
        data: list[dict] | None = None,
        source: str = "local",
        parsing: list[dict[str, str]] | None = None,
    ) -> PipelineBuilder:
        ds: dict[str, Any] = {"type": type, "source": source}
        if type == "file":
            ds["path"] = path or ""
        elif type == "memory" and data is not None:
            ds["path"] = data
        elif path is not None:
            ds["path"] = path
        if parsing:
            ds["parsing"] = parsing
        self._datasets[name] = ds
        if self._first_dataset is None:
            self._first_dataset = name
        return self

    def output(
        self,
        *,
        type: str = "file",
        path: str = "",
        intermediate_dir: str | None = None,
    ) -> PipelineBuilder:
        self._output_config = {"type": type, "path": path}
        if intermediate_dir is not None:
            self._output_config["intermediate_dir"] = intermediate_dir
        return self

    # ── step control ────────────────────────────────────────────────

    def step(self, name: str | None = None, *, input: str | None = None) -> PipelineBuilder:
        """Start a new explicit step.  Call ``.step()`` with no args to
        return to auto-step mode (one step per operation)."""
        if self._explicit_step is not None:
            if not self._explicit_step["operations"]:
                self._steps.pop()
            self._explicit_step = None

        if name is not None:
            step: dict[str, Any] = {"name": name, "operations": []}
            step_input = input or self._last_step_name or self._first_dataset
            if step_input:
                step["input"] = step_input
            self._steps.append(step)
            self._last_step_name = name
            self._explicit_step = step

        return self

    # ── private helpers ─────────────────────────────────────────────

    def _auto_name(self, op_type: str) -> str:
        self._op_counter[op_type] = self._op_counter.get(op_type, 0) + 1
        return f"{op_type}_{self._op_counter[op_type]}"

    def _add_op(self, op_type: str, name: str | None, config: dict[str, Any]) -> PipelineBuilder:
        if name is None:
            name = self._auto_name(op_type)
        config = {"name": name, "type": op_type, **{k: v for k, v in config.items() if v is not None}}
        self._operations.append(config)

        if self._explicit_step is not None:
            self._explicit_step["operations"].append(name)
        else:
            step_input = self._last_step_name or self._first_dataset
            step: dict[str, Any] = {"name": f"step_{name}", "operations": [name]}
            if step_input:
                step["input"] = step_input
            self._steps.append(step)
            self._last_step_name = step["name"]

        return self

    # ── LLM operations ──────────────────────────────────────────────

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
    ) -> PipelineBuilder:
        return self._add_op("map", name, {
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
    ) -> PipelineBuilder:
        return self._add_op("parallel_map", name, {
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
    ) -> PipelineBuilder:
        return self._add_op("filter", name, {
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
    ) -> PipelineBuilder:
        return self._add_op("reduce", name, {
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
    ) -> PipelineBuilder:
        return self._add_op("resolve", name, {
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
        name: str | None = None,
        *,
        left: str,
        right: str,
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
    ) -> PipelineBuilder:
        if name is None:
            name = self._auto_name("equijoin")
        config = {
            k: v for k, v in {
                "name": name, "type": "equijoin",
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
        self._operations.append(config)

        step_name = f"step_{name}"
        self._steps.append({
            "name": step_name,
            "operations": [{name: {"left": left, "right": right}}],
        })
        self._last_step_name = step_name
        return self

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
    ) -> PipelineBuilder:
        return self._add_op("extract", name, {
            "prompt": prompt, "document_keys": document_keys, "model": model,
            "format_extraction": format_extraction,
            "extraction_key_suffix": extraction_key_suffix,
            "extraction_method": extraction_method, "timeout": timeout,
            "litellm_completion_kwargs": litellm_completion_kwargs,
            "limit": limit, **kwargs,
        })

    # ── structural operations ───────────────────────────────────────

    def split(
        self,
        name: str | None = None,
        *,
        split_key: str | None = None,
        method: str | None = None,
        method_kwargs: dict[str, Any] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> PipelineBuilder:
        return self._add_op("split", name, {
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
    ) -> PipelineBuilder:
        return self._add_op("gather", name, {
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
    ) -> PipelineBuilder:
        return self._add_op("unnest", name, {
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
    ) -> PipelineBuilder:
        return self._add_op("cluster", name, {
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
    ) -> PipelineBuilder:
        return self._add_op("sample", name, {
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
    ) -> PipelineBuilder:
        return self._add_op("code_map", name, {
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
    ) -> PipelineBuilder:
        return self._add_op("code_reduce", name, {
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
    ) -> PipelineBuilder:
        return self._add_op("code_filter", name, {
            "code": code, "concurrent_thread_count": concurrent_thread_count,
            "limit": limit, **kwargs,
        })

    # ── build / run / optimize ──────────────────────────────────────

    def build(self):
        """Build a ``Pipeline`` object from this builder."""
        from docetl.api import Pipeline
        from docetl.base_schemas import PipelineOutput, PipelineStep

        datasets = {}
        for ds_name, ds_cfg in self._datasets.items():
            from docetl.schemas import Dataset
            datasets[ds_name] = Dataset(**ds_cfg)

        ops_list: list = []
        for op in self._operations:
            op_type = op.get("type", "map")
            schema_cls = Pipeline._OP_TYPE_REGISTRY.get(op_type)
            filtered = {k: v for k, v in op.items() if v is not None}
            if schema_cls is not None:
                try:
                    ops_list.append(schema_cls(**filtered))
                except Exception:
                    from docetl.schemas import MapOp
                    ops_list.append(MapOp.model_construct(**filtered))
            else:
                from docetl.schemas import MapOp
                ops_list.append(MapOp.model_construct(**filtered))

        steps = [
            PipelineStep(**{k: v for k, v in s.items() if v is not None})
            for s in self._steps
        ]

        out_cfg = self._output_config or {"type": "file", "path": ""}
        output = PipelineOutput(**out_cfg)

        return Pipeline(
            name=self._name,
            datasets=datasets,
            operations=ops_list,
            steps=steps,
            output=output,
            parsing_tools=self._parsing_tools or [],
            default_model=self._default_model,
            rate_limits=self._rate_limits,
            optimizer_config=self._optimizer_config or {},
            **self._extra_config,
        )

    def run(self, max_threads: int | None = None) -> float:
        return self.build().run(max_threads=max_threads)

    def optimize(self, **kwargs):
        return self.build().optimize(**kwargs)

    # ── YAML loading ────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> PipelineBuilder:
        """Load a YAML pipeline config and return a builder."""
        with open(path) as f:
            config = yaml.safe_load(f)

        known_keys = {
            "datasets", "operations", "pipeline", "default_model",
            "rate_limits", "optimizer_config", "parsing_tools",
        }
        extra = {k: v for k, v in config.items() if k not in known_keys}

        builder = cls(
            name="pipeline",
            default_model=config.get("default_model"),
            rate_limits=config.get("rate_limits"),
            optimizer_config=config.get("optimizer_config"),
            parsing_tools=config.get("parsing_tools"),
            **extra,
        )

        for name, ds in config.get("datasets", {}).items():
            builder._datasets[name] = ds
            if builder._first_dataset is None:
                builder._first_dataset = name

        out = config.get("pipeline", {}).get("output", {})
        if out:
            builder._output_config = out

        ops_by_name = {op["name"]: op for op in config.get("operations", [])}

        for step_cfg in config.get("pipeline", {}).get("steps", []):
            step: dict[str, Any] = {
                "name": step_cfg["name"],
                "operations": [],
            }
            if "input" in step_cfg and step_cfg["input"] is not None:
                step["input"] = step_cfg["input"]

            for op_ref in step_cfg.get("operations", []):
                if isinstance(op_ref, str):
                    if op_ref in ops_by_name:
                        builder._operations.append(ops_by_name[op_ref])
                    step["operations"].append(op_ref)
                elif isinstance(op_ref, dict):
                    op_name = list(op_ref.keys())[0]
                    if op_name in ops_by_name:
                        builder._operations.append(ops_by_name[op_name])
                    step["operations"].append(op_ref)

            builder._steps.append(step)
            builder._last_step_name = step["name"]

        return builder

    # ── code generation ─────────────────────────────────────────────

    def to_python(self) -> str:
        """Generate Python source code that recreates this pipeline."""
        ops_by_name = {op["name"]: op for op in self._operations}
        lines: list[str] = ["from docetl.builder import PipelineBuilder", ""]

        # Constructor
        ctor = [repr(self._name)]
        if self._default_model:
            ctor.append(f"default_model={repr(self._default_model)}")
        if self._rate_limits:
            ctor.append(f"rate_limits={_fmt(self._rate_limits)}")
        if self._optimizer_config:
            ctor.append(f"optimizer_config={_fmt(self._optimizer_config)}")
        if self._parsing_tools:
            ctor.append(f"parsing_tools={_fmt(self._parsing_tools)}")
        for k, v in self._extra_config.items():
            ctor.append(f"{k}={_fmt(v)}")

        lines.append("pipe = (")
        lines.append(f"    PipelineBuilder({', '.join(ctor)})")

        # Datasets
        for ds_name, ds_cfg in self._datasets.items():
            ds_parts = [repr(ds_name)]
            for k, v in ds_cfg.items():
                ds_parts.append(f"{k}={_fmt(v)}")
            lines.append(f"    .dataset({', '.join(ds_parts)})")

        # Output
        if self._output_config:
            out_parts = [f"{k}={_fmt(v)}" for k, v in self._output_config.items() if v is not None]
            lines.append(f"    .output({', '.join(out_parts)})")

        # Steps → operation calls
        for step in self._steps:
            step_ops = step.get("operations", [])
            is_multi = len(step_ops) > 1 and not any(isinstance(r, dict) for r in step_ops)

            if is_multi:
                sp = [repr(step["name"])]
                if "input" in step:
                    sp.append(f"input={repr(step['input'])}")
                lines.append(f"    .step({', '.join(sp)})")

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
                    lines.append(_format_equijoin_call(op, join_cfg))

        lines.append(")")
        lines.append("")
        lines.append("pipe.run()")
        lines.append("")
        return "\n".join(lines)


# ── YAML to Python (standalone) ────────────────────────────────────

def yaml_to_python(yaml_path: str) -> str:
    """Convert a YAML pipeline config file to equivalent Python builder code."""
    return PipelineBuilder.from_yaml(yaml_path).to_python()


# ── formatting helpers ──────────────────────────────────────────────

_SKIP_KEYS = {"name", "type"}


def _fmt(value: Any) -> str:
    if isinstance(value, str) and "\n" in value:
        return f'"""{value}"""'
    return repr(value)


def _format_op_call(op: dict[str, Any]) -> str:
    op_type = op.get("type", "map")
    method = op_type

    parts: list[str] = [repr(op["name"])]
    for k, v in op.items():
        if k in _SKIP_KEYS or v is None:
            continue
        parts.append(f"{k}={_fmt(v)}")

    joined = ", ".join(parts)
    prefix = f"    .{method}("

    if len(prefix) + len(joined) + 1 <= 100:
        return f"{prefix}{joined})"

    indent = " " * (len(prefix))
    formatted = f",\n{indent}".join(parts)
    return f"{prefix}{formatted})"


def _format_equijoin_call(op: dict[str, Any], join_cfg: dict[str, str]) -> str:
    parts: list[str] = [repr(op["name"])]
    parts.append(f"left={repr(join_cfg['left'])}")
    parts.append(f"right={repr(join_cfg['right'])}")
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
