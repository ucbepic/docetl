"""Pandas DataFrame accessor for semantic (LLM-powered) operations.

Usage::

    import pandas as pd
    import docetl

    docetl.default_model = "gpt-4o-mini"

    df = pd.DataFrame({"text": ["Apple is a tech company", "Microsoft makes Windows"]})
    result = df.semantic.map(
        prompt="Extract company name from: {{input.text}}",
        output={"schema": {"company": "str"}},
    )
"""

from typing import Any, NamedTuple

import pandas as pd

from docetl import _config
from docetl.frame import from_list


class OpHistory(NamedTuple):
    op_type: str
    config: dict[str, Any]
    output_columns: list[str]


@pd.api.extensions.register_dataframe_accessor("semantic")
class SemanticAccessor:
    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._history: list[OpHistory] = df.attrs.get("_semantic_history", []).copy()
        self._costs: float = df.attrs.get("_semantic_costs", 0.0)

    def _run_single_op(
        self, op_type: str, op_name: str, data: list[dict], **op_kwargs
    ) -> tuple[list[dict], float]:
        """Execute a single operation and return (results, cost)."""
        frame = (
            from_list(data).map(**op_kwargs)
            if op_type == "map"
            else getattr(from_list(data), op_type)(**op_kwargs)
        )
        runner = frame._build_runner()
        runner.load()
        output, _, _ = runner.last_op_container.next()
        return output, runner.total_cost

    def _bare_runner(self):
        """A pipeline-less DSLRunner carrying all module-level settings
        (model, rate limits, cache, fallbacks, threads)."""
        from docetl.runner import DSLRunner

        runner_config: dict[str, Any] = {
            "default_model": "gpt-4o-mini",
            **_config.runner_settings(),
            "operations": [],
            "datasets": {},
            "pipeline": {"steps": []},
        }
        return DSLRunner(
            runner_config, max_threads=_config.max_threads, from_df_accessors=True
        )

    def _run_op_direct(
        self, op_type: str, data: list[dict], config: dict[str, Any], runner=None
    ) -> tuple[list[dict], float]:
        """Execute an operation using the operation class directly (for cases
        that need runner access, like JoinOptimizer)."""
        if runner is None:
            runner = self._bare_runner()

        from docetl.operations import get_operation

        op_class = get_operation(config["type"])
        op = op_class(
            runner=runner,
            config=config,
            default_model=runner.default_model,
            max_threads=runner.max_threads,
            console=runner.console,
            status=runner.status,
        )

        if config["type"] == "equijoin":
            results, cost = op.execute(data["left"], data["right"])
        elif config["type"] == "filter":
            results, cost = op.execute(data, False)
        else:
            results, cost = op.execute(data)

        return results, cost

    def _record(
        self, data: list[dict], op_type: str, config: dict[str, Any], cost: float
    ) -> pd.DataFrame:
        result_df = pd.DataFrame(data)
        new_cols = list(set(result_df.columns) - set(self._df.columns))
        self._history.append(OpHistory(op_type, config, new_cols))
        self._costs += cost
        result_df.attrs["_semantic_history"] = self._history
        result_df.attrs["_semantic_costs"] = self._costs
        return result_df

    # ── operations ─────────────────────────────────────────────────

    def map(
        self,
        prompt: str,
        output: dict[str, Any] = None,
        *,
        output_schema: dict[str, Any] = None,
        **kwargs,
    ) -> pd.DataFrame:
        if output_schema is not None and output is None:
            output = {"schema": output_schema}
        if output is None:
            raise ValueError("'output' must be provided with a 'schema' key")

        config = {
            "type": "map",
            "name": f"semantic_map_{len(self._history)}",
            "prompt": prompt,
            "output": output,
            **kwargs,
        }
        results, cost = self._run_op_direct("map", self._df.to_dict("records"), config)
        return self._record(results, "map", config, cost)

    def filter(
        self,
        prompt: str,
        *,
        output: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        if output_schema is not None and output is None:
            output = {"schema": output_schema}
        if output is None:
            output = {"schema": {"keep": "bool"}}

        config = {
            "type": "filter",
            "name": f"semantic_filter_{len(self._history)}",
            "prompt": prompt,
            "output": output,
            **kwargs,
        }
        results, cost = self._run_op_direct(
            "filter", self._df.to_dict("records"), config
        )
        return self._record(results, "filter", config, cost)

    def reduce(
        self,
        prompt: str,
        output: dict[str, Any] = None,
        *,
        output_schema: dict[str, Any] = None,
        reduce_keys: str | list[str] = ["_all"],
        **kwargs,
    ) -> pd.DataFrame:
        if output_schema is not None and output is None:
            output = {"schema": output_schema}
        if output is None:
            raise ValueError("'output' must be provided with a 'schema' key")
        if isinstance(reduce_keys, str):
            reduce_keys = [reduce_keys]

        config = {
            "type": "reduce",
            "name": f"semantic_reduce_{len(self._history)}",
            "reduce_key": reduce_keys,
            "prompt": prompt,
            "output": output,
            **kwargs,
        }
        results, cost = self._run_op_direct(
            "reduce", self._df.to_dict("records"), config
        )
        return self._record(results, "reduce", config, cost)

    def merge(
        self,
        right: pd.DataFrame,
        comparison_prompt: str,
        *,
        fuzzy: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        left_data = self._df.to_dict("records")
        right_data = right.to_dict("records")

        config = {
            "type": "equijoin",
            "name": f"semantic_merge_{len(self._history)}",
            "comparison_prompt": comparison_prompt,
            **kwargs,
        }

        runner = self._bare_runner()
        if (
            fuzzy
            and "blocking_threshold" not in kwargs
            and "blocking_conditions" not in kwargs
        ):
            from docetl.optimizer import Optimizer
            from docetl.optimizers.join_optimizer import JoinOptimizer

            runner.optimizer = Optimizer(runner)
            optimizer = JoinOptimizer(
                runner,
                config,
                target_recall=kwargs.get("target_recall", 0.95),
                estimated_selectivity=kwargs.get("estimated_selectivity"),
            )
            config, opt_cost, _ = optimizer.optimize_equijoin(
                left_data, right_data, skip_map_gen=True, skip_containment_gen=True
            )
        else:
            opt_cost = 0.0

        results, cost = self._run_op_direct(
            "equijoin", {"left": left_data, "right": right_data}, config, runner=runner
        )
        return self._record(results, "equijoin", config, cost + opt_cost)

    def agg(
        self,
        *,
        reduce_prompt: str,
        output: dict[str, Any] = None,
        output_schema: dict[str, Any] = None,
        fuzzy: bool = False,
        comparison_prompt: str | None = None,
        resolution_prompt: str | None = None,
        resolution_output: dict[str, Any] | None = None,
        reduce_keys: str | list[str] = ["_all"],
        resolve_kwargs: dict[str, Any] = {},
        reduce_kwargs: dict[str, Any] = {},
    ) -> pd.DataFrame:
        if output_schema is not None and output is None:
            output = {"schema": output_schema}
        if output is None:
            raise ValueError("'output' must be provided with a 'schema' key")
        if isinstance(reduce_keys, str):
            reduce_keys = [reduce_keys]

        input_data = self._df.to_dict("records")

        # Resolve phase
        if reduce_keys != ["_all"] and fuzzy and len(input_data) > 5:
            if comparison_prompt is None:
                fields = ", ".join(f"{k}: {{{{ input1.{k} }}}}" for k in reduce_keys)
                comparison_prompt = f"Do these two records represent the same concept?\nRecord 1: {fields}\nRecord 2: {fields.replace('input1', 'input2')}"

            resolve_config = {
                "type": "resolve",
                "name": f"semantic_resolve_{len(self._history)}",
                "comparison_prompt": comparison_prompt,
                **resolve_kwargs,
            }
            if resolution_prompt:
                resolve_config["resolution_prompt"] = resolution_prompt
                resolve_config["output"] = resolution_output or {"keys": reduce_keys}
            else:
                resolve_config["output"] = {"keys": reduce_keys}

            runner = self._bare_runner()
            if (
                "blocking_threshold" not in resolve_kwargs
                and "blocking_conditions" not in resolve_kwargs
            ):
                from docetl.optimizer import Optimizer
                from docetl.optimizers.join_optimizer import JoinOptimizer

                runner.optimizer = Optimizer(runner)
                optimizer = JoinOptimizer(
                    runner,
                    resolve_config,
                    target_recall=resolve_kwargs.get("target_recall", 0.95),
                    estimated_selectivity=resolve_kwargs.get("estimated_selectivity"),
                )
                resolve_config, opt_cost = optimizer.optimize_resolve(input_data)
            else:
                opt_cost = 0.0

            input_data, resolve_cost = self._run_op_direct(
                "resolve", input_data, resolve_config, runner=runner
            )
            self._record(input_data, "resolve", resolve_config, resolve_cost + opt_cost)

        # Reduce phase
        reduce_config = {
            "type": "reduce",
            "name": f"semantic_reduce_{len(self._history)}",
            "reduce_key": reduce_keys,
            "prompt": reduce_prompt,
            "output": output,
            **reduce_kwargs,
        }
        results, cost = self._run_op_direct("reduce", input_data, reduce_config)
        return self._record(results, "reduce", reduce_config, cost)

    def split(
        self, split_key: str, method: str, method_kwargs: dict[str, Any], **kwargs
    ) -> pd.DataFrame:
        config = {
            "type": "split",
            "name": f"semantic_split_{len(self._history)}",
            "split_key": split_key,
            "method": method,
            "method_kwargs": method_kwargs,
            **kwargs,
        }
        results, cost = self._run_op_direct(
            "split", self._df.to_dict("records"), config
        )
        return self._record(results, "split", config, cost)

    def gather(
        self,
        content_key: str,
        doc_id_key: str,
        order_key: str,
        peripheral_chunks: dict[str, Any] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        config = {
            "type": "gather",
            "name": f"semantic_gather_{len(self._history)}",
            "content_key": content_key,
            "doc_id_key": doc_id_key,
            "order_key": order_key,
            **kwargs,
        }
        if peripheral_chunks is not None:
            config["peripheral_chunks"] = peripheral_chunks
        results, cost = self._run_op_direct(
            "gather", self._df.to_dict("records"), config
        )
        return self._record(results, "gather", config, cost)

    def unnest(
        self,
        unnest_key: str,
        keep_empty: bool = False,
        expand_fields: list[str] | None = None,
        recursive: bool = False,
        depth: int | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        config = {
            "type": "unnest",
            "name": f"semantic_unnest_{len(self._history)}",
            "unnest_key": unnest_key,
            "keep_empty": keep_empty,
            "recursive": recursive,
            **kwargs,
        }
        if expand_fields is not None:
            config["expand_fields"] = expand_fields
        if depth is not None:
            config["depth"] = depth
        results, cost = self._run_op_direct(
            "unnest", self._df.to_dict("records"), config
        )
        return self._record(results, "unnest", config, cost)

    @property
    def total_cost(self) -> float:
        return self._costs

    @property
    def history(self) -> list[OpHistory]:
        return self._history.copy()
