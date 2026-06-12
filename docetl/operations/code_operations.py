import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic import Field, field_validator

from docetl.operations.base import BaseOperation
from docetl.operations.utils import RichLoopBar


def resolve_transform(code: Any):
    """Return the transform callable from a ``code`` config value.

    ``code`` is either a callable (used directly; lambdas and closures work)
    or a string of Python source defining a ``transform`` function.
    """
    if callable(code):
        return code
    namespace = {}
    exec(code, namespace)
    transform = namespace.get("transform")
    if not callable(transform):
        raise ValueError("Code must define a 'transform' function")
    return transform


def _validate_code(v: Any) -> Any:
    if isinstance(v, str) or callable(v):
        return v
    raise TypeError("code must be a string or a callable")


class CodeMapOperation(BaseOperation):
    class schema(BaseOperation.schema):
        type: str = "code_map"
        code: Any
        concurrent_thread_count: int = os.cpu_count()
        drop_keys: list[str] | None = None
        limit: int | None = Field(None, gt=0)

        validate_code = field_validator("code")(_validate_code)

    def syntax_check(self) -> None:
        config = self.schema(**self.config)
        try:
            resolve_transform(config.code)
        except Exception as e:
            raise ValueError(f"Invalid code configuration: {str(e)}")

    def execute(self, input_data: list[dict]) -> tuple[list[dict], float]:
        limit_value = self.config.get("limit")
        if limit_value is not None:
            input_data = input_data[:limit_value]

        transform_fn = resolve_transform(self.config["code"])

        results = []
        with ThreadPoolExecutor(
            max_workers=self.config.get("concurrent_thread_count", os.cpu_count())
        ) as executor:
            futures = [executor.submit(transform_fn, doc) for doc in input_data]
            pbar = RichLoopBar(
                range(len(futures)),
                desc=f"Processing {self.config['name']} (code_map)",
                console=self.console,
            )
            for i in pbar:
                result = futures[i].result()
                if self.config.get("drop_keys"):
                    result = {
                        k: v
                        for k, v in result.items()
                        if k not in self.config["drop_keys"]
                    }
                doc = input_data[i]
                merged_result = {**doc, **result}
                results.append(merged_result)

        return results, 0.0


class CodeReduceOperation(BaseOperation):
    class schema(BaseOperation.schema):
        type: str = "code_reduce"
        code: Any
        concurrent_thread_count: int = os.cpu_count()
        limit: int | None = Field(None, gt=0)

        validate_code = field_validator("code")(_validate_code)

    def syntax_check(self) -> None:
        config = self.schema(**self.config)
        try:
            resolve_transform(config.code)
        except Exception as e:
            raise ValueError(f"Invalid code configuration: {str(e)}")

    def execute(self, input_data: list[dict]) -> tuple[list[dict], float]:
        reduce_fn = resolve_transform(self.config["code"])

        reduce_keys = self.config.get("reduce_key", "_all")
        if not isinstance(reduce_keys, list):
            reduce_keys = [reduce_keys]

        if reduce_keys == ["_all"] or reduce_keys == "_all":
            grouped_data = [("_all", input_data)]
        else:

            def get_group_key(item):
                return tuple(item[key] for key in reduce_keys)

            grouped_data = {}
            for item in input_data:
                key = get_group_key(item)
                if key not in grouped_data:
                    grouped_data[key] = []
                grouped_data[key].append(item)

            grouped_data = list(grouped_data.items())

        limit_value = self.config.get("limit")
        if limit_value is not None:
            # Sort by group size (smallest first) and take the limit
            grouped_data = sorted(grouped_data, key=lambda x: len(x[1]))
            grouped_data = grouped_data[:limit_value]

        results = []
        with ThreadPoolExecutor(
            max_workers=self.config.get("concurrent_thread_count", os.cpu_count())
        ) as executor:
            futures = [executor.submit(reduce_fn, group) for _, group in grouped_data]
            pbar = RichLoopBar(
                range(len(futures)),
                desc=f"Processing {self.config['name']} (code_reduce)",
                console=self.console,
            )
            for i, (key, group) in zip(pbar, grouped_data):
                result = futures[i].result()

                # Apply pass-through at the group level
                if self.config.get("pass_through", False) and group:
                    for k, v in group[0].items():
                        if k not in result:
                            result[k] = v

                # Also add the reduce key
                if reduce_keys != ["_all"]:
                    for k in reduce_keys:
                        if k not in result:
                            result[k] = group[0][k]

                result[f"_counts_prereduce_{self.config['name']}"] = len(group)

                results.append(result)

        return results, 0.0


class CodeFilterOperation(BaseOperation):
    class schema(BaseOperation.schema):
        type: str = "code_filter"
        code: Any
        concurrent_thread_count: int = os.cpu_count()
        limit: int | None = Field(None, gt=0)

        validate_code = field_validator("code")(_validate_code)

    def syntax_check(self) -> None:
        config = self.schema(**self.config)
        try:
            resolve_transform(config.code)
        except Exception as e:
            raise ValueError(f"Invalid code configuration: {str(e)}")

    def execute(self, input_data: list[dict]) -> tuple[list[dict], float]:
        filter_fn = resolve_transform(self.config["code"])

        limit_value = self.config.get("limit")
        results = []
        with ThreadPoolExecutor(
            max_workers=self.config.get("concurrent_thread_count", os.cpu_count())
        ) as executor:
            futures = [executor.submit(filter_fn, doc) for doc in input_data]
            pbar = RichLoopBar(
                range(len(futures)),
                desc=f"Processing {self.config['name']} (code_filter)",
                console=self.console,
            )
            for i in pbar:
                should_keep = futures[i].result()
                if should_keep:
                    results.append(input_data[i])
                    if limit_value is not None and len(results) >= limit_value:
                        break
        return results, 0.0
