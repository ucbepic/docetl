import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from docetl.operations.base import BaseOperation
from docetl.operations.utils import RichLoopBar


class CodeMapOperation(BaseOperation):
    class schema(BaseOperation.schema):
        type: str = "code_map"
        code: str
        concurrent_thread_count: int = os.cpu_count()
        drop_keys: Optional[List[str]] = None

    def syntax_check(self) -> None:
        config = self.schema(**self.config)
        try:
            namespace = {}
            exec(config.code, namespace)
            if "transform" not in namespace:
                raise ValueError("Code must define a 'transform' function")
            if not callable(namespace["transform"]):
                raise ValueError("'transform' must be a callable function")
        except Exception as e:
            raise ValueError(f"Invalid code configuration: {str(e)}")

    def execute(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        namespace = {}
        exec(self.config["code"], namespace)
        transform_fn = namespace["transform"]

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
        code: str
        concurrent_thread_count: int = os.cpu_count()

    def syntax_check(self) -> None:
        config = self.schema(**self.config)
        try:
            namespace = {}
            exec(config.code, namespace)
            if "transform" not in namespace:
                raise ValueError("Code must define a 'transform' function")
            if not callable(namespace["transform"]):
                raise ValueError("'transform' must be a callable function")
        except Exception as e:
            raise ValueError(f"Invalid code configuration: {str(e)}")

    def execute(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        namespace = {}
        exec(self.config["code"], namespace)
        reduce_fn = namespace["transform"]

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
        code: str
        concurrent_thread_count: int = os.cpu_count()

    def syntax_check(self) -> None:
        config = self.schema(**self.config)
        try:
            namespace = {}
            exec(config.code, namespace)
            if "transform" not in namespace:
                raise ValueError("Code must define a 'transform' function")
            if not callable(namespace["transform"]):
                raise ValueError("'transform' must be a callable function")
        except Exception as e:
            raise ValueError(f"Invalid code configuration: {str(e)}")

    def execute(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        namespace = {}
        exec(self.config["code"], namespace)
        filter_fn = namespace["transform"]

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
        return results, 0.0
