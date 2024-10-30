from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from docetl.operations.base import BaseOperation
from docetl.operations.utils import RichLoopBar
from pydantic import Field

class CodeMapOperation(BaseOperation):
    class schema(BaseOperation.schema):
        type: str = "code_map"
        code: str
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
        with ThreadPoolExecutor() as executor:
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
                        k: v for k, v in result.items() 
                        if k not in self.config["drop_keys"]
                    }
                results.append(result)
        return results, 0.0

class CodeReduceOperation(BaseOperation):
    class schema(BaseOperation.schema):
        type: str = "code_reduce"
        code: str

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

    def execute(self, input_data: List[Dict]) -> Tuple[Dict, float]:
        namespace = {}
        exec(self.config["code"], namespace)
        reduce_fn = namespace["transform"]
        
        result = reduce_fn(input_data)
        return result, 0.0

class CodeFilterOperation(BaseOperation):
    class schema(BaseOperation.schema):
        type: str = "code_filter"
        code: str

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
        with ThreadPoolExecutor() as executor:
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