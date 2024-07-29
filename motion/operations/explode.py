from typing import Dict, List, Any, Tuple
from motion.operations.base import BaseOperation
from rich.console import Console


class ExplodeOperation(BaseOperation):
    def syntax_check(self) -> None:
        required_keys = ["explode_key"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(
                    f"Missing required key '{key}' in ExplodeOperation configuration"
                )

    def execute(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        explode_key = self.config["explode_key"]
        results = []

        for item in input_data:
            if explode_key not in item:
                raise KeyError(f"Explode key '{explode_key}' not found in item")
            if not isinstance(item[explode_key], (list, tuple, set)):
                raise TypeError(f"Value of explode key '{explode_key}' is not iterable")

            for value in item[explode_key]:
                new_item = item.copy()
                new_item[explode_key] = value
                results.append(new_item)

        return results, 0
