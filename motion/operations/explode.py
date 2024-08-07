from typing import Dict, List, Any, Tuple
from motion.operations.base import BaseOperation
from rich.console import Console


class ExplodeOperation(BaseOperation):
    """
    A class that represents an operation to explode a list-like value in a dictionary into multiple dictionaries.

    This operation takes a list of dictionaries and a specified key, and creates a new dictionary for each element
    in the list-like value of that key, copying all other key-value pairs.

    Inherits from:
        BaseOperation

    Usage:
    ```python
    from motion.operations import ExplodeOperation

    config = {"explode_key": "tags"}
    input_data = [
        {"id": 1, "tags": ["a", "b", "c"]},
        {"id": 2, "tags": ["d", "e"]}
    ]

    explode_op = ExplodeOperation(config)
    result, _ = explode_op.execute(input_data)

    # Result will be:
    # [
    #     {"id": 1, "tags": "a"},
    #     {"id": 1, "tags": "b"},
    #     {"id": 1, "tags": "c"},
    #     {"id": 2, "tags": "d"},
    #     {"id": 2, "tags": "e"}
    # ]
    ```
    """

    def syntax_check(self) -> None:
        """
        Checks if the required configuration key is present in the operation's config.

        Raises:
            ValueError: If the required 'explode_key' is missing from the configuration.
        """

        required_keys = ["explode_key"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(
                    f"Missing required key '{key}' in ExplodeOperation configuration"
                )

    def execute(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        """
        Executes the explode operation on the input data.

        Args:
            input_data (List[Dict]): A list of dictionaries to process.

        Returns:
            Tuple[List[Dict], float]: A tuple containing the processed list of dictionaries
            and a float value (always 0 in this implementation).

        Raises:
            KeyError: If the specified explode_key is not found in an input dictionary.
            TypeError: If the value of the explode_key is not iterable (list, tuple, or set).

        Example:
        ```python
        explode_op = ExplodeOperation({"explode_key": "colors"})
        input_data = [
            {"id": 1, "colors": ["red", "blue"]},
            {"id": 2, "colors": ["green"]}
        ]
        result, _ = explode_op.execute(input_data)
        # Result will be:
        # [
        #     {"id": 1, "colors": "red"},
        #     {"id": 1, "colors": "blue"},
        #     {"id": 2, "colors": "green"}
        # ]
        ```
        """

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
