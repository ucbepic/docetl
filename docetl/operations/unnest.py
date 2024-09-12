import copy
from typing import Dict, List, Tuple

from docetl.operations.base import BaseOperation


class UnnestOperation(BaseOperation):
    """
    A class that represents an operation to unnest a list-like or dictionary value in a dictionary into multiple dictionaries.

    This operation takes a list of dictionaries and a specified key, and creates new dictionaries based on the value type:
    - For list-like values: Creates a new dictionary for each element in the list, copying all other key-value pairs.
    - For dictionary values: Expands specified fields from the nested dictionary into the parent dictionary.

    Inherits from:
        BaseOperation

    Usage:
    ```python
    from docetl.operations import UnnestOperation

    # Unnesting a list
    config_list = {"unnest_key": "tags"}
    input_data_list = [
        {"id": 1, "tags": ["a", "b", "c"]},
        {"id": 2, "tags": ["d", "e"]}
    ]

    unnest_op_list = UnnestOperation(config_list)
    result_list, _ = unnest_op_list.execute(input_data_list)

    # Result will be:
    # [
    #     {"id": 1, "tags": "a"},
    #     {"id": 1, "tags": "b"},
    #     {"id": 1, "tags": "c"},
    #     {"id": 2, "tags": "d"},
    #     {"id": 2, "tags": "e"}
    # ]

    # Unnesting a dictionary
    config_dict = {"unnest_key": "user", "expand_fields": ["name", "age"]}
    input_data_dict = [
        {"id": 1, "user": {"name": "Alice", "age": 30, "email": "alice@example.com"}},
        {"id": 2, "user": {"name": "Bob", "age": 25, "email": "bob@example.com"}}
    ]

    unnest_op_dict = UnnestOperation(config_dict)
    result_dict, _ = unnest_op_dict.execute(input_data_dict)

    # Result will be:
    # [
    #     {"id": 1, "name": "Alice", "age": 30, "user": {"name": "Alice", "age": 30, "email": "alice@example.com"}},
    #     {"id": 2, "name": "Bob", "age": 25, "user": {"name": "Bob", "age": 25, "email": "bob@example.com"}}
    # ]
    ```
    """

    def syntax_check(self) -> None:
        """
        Checks if the required configuration key is present in the operation's config.

        Raises:
            ValueError: If the required 'unnest_key' is missing from the configuration.
        """

        required_keys = ["unnest_key"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(
                    f"Missing required key '{key}' in UnnestOperation configuration"
                )

    def execute(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        """
        Executes the unnest operation on the input data.

        Args:
            input_data (List[Dict]): A list of dictionaries to process.

        Returns:
            Tuple[List[Dict], float]: A tuple containing the processed list of dictionaries
            and a float value (always 0 in this implementation).

        Raises:
            KeyError: If the specified unnest_key is not found in an input dictionary.
            TypeError: If the value of the unnest_key is not iterable (list, tuple, set, or dict).
            ValueError: If unnesting a dictionary and 'expand_fields' is not provided in the config.

        The operation supports unnesting of both list-like values and dictionary values:

        1. For list-like values (list, tuple, set):
           Each element in the list becomes a separate dictionary in the output.

        2. For dictionary values:
           The operation expands specified fields from the nested dictionary into the parent dictionary.
           The 'expand_fields' config parameter must be provided to specify which fields to expand.

        Examples:
        ```python
        # Unnesting a list
        unnest_op = UnnestOperation({"unnest_key": "colors"})
        input_data = [
            {"id": 1, "colors": ["red", "blue"]},
            {"id": 2, "colors": ["green"]}
        ]
        result, _ = unnest_op.execute(input_data)
        # Result will be:
        # [
        #     {"id": 1, "colors": "red"},
        #     {"id": 1, "colors": "blue"},
        #     {"id": 2, "colors": "green"}
        # ]

        # Unnesting a dictionary
        unnest_op = UnnestOperation({"unnest_key": "details", "expand_fields": ["color", "size"]})
        input_data = [
            {"id": 1, "details": {"color": "red", "size": "large", "stock": 5}},
            {"id": 2, "details": {"color": "blue", "size": "medium", "stock": 3}}
        ]
        result, _ = unnest_op.execute(input_data)
        # Result will be:
        # [
        #     {"id": 1, "details": {"color": "red", "size": "large", "stock": 5}, "color": "red", "size": "large"},
        #     {"id": 2, "details": {"color": "blue", "size": "medium", "stock": 3}, "color": "blue", "size": "medium"}
        # ]
        ```

        Note: When unnesting dictionaries, the original nested dictionary is preserved in the output,
        and the specified fields are expanded into the parent dictionary.
        """

        unnest_key = self.config["unnest_key"]
        recursive = self.config.get("recursive", False)
        depth = self.config.get("depth", None)
        if not depth:
            depth = 1 if not recursive else float("inf")
        results = []

        def unnest_recursive(item, key, level=0):
            if level == 0 and not isinstance(item[key], (list, tuple, set, dict)):
                raise TypeError(f"Value of unnest key '{key}' is not iterable")

            if level > 0 and not isinstance(item[key], (list, tuple, set, dict)):
                return [item]

            if level >= depth:
                return [item]

            if isinstance(item[key], dict):
                expand_fields = self.config.get("expand_fields")
                if expand_fields is None:
                    expand_fields = item[key].keys()
                new_item = copy.deepcopy(item)
                for field in expand_fields:
                    if field in new_item[key]:
                        new_item[field] = new_item[key][field]
                    else:
                        new_item[field] = None
                return [new_item]
            else:
                nested_results = []
                for value in item[key]:
                    new_item = copy.deepcopy(item)
                    new_item[key] = value
                    if recursive and isinstance(value, (list, tuple, set, dict)):
                        nested_results.extend(
                            unnest_recursive(new_item, key, level + 1)
                        )
                    else:
                        nested_results.append(new_item)
                return nested_results

        for item in input_data:
            if unnest_key not in item:
                raise KeyError(
                    f"Unnest key '{unnest_key}' not found in item. Other keys are {item.keys()}"
                )

            results.extend(unnest_recursive(item, unnest_key))

            if not item[unnest_key] and self.config.get("keep_empty", False):
                expand_fields = self.config.get("expand_fields")
                new_item = copy.deepcopy(item)
                if isinstance(item[unnest_key], dict):
                    if expand_fields is None:
                        expand_fields = item[unnest_key].keys()
                    for field in expand_fields:
                        new_item[field] = None
                else:
                    new_item[unnest_key] = None
                results.append(new_item)

        # Assert that no keys are missing after the operation
        if results:
            original_keys = set(input_data[0].keys())
            assert original_keys.issubset(
                set(results[0].keys())
            ), "Keys lost during unnest operation"

        return results, 0
