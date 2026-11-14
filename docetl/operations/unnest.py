import copy
import re

from jinja2 import Environment, TemplateSyntaxError, nodes
from pydantic import field_validator

from docetl.operations.base import BaseOperation, Cardinality


def _field_path(expression):
    """Return the address of a field, excluding computed Jinja expressions."""
    error = "unnest_key must be a field path with literal string or integer indices"
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError(error)
    try:
        body = Environment().parse("{{ " + expression + " }}").body
    except TemplateSyntaxError as exc:
        raise ValueError(error) from exc
    if (
        len(body) != 1
        or not isinstance(body[0], nodes.Output)
        or len(body[0].nodes) != 1
    ):
        raise ValueError(error)
    node = body[0].nodes[0]
    parts = []
    while isinstance(node, (nodes.Getattr, nodes.Getitem)):
        if isinstance(node, nodes.Getattr):
            parts.append(node.attr)
        else:
            index = node.arg
            if (
                isinstance(index, nodes.Neg)
                and isinstance(index.node, nodes.Const)
                and type(index.node.value) is int
            ):
                parts.append(-index.node.value)
            elif isinstance(index, nodes.Const) and type(index.value) in (str, int):
                parts.append(index.value)
            else:
                raise ValueError(error)
        node = node.node
    if not isinstance(node, nodes.Name):
        # The string contains an invalid expression, not an invalid argument type.
        raise ValueError(error)  # noqa: TRY004
    parts.append(node.name)
    return tuple(reversed(parts))


def _lookup_path(item, path):
    for key in path:
        item = item[key]
    return item


def _replace_path(item, path, value):
    key = path[0]
    replacement = value if len(path) == 1 else _replace_path(item[key], path[1:], value)
    if isinstance(item, tuple):
        updated = list(item)
        updated[key] = replacement
        return tuple(updated)
    # deepcopy preserves aliases; copy the path to leave aliased siblings intact.
    updated = copy.copy(item)
    updated[key] = replacement
    return updated


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

    class schema(BaseOperation.schema):
        type: str = "unnest"
        unnest_key: str
        keep_empty: bool | None = None
        expand_fields: list[str] | None = None
        recursive: bool | None = None
        depth: int | None = None

        @field_validator("unnest_key")
        @classmethod
        def validate_unnest_key(cls, value):
            _field_path(value)
            return value

    @classmethod
    def transform_schema(cls, schema, config):
        result = super().transform_schema(schema, config)
        try:
            path = _field_path(config.get("unnest_key"))
        except ValueError:
            path = ()
        key = path[0] if len(path) == 1 else None
        declared = result.get(key)
        if isinstance(declared, str):
            # list[X] fields become one X per row
            match = re.fullmatch(r"list\[(.+)\]", declared.strip())
            if match:
                result[key] = match.group(1)
        for field in config.get("expand_fields") or []:
            # expanded from dict values; element types aren't declared anywhere
            result.setdefault(field, "string")
        return result

    # ── plan traits ────────────────────────────────────────────────

    @classmethod
    def cardinality(cls, config):
        return Cardinality.ONE_TO_MANY

    @classmethod
    def fields_read(cls, config):
        try:
            return frozenset({_field_path(config.get("unnest_key"))[0]})
        except ValueError:
            return None

    @classmethod
    def fields_written(cls, config):
        # List unnest overwrites unnest_key with each element, but a
        # dict-valued unnest_key with no expand_fields configured expands
        # to the dict's *runtime* keys (execute defaults expand_fields to
        # item[key].keys()) — statically unknowable, so fail closed.
        roots = cls.fields_read(config)
        if roots is None or config.get("expand_fields") is None:
            return None
        return roots | frozenset(config["expand_fields"])

    @classmethod
    def is_row_local(cls, config):
        return True

    @classmethod
    def preserves_order(cls, config):
        return True

    def execute(self, input_data: list[dict]) -> tuple[list[dict], float]:
        """
        Executes the unnest operation on the input data.

        Args:
            input_data (list[dict]): A list of dictionaries to process.

        Returns:
            tuple[list[dict], float]: A tuple containing the processed list of dictionaries
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
        path = _field_path(unnest_key)
        recursive = self.config.get("recursive", False)
        depth = self.config.get("depth", None)
        if not depth:
            depth = 1 if not recursive else float("inf")
        results = []

        def unnest_recursive(item, value, level=0):
            if level == 0 and not isinstance(value, (list, tuple, set, dict)):
                raise TypeError(f"Value of unnest key '{unnest_key}' is not iterable")

            if level > 0 and not isinstance(value, (list, tuple, set, dict)):
                return [item]

            if level >= depth:
                return [item]

            if isinstance(value, dict):
                expand_fields = self.config.get("expand_fields")
                if expand_fields is None:
                    expand_fields = value.keys()
                new_item = copy.deepcopy(item)
                nested = _lookup_path(new_item, path)
                for field in expand_fields:
                    new_item[field] = nested.get(field)
                return [new_item]
            else:
                if not value and self.config.get("keep_empty", False):
                    return [_replace_path(copy.deepcopy(item), path, None)]
                nested_results = []
                for element in value:
                    new_item = copy.deepcopy(item)
                    new_value = copy.deepcopy(element)
                    new_item = _replace_path(new_item, path, new_value)
                    if recursive and isinstance(new_value, (list, tuple, set, dict)):
                        nested_results.extend(
                            unnest_recursive(new_item, new_value, level + 1)
                        )
                    else:
                        nested_results.append(new_item)
                return nested_results

        for item in input_data:
            try:
                unnest_value = _lookup_path(item, path)
            except (KeyError, IndexError, TypeError) as exc:
                raise KeyError(
                    f"Unnest key '{unnest_key}' not found in item. Other keys are {item.keys()}"
                ) from exc

            results.extend(unnest_recursive(item, unnest_value))

        # Assert that no keys are missing after the operation
        if results:
            original_keys = set(input_data[0].keys())
            assert original_keys.issubset(
                set(results[0].keys())
            ), "Keys lost during unnest operation"

        return results, 0
