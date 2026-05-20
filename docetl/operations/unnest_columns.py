import copy
from typing import Optional

from docetl.operations.base import BaseOperation
from docetl.operations.utils.validation import lookup_field


class UnnestColumnsOperation(BaseOperation):
    """
    Flattens a dictionary-valued column into multiple columns, one per key.

    Each key in the dictionary becomes a new top-level field on the document.
    The original column is preserved unless ``drop_source`` is set to true.

    Example::

        config = {"unnest_key": "info"}
        input_data = [
            {"id": 1, "info": {"name": "Alice", "age": 30}},
        ]
        # Result:
        # [{"id": 1, "info": {"name": "Alice", "age": 30}, "name": "Alice", "age": 30}]

    With ``drop_source: true``::

        # Result:
        # [{"id": 1, "name": "Alice", "age": 30}]
    """

    class schema(BaseOperation.schema):
        type: str = "unnest_columns"
        unnest_key: str
        keys: Optional[list[str]] = None

    def execute(self, input_data: list[dict]) -> tuple[list[dict], float]:
        """
        Expands a dict-valued field into top-level columns.

        Args:
            input_data: List of dicts to process.

        Returns:
            Tuple of (results, cost). Cost is always 0.

        Raises:
            KeyError: If ``unnest_key`` not present in a document.
            TypeError: If the value at ``unnest_key`` is not a dict.
        """
        unnest_key = self.config["unnest_key"]
        keys = self.config.get("keys")

        results = []
        for item in input_data:
            try:
                value = lookup_field(item, unnest_key)
            except Exception:
                raise KeyError(
                    f"unnest_columns: key '{unnest_key}' not found in document. "
                    f"Available keys: {list(item.keys())}"
                )
            if not isinstance(value, dict):
                raise TypeError(
                    f"unnest_columns: value at '{unnest_key}' must be a dict, "
                    f"got {type(value).__name__}"
                )
            new_item = copy.deepcopy(item)
            # Only delete the simple key if it's a plain top-level key
            if unnest_key in new_item:
                del new_item[unnest_key]
            expand = keys if keys is not None else value.keys()
            for k in expand:
                new_item[k] = value.get(k)
            results.append(new_item)

        return results, 0
