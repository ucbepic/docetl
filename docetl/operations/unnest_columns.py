import copy
from typing import Optional

from docetl.operations.base import BaseOperation, Cardinality
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

    # ── plan traits ────────────────────────────────────────────────

    @classmethod
    def cardinality(cls, config):
        return Cardinality.ONE_TO_ONE

    @classmethod
    def fields_read(cls, config):
        if not config.get("unnest_key"):
            return None
        return frozenset({config["unnest_key"]})

    @classmethod
    def fields_written(cls, config):
        if config.get("keys") is None:
            return None  # expanded columns come from runtime dict keys
        # unnest_key itself is removed from the output rows
        return frozenset(config["keys"]) | frozenset({config.get("unnest_key")})

    @classmethod
    def fields_removed(cls, config):
        removed = set(super().fields_removed(config))
        if config.get("unnest_key"):
            removed.add(config["unnest_key"])
        return frozenset(removed)

    @classmethod
    def is_row_local(cls, config):
        return True

    @classmethod
    def preserves_order(cls, config):
        return True

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
