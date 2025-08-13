import uuid
from typing import Any

from docetl.operations.base import BaseOperation


class AddUuidOperation(BaseOperation):
    """
    A class that implements an operation to add a UUID to each document.

    This class extends BaseOperation to:
    1. Generate a unique UUID for each document
    2. Add the UUID under a key formatted as {operation_name}_id
    """

    class schema(BaseOperation.schema):
        type: str = "add_uuid"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.config["name"]

    def execute(
        self, input_data: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], float]:
        results = []
        cost = 0.0

        # If there's an id key in the config, use that as the id key
        if "id_key" in self.config:
            id_key = self.config["id_key"]
        else:
            id_key = f"{self.name}_id"

        for item in input_data:
            result = item.copy()
            result[id_key] = str(uuid.uuid4())
            results.append(result)

        return results, cost
