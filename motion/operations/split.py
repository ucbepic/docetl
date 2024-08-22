import uuid
from typing import Dict, List, Tuple

import tiktoken
from motion.operations.base import BaseOperation


class SplitOperation(BaseOperation):
    """
    A class that implements a split operation on input data, dividing it into manageable chunks.

    This class extends BaseOperation to:
    1. Split input data into chunks of specified size based on the 'split_key' and 'chunk_size' configuration.
    2. Assign unique identifiers to each original document and number chunks sequentially.
    3. Return results containing:
       - {split_key}_chunk: The content of the split chunk.
       - {name}_id: A unique identifier for each original document.
       - {name}_chunk_num: The sequential number of the chunk within its original document.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.config["name"]

    def syntax_check(self) -> None:
        required_keys = ["split_key", "chunk_size"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(
                    f"Missing required key '{key}' in SplitOperation configuration"
                )

        if not isinstance(self.config["split_key"], str):
            raise TypeError("'split_key' must be a string")

        if (
            not isinstance(self.config["chunk_size"], int)
            or self.config["chunk_size"] <= 0
        ):
            raise ValueError("'chunk_size' must be a positive integer")

        if "model" in self.config and not isinstance(self.config["model"], str):
            raise TypeError("'model' in configuration must be a string")

    def execute(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        split_key = self.config["split_key"]
        chunk_size = self.config["chunk_size"]
        results = []
        cost = 0.0

        encoder = tiktoken.encoding_for_model(
            self.config.get("model", self.default_model)
        )

        for item in input_data:
            if split_key not in item:
                raise KeyError(f"Split key '{split_key}' not found in item")

            content = item[split_key]
            tokens = encoder.encode(content)

            # Generate a unique document ID
            doc_id = str(uuid.uuid4())

            for chunk_num, i in enumerate(range(0, len(tokens), chunk_size), start=1):
                chunk_tokens = tokens[i : i + chunk_size]
                chunk = encoder.decode(chunk_tokens)

                result = item.copy()
                result.update(
                    {
                        f"{split_key}_chunk": chunk,
                        f"{self.name}_id": doc_id,
                        f"{self.name}_chunk_num": chunk_num,
                    }
                )

                results.append(result)

        return results, cost
