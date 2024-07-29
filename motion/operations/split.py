from typing import Dict, List, Any, Tuple
import tiktoken
from motion.operations.base import BaseOperation
from rich.console import Console


class SplitOperation(BaseOperation):
    def syntax_check(self) -> None:
        required_keys = ["split_key", "chunk_size", "overlap_size"]
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

        if (
            not isinstance(self.config["overlap_size"], int)
            or self.config["overlap_size"] < 0
        ):
            raise ValueError("'overlap_size' must be a non-negative integer")

        if self.config["overlap_size"] >= self.config["chunk_size"]:
            raise ValueError("'overlap_size' must be less than 'chunk_size'")

        # Check if the model is specified (optional)
        if "model" in self.config and not isinstance(self.config["model"], str):
            raise TypeError("'model' in configuration must be a string")

    def execute(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        split_key = self.config["split_key"]
        chunk_size = self.config["chunk_size"]
        overlap_size = self.config["overlap_size"]
        results = []

        encoder = tiktoken.encoding_for_model(
            self.config.get("model", self.default_model)
        )

        for item in input_data:
            if split_key not in item:
                raise KeyError(f"Split key '{split_key}' not found in item")

            content = item[split_key]
            tokens = encoder.encode(content)
            start = 0

            while start < len(tokens):
                end = start + chunk_size
                chunk_tokens = tokens[start:end]
                chunk = encoder.decode(chunk_tokens)
                chunk_id = f"chunk_{start}_{end}"
                chunk_data = {"chunk_id": chunk_id, "chunk_content": chunk, **item}

                results.append(chunk_data)

                start = end - overlap_size

        return results, 0
