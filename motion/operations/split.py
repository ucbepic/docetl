from typing import Dict, List, Any, Tuple
import tiktoken
from motion.operations.base import BaseOperation
from rich.console import Console
import math


class SplitOperation(BaseOperation):
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

        if "peripheral_chunks" in self.config:
            if not isinstance(self.config["peripheral_chunks"], dict):
                raise TypeError("'peripheral_chunks' must be a dictionary")

            for direction in ["previous", "next"]:
                if direction in self.config["peripheral_chunks"]:
                    direction_config = self.config["peripheral_chunks"][direction]
                    if not isinstance(direction_config, dict):
                        raise TypeError(
                            f"'peripheral_chunks.{direction}' must be a dictionary"
                        )

                    for section in ["head", "middle", "tail"]:
                        if section in direction_config:
                            section_config = direction_config[section]
                            if not isinstance(section_config, dict):
                                raise TypeError(
                                    f"'peripheral_chunks.{direction}.{section}' must be a dictionary"
                                )

                            if "type" in section_config and section_config[
                                "type"
                            ] not in ["full", "summary"]:
                                raise ValueError(
                                    f"'type' in {direction}.{section} must be 'full' or 'summary'"
                                )

                            if section != "middle":
                                if (
                                    "count" not in section_config
                                    or not isinstance(
                                        section_config["count"], (int, float)
                                    )
                                    or section_config["count"] <= 0
                                ):
                                    raise ValueError(
                                        f"'count' in {direction}.{section} must be a positive number"
                                    )

        # Check if the model is specified (optional)
        if "model" in self.config and not isinstance(self.config["model"], str):
            raise TypeError("'model' in configuration must be a string")

        # Check for main chunk delimiters (also optional)
        if "main_chunk_start" in self.config and not isinstance(
            self.config["main_chunk_start"], str
        ):
            raise TypeError("'main_chunk_start' must be a string")
        if "main_chunk_end" in self.config and not isinstance(
            self.config["main_chunk_end"], str
        ):
            raise TypeError("'main_chunk_end' must be a string")

    def execute(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        split_key = self.config["split_key"]
        chunk_size = self.config["chunk_size"]
        peripheral_chunks = self.config.get("peripheral_chunks", {})
        main_chunk_start = self.config.get("main_chunk_start", "<MAIN_CHUNK>")
        main_chunk_end = self.config.get("main_chunk_end", "</MAIN_CHUNK>")
        results = []

        encoder = tiktoken.encoding_for_model(
            self.config.get("model", self.default_model)
        )

        for item in input_data:
            if split_key not in item:
                raise KeyError(f"Split key '{split_key}' not found in item")

            content = item[split_key]
            tokens = encoder.encode(content)
            chunks = []

            # Split tokens into chunks
            for i in range(0, len(tokens), chunk_size):
                chunk_tokens = tokens[i : i + chunk_size]
                chunk = encoder.decode(chunk_tokens)
                chunk_id = f"chunk_{i//chunk_size}"
                chunks.append({"chunk_id": chunk_id, "chunk_content": chunk})

            # Process chunks with peripheral context
            for i, chunk in enumerate(chunks):
                previous_chunks = self.process_peripheral_chunks(
                    chunks[:i], peripheral_chunks.get("previous", {})
                )
                next_chunks = self.process_peripheral_chunks(
                    chunks[i + 1 :], peripheral_chunks.get("next", {}), reverse=True
                )

                # Create result with chunk intermediates
                chunk_intermediates = {
                    "chunk_content": chunk["chunk_content"],
                    "previous_chunks": previous_chunks,
                    "next_chunks": next_chunks,
                    split_key: content,
                }

                # Create result with only original item key-value pairs, chunk_id, and chunk
                result = item.copy()
                result.update(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "chunk_content": self.combine_chunks(
                            previous_chunks,
                            chunk,
                            next_chunks,
                            encoder,
                            main_chunk_start,
                            main_chunk_end,
                        ),
                        "_chunk_intermediates": chunk_intermediates,
                    }
                )

                results.append(result)

        return results, 0

    def process_peripheral_chunks(self, chunks, config, reverse=False):
        if reverse:
            chunks = list(reversed(chunks))

        processed_chunks = []
        total_chunks = len(chunks)

        # Process head
        head_config = config.get("head", {})
        head_count = min(math.floor(head_config.get("count", 0)), total_chunks)
        head_partial = min(head_config.get("count", 0) - head_count, 1)
        for i in range(head_count):
            processed_chunks.append(
                self.process_chunk(chunks[i], head_config.get("type", "full"))
            )
        if head_partial > 0 and head_count < total_chunks:
            processed_chunks.append(
                self.process_partial_chunk(
                    chunks[head_count], head_config.get("type", "full"), head_partial
                )
            )

        # Process tail
        tail_config = config.get("tail", {})
        tail_count = min(
            math.floor(tail_config.get("count", 0)), total_chunks - head_count
        )
        tail_partial = min(tail_config.get("count", 0) - tail_count, 1)
        tail_start = max(head_count, total_chunks - tail_count)
        tail_chunks = []
        for i in range(tail_start, total_chunks):
            tail_chunks.append(
                self.process_chunk(chunks[i], tail_config.get("type", "full"))
            )
        if (
            tail_partial > 0
            and tail_start > head_count
            and tail_start - 1 < total_chunks
        ):
            tail_chunks.insert(
                0,
                self.process_partial_chunk(
                    chunks[tail_start - 1],
                    tail_config.get("type", "full"),
                    tail_partial,
                ),
            )

        # Process middle
        middle_config = config.get("middle", {})
        if middle_config:
            middle_start = head_count + (1 if head_partial > 0 else 0)
            middle_end = tail_start - (1 if tail_partial > 0 else 0)
            for i in range(middle_start, middle_end):
                processed_chunks.append(
                    self.process_chunk(chunks[i], middle_config.get("type", "summary"))
                )

        processed_chunks.extend(tail_chunks)

        return list(reversed(processed_chunks)) if reverse else processed_chunks

    def process_chunk(self, chunk, chunk_type):
        if chunk_type == "full":
            return chunk
        else:  # chunk_type == "summary"
            return self.summarize_chunk(chunk)

    def process_partial_chunk(self, chunk, chunk_type, ratio):
        if chunk_type == "full":
            partial_content = chunk["chunk_content"][
                : int(len(chunk["chunk_content"]) * ratio)
            ]
            return {
                "chunk_id": chunk["chunk_id"],
                "chunk_content": partial_content,
                "fraction": ratio,
            }
        else:  # chunk_type == "summary"
            return self.summarize_chunk(chunk, ratio)

    def summarize_chunk(
        self, chunk: Dict[str, str], ratio: float = 1.0
    ) -> Dict[str, str]:
        # This is a placeholder function. In a real implementation,
        # you would want to implement an actual summarization algorithm here.
        summary = f"Prefix of {chunk['chunk_id']}: {chunk['chunk_content'][:50]}..."
        return {
            "chunk_id": chunk["chunk_id"],
            "summary": summary[: int(len(summary) * ratio)],
            "fraction": ratio,
        }

    def combine_chunks(
        self,
        previous_chunks,
        main_chunk,
        next_chunks,
        encoder,
        main_chunk_start,
        main_chunk_end,
    ):
        combined_parts = []

        # Process previous chunks
        if previous_chunks:
            combined_parts.append("--- Previous Context ---")
            self._process_peripheral_chunks_for_combination(
                previous_chunks, combined_parts, encoder
            )
            combined_parts.append("--- End Previous Context ---\n")
        else:
            # Show skipped tokens even if there are no previous chunks
            main_chunk_num = int(main_chunk["chunk_id"].split("_")[1])
            skipped_tokens = max(0, (main_chunk_num - 1) * self.config["chunk_size"])
            if skipped_tokens > 0:
                combined_parts.append(
                    f"[... {skipped_tokens} tokens skipped before this chunk ...]"
                )

        # Process main chunk
        if not previous_chunks and not next_chunks:
            # Do not use delimiters if there are no peripherals
            combined_parts.append(main_chunk["chunk_content"])
        else:
            # If there are peripherals, use delimiters
            combined_parts.append(
                f"{main_chunk_start}\n{main_chunk['chunk_content']}\n{main_chunk_end}"
            )

        # Process next chunks
        if next_chunks:
            combined_parts.append("\n--- Next Context ---")
            self._process_peripheral_chunks_for_combination(
                next_chunks, combined_parts, encoder
            )
            combined_parts.append("--- End Next Context ---")
        else:
            # Show skipped tokens even if there are no next chunks
            total_chunks = int(main_chunk["chunk_id"].split("_")[1])
            skipped_tokens = (
                total_chunks - int(main_chunk["chunk_id"].split("_")[1])
            ) * self.config["chunk_size"]
            combined_parts.append(
                f"[... {skipped_tokens} tokens skipped after this chunk ...]"
            )

        return "\n".join(combined_parts)

    def _process_peripheral_chunks_for_combination(
        self, chunks, combined_parts, encoder
    ):
        last_chunk_num = None
        for chunk in chunks:
            current_chunk_num = int(chunk["chunk_id"].split("_")[1])

            if last_chunk_num is not None and current_chunk_num - last_chunk_num > 1:
                skipped_start = last_chunk_num + 1
                skipped_end = current_chunk_num - 1
                skipped_tokens = (skipped_end - skipped_start + 1) * self.config[
                    "chunk_size"
                ]
                combined_parts.append(
                    f"[... {skipped_tokens} tokens skipped between chunks {skipped_start} and {skipped_end} ...]"
                )

            if "summary" in chunk:
                combined_parts.append(
                    f"[Chunk {current_chunk_num} Summary] {chunk['summary'].strip()}"
                )
            elif "chunk_content" in chunk:
                combined_parts.append(
                    f"[Chunk {current_chunk_num}] {chunk['chunk_content'].strip()}"
                )

            last_chunk_num = current_chunk_num
