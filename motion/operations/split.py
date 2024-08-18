from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Tuple
import uuid
from motion.operations.utils import call_llm, parse_llm_response
import tiktoken
from motion.operations.base import BaseOperation
from rich.console import Console
import math
from litellm import completion_cost
from jinja2 import Template


class SplitOperation(BaseOperation):
    """
    A class that implements a split operation on input data, dividing it into manageable chunks with context.

    This class extends BaseOperation to:
    1. Split input data into chunks of specified size based on the 'split_key' and 'chunk_size' configuration.
    2. Add peripheral context to each chunk (previous and next chunks).
    3. Optionally summarize or include full content of peripheral chunks.
    4. Return results containing:
       - chunk_content: A formatted string containing the main chunk and peripheral chunks.
       - chunk_id: A unique identifier for each chunk.
       - _chunk_intermediates: A dictionary containing detailed information about the main chunk and its peripheral chunks, including their content, positions, and any additional metadata used in the splitting process.

    The peripheral configuration impacts chunk content by determining:
    - Number of previous and next chunks to include (specified in 'peripheral_chunks' config).
    - Whether to include full or summarized content for these chunks (specified by 'type' in each direction's config).
    - How partial chunks at the start/end of the sequence are handled (specified in 'head' and 'tail' configs).

    This allows for flexible context-aware splitting of large datasets, ensuring each chunk
    has access to relevant surrounding information.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def syntax_check(self) -> None:
        """
        Perform comprehensive syntax checks on the configuration of the SplitOperation.

        This method validates the presence and correctness of all required configuration keys,
        and ensures the correct structure and types of the entire configuration.

        The method performs the following checks:
        1. Verifies the presence of all required keys in the configuration.
        2. Checks if 'split_key' is a string.
        3. Validates that 'chunk_size' is a positive integer.
        4. If present, checks the structure and content of the 'peripheral_chunks' configuration.
        5. Verifies types of various configuration values (e.g., 'model' as string).
        6. Checks for the presence and validity of optional configurations like 'main_chunk_start' and 'main_chunk_end'.
        7. Validates the 'summary_prompt' if present.

        Raises:
            ValueError: If any required configuration is missing or if any configuration aspect is incorrect or inconsistent.
            TypeError: If any configuration value has an incorrect type.
        """
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

                            # If it's a summary, make sure there's a summary prompt
                            if section_config.get("type") == "summary":
                                if "summary_prompt" not in self.config:
                                    raise ValueError(
                                        f"'summary_prompt' is required in the configuration when using summary type in {direction}.{section}"
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

        # Check for summary_prompt (optional)
        if "summary_prompt" in self.config:
            if not isinstance(self.config["summary_prompt"], str):
                raise TypeError("'summary_prompt' must be a string")
            try:
                Template(self.config["summary_prompt"])
            except Exception as e:
                raise ValueError(f"Invalid Jinja2 template for 'summary_prompt': {e}")

        if "chunk_group_id_field" in self.config:
            if not isinstance(self.config["chunk_group_id_field"], str):
                raise TypeError("'chunk_group_id_field' must be a string")

    def execute(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        """
        Execute the split operation on the provided input data.

        This method splits the input data into chunks based on the specified split key and chunk size.
        It also adds peripheral context to each chunk if specified in the configuration.

        Args:
            input_data (List[Dict]): The input data to process.

        Returns:
            Tuple[List[Dict], float]: A tuple containing the processed results (split chunks with context)
                                      and the total cost of the operation.

        Raises:
            KeyError: If the split key is not found in an input item.
        """
        split_key = self.config["split_key"]
        chunk_size = self.config["chunk_size"]
        peripheral_chunks = self.config.get("peripheral_chunks", {})
        main_chunk_start = self.config.get("main_chunk_start", "<MAIN_CHUNK>")
        main_chunk_end = self.config.get("main_chunk_end", "</MAIN_CHUNK>")
        results = []
        cost = 0.0
        self._summarize_chunks = False
        if "peripheral_chunks" in self.config:
            for direction in ["previous", "next"]:
                if direction in self.config["peripheral_chunks"]:
                    for section in ["head", "middle", "tail"]:
                        if (
                            section in self.config["peripheral_chunks"][direction]
                            and self.config["peripheral_chunks"][direction][
                                section
                            ].get("type", "full")
                            == "summary"
                        ):
                            self._summarize_chunks = True
                            break

        encoder = tiktoken.encoding_for_model(
            self.config.get("model", self.default_model)
        )

        def process_item(item):
            if split_key not in item:
                raise KeyError(f"Split key '{split_key}' not found in item")

            content = item[split_key]
            tokens = encoder.encode(content)
            chunks = []

            # Split tokens into chunks
            doc_id = str(uuid.uuid4())
            for i in range(0, len(tokens), chunk_size):
                chunk_tokens = tokens[i : i + chunk_size]
                chunk = encoder.decode(chunk_tokens)

                chunk_id = f"chunk_{i//chunk_size}_{doc_id}"
                if "chunk_group_id_field" in self.config:
                    chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "chunk_content": chunk,
                            self.config["chunk_group_id_field"]: doc_id,
                        }
                    )
                else:
                    chunks.append({"chunk_id": chunk_id, "chunk_content": chunk})

            return item, chunks

        def summarize_chunk(chunk):
            summary, summary_cost = self.summarize_chunk(chunk)
            return {**chunk, "summary": summary}, summary_cost

        with ThreadPoolExecutor(self.max_threads) as executor:
            # Process items in parallel
            item_chunks = list(executor.map(process_item, input_data))

            # Summarize chunks if needed
            if self._summarize_chunks:
                all_chunks = [chunk for _, chunks in item_chunks for chunk in chunks]
                summarized_chunks = list(executor.map(summarize_chunk, all_chunks))
                summarized_chunks_dict = {
                    chunk["chunk_id"]: chunk for chunk, _ in summarized_chunks
                }
                cost += sum(summary_cost for _, summary_cost in summarized_chunks)

            # Process chunks with peripheral context
            for item, chunks in item_chunks:
                if self._summarize_chunks:
                    chunks = [
                        summarized_chunks_dict[chunk["chunk_id"]] for chunk in chunks
                    ]

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
                        split_key: item[split_key],
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
                                main_chunk_start,
                                main_chunk_end,
                                len(chunks),
                            ),
                            "_chunk_intermediates": chunk_intermediates,
                        }
                    )
                    if "chunk_group_id_field" in self.config:
                        result[self.config["chunk_group_id_field"]] = chunk[
                            self.config["chunk_group_id_field"]
                        ]

                    results.append(result)

        return results, cost

    def process_peripheral_chunks(
        self, chunks: List[Dict], config: Dict, reverse=False
    ):
        """
        Process peripheral chunks based on the provided configuration.

        This method handles the processing of chunks before or after the main chunk,
        applying the specified configuration for head, middle, and tail sections.

        Args:
            chunks (List[Dict]): The list of chunks to process.
            config (Dict): The configuration for processing peripheral chunks.
            reverse (bool, optional): Whether to process the chunks in reverse order. Defaults to False.

        Returns:
            List[Dict]: The processed peripheral chunks.
        """
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
                self.process_chunk(chunks[i], head_config.get("type", "full"), 1.0)
            )
        if head_partial > 0 and head_count < total_chunks:
            processed_chunks.append(
                self.process_chunk(
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
                self.process_chunk(chunks[i], tail_config.get("type", "full"), 1.0)
            )
        if (
            tail_partial > 0
            and tail_start > head_count
            and tail_start - 1 < total_chunks
        ):
            tail_chunks.insert(
                0,
                self.process_chunk(
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
                    self.process_chunk(
                        chunks[i], middle_config.get("type", "summary"), 1.0
                    )
                )

        processed_chunks.extend(tail_chunks)

        return list(reversed(processed_chunks)) if reverse else processed_chunks

    def process_chunk(
        self, chunk: Dict[str, str], type: str, ratio: float = 1.0
    ) -> Dict[str, str]:
        """
        Process a chunk based on the specified type and ratio.

        This method handles the processing of a chunk, either by summarizing it or
        by returning a portion of its content based on the given ratio.

        Args:
            chunk (Dict[str, str]): The chunk to process.
            type (str): The type of processing to apply ('summary' or 'full').
            ratio (float): The ratio of content to return (0.0 to 1.0).

        Returns:
            Dict[str, str]: The processed chunk with either summarized or partial content.
        """
        if type == "summary":
            summary = chunk["summary"]
            return {**chunk, "summary": summary[: int(len(summary) * ratio)]}
        else:  # 'full' type
            content = chunk["chunk_content"]
            truncated_content = content[: int(len(content) * ratio)]
            return {
                key: (truncated_content if key == "chunk_content" else value)
                for key, value in chunk.items()
                if key != "summary"
            }

    def summarize_chunk(self, chunk: Dict[str, str]) -> Tuple[str, float]:
        """
        Summarize a chunk of text. Using the summary_prompt from the configuration.

        Args:
            chunk (Dict[str, str]): The chunk to summarize.

        Returns:
            Tuple[str, float]: The summarized chunk and the cost of the operation.
        """
        template = Template(self.config["summary_prompt"])
        summary_prompt = template.render(chunk_content=chunk["chunk_content"])

        summary_response = call_llm(
            self.config.get("summary_model", self.default_model),
            "summary",
            summary_prompt,
            {"summary": "str"},
        )
        summary = parse_llm_response(summary_response)[0]

        return summary["summary"], completion_cost(summary_response)

    def combine_chunks(
        self,
        previous_chunks,
        main_chunk,
        next_chunks,
        main_chunk_start,
        main_chunk_end,
        total_chunks,
    ):
        """
        Combine the main chunk with its peripheral chunks.

        This method combines the main chunk with its previous and next chunks,
        adding appropriate delimiters and context information.

        Args:
            previous_chunks (List[Dict]): The chunks before the main chunk.
            main_chunk (Dict): The main chunk to be combined.
            next_chunks (List[Dict]): The chunks after the main chunk.
            main_chunk_start (str): The delimiter to mark the start of the main chunk.
            main_chunk_end (str): The delimiter to mark the end of the main chunk.

        Returns:
            str: The combined chunk content with context.
        """
        combined_parts = []

        # Process previous chunks
        if previous_chunks:
            combined_parts.append("--- Previous Context ---")
            self._process_peripheral_chunks_for_combination(
                previous_chunks, combined_parts
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
            self._process_peripheral_chunks_for_combination(next_chunks, combined_parts)
            combined_parts.append("--- End Next Context ---")
        else:
            # Show skipped tokens even if there are no next chunks
            skipped_tokens = (
                total_chunks - int(main_chunk["chunk_id"].split("_")[1])
            ) * self.config["chunk_size"]
            combined_parts.append(
                f"[... {skipped_tokens} tokens skipped after this chunk ...]"
            )

        return "\n".join(combined_parts)

    def _process_peripheral_chunks_for_combination(self, chunks, combined_parts):
        """
        Process peripheral chunks for combination with the main chunk.

        This method processes the peripheral chunks, adding them to the combined_parts list
        and inserting information about skipped tokens between non-consecutive chunks.

        Args:
            chunks (List[Dict]): The peripheral chunks to process.
            combined_parts (List[str]): The list to append processed chunk information to.
        """
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
