from typing import Any

from pydantic import field_validator

from docetl.operations.base import BaseOperation
from docetl.operations.sample import SampleOperation
from docetl.operations.utils import strict_render


class GatherOperation(BaseOperation):
    """
    A class that implements a gather operation on input data, adding contextual information from surrounding chunks.

    This class extends BaseOperation to:
    1. Group chunks by their document ID.
    2. Order chunks within each group.
    3. Add peripheral context to each chunk based on the configuration.
    4. Include headers for each chunk and its upward hierarchy.
    5. Return results containing the rendered chunks with added context, including information about skipped characters and headers.
    """

    class schema(BaseOperation.schema):
        type: str = "gather"
        content_key: str
        doc_id_key: str
        order_key: str
        peripheral_chunks: dict[str, Any] | None = None
        retrieval: dict[str, Any] | None = None
        doc_header_key: str | None = None
        main_chunk_start: str | None = None
        main_chunk_end: str | None = None

        @field_validator("peripheral_chunks")
        def validate_peripheral_chunks(cls, v):
            if v is None:
                return v
            for direction in ["previous", "next"]:
                if direction not in v:
                    continue
                for section in ["head", "middle", "tail"]:
                    if section in v[direction]:
                        section_config = v[direction][section]
                        if section != "middle" and "count" not in section_config:
                            raise ValueError(
                                f"Missing 'count' in {direction}.{section} configuration"
                            )
            return v

        @field_validator("retrieval")
        def validate_retrieval(cls, v):
            if v is None:
                return v
            if not isinstance(v, dict):
                raise TypeError("'retrieval' must be a dictionary")
            for req in ["method", "k", "keys", "query"]:
                if req not in v:
                    raise ValueError(f"Missing '{req}' in retrieval configuration")
            if v["method"] not in {"embedding", "fts"}:
                raise ValueError("'retrieval.method' must be 'embedding' or 'fts'")
            if not isinstance(v["k"], (int, float)) or v["k"] <= 0:
                raise ValueError("'retrieval.k' must be a positive number")
            keys = v["keys"]
            if not isinstance(keys, list) or not all(isinstance(x, str) for x in keys):
                raise TypeError("'retrieval.keys' must be a list of strings")
            if "embedding_model" in v and not isinstance(v["embedding_model"], str):
                raise TypeError("'retrieval.embedding_model' must be a string")
            if "content_key" in v and not isinstance(v["content_key"], str):
                raise TypeError("'retrieval.content_key' must be a string")
            return v

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the GatherOperation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)

    def syntax_check(self) -> None:
        """Perform a syntax check on the operation configuration."""
        # Validate the schema using Pydantic
        self.schema(**self.config)

    def execute(self, input_data: list[dict]) -> tuple[list[dict], float]:
        """
        Execute the gather operation on the input data.

        Args:
            input_data (list[dict]): The input data to process.

        Returns:
            tuple[list[dict], float]: A tuple containing the processed results and the cost of the operation.
        """
        content_key = self.config["content_key"]
        doc_id_key = self.config["doc_id_key"]
        order_key = self.config["order_key"]
        peripheral_config = self.config.get("peripheral_chunks", {})
        retrieval_config = self.config.get("retrieval", {}) or {}
        main_chunk_start = self.config.get(
            "main_chunk_start", "--- Begin Main Chunk ---"
        )
        main_chunk_end = self.config.get("main_chunk_end", "--- End Main Chunk ---")
        doc_header_key = self.config.get("doc_header_key", None)
        results = []
        cost = 0.0

        # Group chunks by document ID
        grouped_chunks = {}
        for item in input_data:
            doc_id = item[doc_id_key]
            if doc_id not in grouped_chunks:
                grouped_chunks[doc_id] = []
            grouped_chunks[doc_id].append(item)

        # Process each group of chunks
        for chunks in grouped_chunks.values():
            # Sort chunks by their order within the document
            chunks.sort(key=lambda x: x[order_key])

            # Process each chunk with its peripheral context and headers
            for i, chunk in enumerate(chunks):
                rendered_chunk, add_cost = self.render_chunk_with_context(
                    chunks,
                    i,
                    peripheral_config,
                    retrieval_config,
                    content_key,
                    order_key,
                    main_chunk_start,
                    main_chunk_end,
                    doc_header_key,
                )

                result = chunk.copy()
                result[f"{content_key}_rendered"] = rendered_chunk
                results.append(result)
                cost += add_cost

        return results, cost

    def render_chunk_with_context(
        self,
        chunks: list[dict],
        current_index: int,
        peripheral_config: dict,
        retrieval_config: dict,
        content_key: str,
        order_key: str,
        main_chunk_start: str,
        main_chunk_end: str,
        doc_header_key: str,
    ) -> tuple[str, float]:
        """
        Render a chunk with its peripheral context and headers.

        Args:
            chunks (list[dict]): List of all chunks in the document.
            current_index (int): Index of the current chunk being processed.
            peripheral_config (dict): Configuration for peripheral chunks.
            content_key (str): Key for the content in each chunk.
            order_key (str): Key for the order of each chunk.
            main_chunk_start (str): String to mark the start of the main chunk.
            main_chunk_end (str): String to mark the end of the main chunk.
            doc_header_key (str): The key for the headers in the current chunk.

        Returns:
            str: Renderted chunk with context and headers.
        """

        # If there is no context configuration at all, return the main chunk only
        if not peripheral_config and not retrieval_config:
            return chunks[current_index][content_key], 0.0

        combined_parts = ["--- Previous Context ---"]
        total_cost = 0.0

        combined_parts.extend(
            self.process_peripheral_chunks(
                chunks[:current_index],
                peripheral_config.get("previous", {}),
                content_key,
                order_key,
            )
        )
        # Retrieved context (position-agnostic) - include here if desired before main
        if retrieval_config:
            candidates_prev = chunks[:current_index]
            retrieved_lines_prev, c = self._retrieve_and_render(
                candidate_chunks=candidates_prev,
                retrieval_cfg=retrieval_config,
                main_chunk=chunks[current_index],
                content_key=content_key,
                order_key=order_key,
            )
            total_cost += c
            if retrieved_lines_prev:
                combined_parts.append("--- Retrieved Context ---")
                combined_parts.extend(retrieved_lines_prev)
                combined_parts.append("--- End Retrieved Context ---")

        combined_parts.append("--- End Previous Context ---\n")

        # Process main chunk
        main_chunk = chunks[current_index]
        if headers := self.render_hierarchy_headers(
            main_chunk, chunks[: current_index + 1], doc_header_key
        ):
            combined_parts.append(headers)
        combined_parts.extend(
            (
                f"{main_chunk_start}",
                f"{main_chunk[content_key]}",
                f"{main_chunk_end}",
                "\n--- Next Context ---",
            )
        )
        combined_parts.extend(
            self.process_peripheral_chunks(
                chunks[current_index + 1 :],
                peripheral_config.get("next", {}),
                content_key,
                order_key,
            )
        )
        # Retrieved context (position-agnostic) - include here if desired after main
        if retrieval_config:
            candidates_next = chunks[current_index + 1 :]
            retrieved_lines_next, c = self._retrieve_and_render(
                candidate_chunks=candidates_next,
                retrieval_cfg=retrieval_config,
                main_chunk=chunks[current_index],
                content_key=content_key,
                order_key=order_key,
            )
            total_cost += c
            if retrieved_lines_next:
                combined_parts.append("--- Retrieved Context ---")
                combined_parts.extend(retrieved_lines_next)
                combined_parts.append("--- End Retrieved Context ---")

        combined_parts.append("--- End Next Context ---")

        # Retrieved context across the document (excluding current chunk)
        if retrieval_config:
            general_candidates = chunks[:current_index] + chunks[current_index + 1 :]
            gen_lines, c = self._retrieve_and_render(
                candidate_chunks=general_candidates,
                retrieval_cfg=retrieval_config,
                main_chunk=chunks[current_index],
                content_key=content_key,
                order_key=order_key,
            )
            total_cost += c
            if gen_lines:
                combined_parts.append("\n--- Retrieved Context ---")
                combined_parts.extend(gen_lines)
                combined_parts.append("--- End Retrieved Context ---")

        return "\n".join(combined_parts), total_cost

    def process_peripheral_chunks(
        self,
        chunks: list[dict],
        config: dict,
        content_key: str,
        order_key: str,
        reverse: bool = False,
    ) -> list[str]:
        """
        Process peripheral chunks according to the configuration.

        Args:
            chunks (list[dict]): List of chunks to process.
            config (dict): Configuration for processing peripheral chunks.
            content_key (str): Key for the content in each chunk.
            order_key (str): Key for the order of each chunk.
            reverse (bool, optional): Whether to process chunks in reverse order. Defaults to False.

        Returns:
            list[str]: List of processed chunk strings.
        """
        if reverse:
            chunks = list(reversed(chunks))

        processed_parts = []
        included_chunks = []
        total_chunks = len(chunks)

        head_config = config.get("head", {})
        tail_config = config.get("tail", {})

        head_count = int(head_config.get("count", 0))
        tail_count = int(tail_config.get("count", 0))
        in_skip = False
        skip_char_count = 0

        for i, chunk in enumerate(chunks):
            if i < head_count:
                section = "head"
            elif i >= total_chunks - tail_count:
                section = "tail"
            elif "middle" in config:
                section = "middle"
            else:
                # Show number of characters skipped
                skipped_chars = len(chunk[content_key])
                if not in_skip:
                    skip_char_count = skipped_chars
                    in_skip = True
                else:
                    skip_char_count += skipped_chars

                continue

            if in_skip:
                processed_parts.append(
                    f"[... {skip_char_count} characters skipped ...]"
                )
                in_skip = False
                skip_char_count = 0

            section_config = config.get(section, {})
            section_content_key = section_config.get("content_key", content_key)

            is_summary = section_content_key != content_key
            summary_suffix = " (Summary)" if is_summary else ""

            chunk_prefix = f"[Chunk {chunk[order_key]}{summary_suffix}]"
            processed_parts.extend((chunk_prefix, f"{chunk[section_content_key]}"))
            included_chunks.append(chunk)

        if in_skip:
            processed_parts.append(f"[... {skip_char_count} characters skipped ...]")

        if reverse:
            processed_parts = list(reversed(processed_parts))

        return processed_parts

    def _retrieve_and_render(
        self,
        candidate_chunks: list[dict],
        retrieval_cfg: dict,
        main_chunk: dict,
        content_key: str,
        order_key: str,
    ) -> tuple[list[str], float]:
        """
        Retrieve top-k relevant chunks from candidates and render them as lines (position-agnostic).
        """
        if not candidate_chunks:
            return [], 0.0

        # Resolve query with the main chunk as context if templated
        query_template = retrieval_cfg.get("query", "")
        if "{{" in query_template and "}}" in query_template:
            query = strict_render(query_template, {"input": main_chunk})
        else:
            query = query_template

        # Build SampleOperation config
        method = retrieval_cfg.get("method", "embedding")
        k = retrieval_cfg.get("k")
        keys = retrieval_cfg.get("keys", [])
        sample_config = {
            "name": "gather_retrieve",
            "type": "sample",
            "method": "top_embedding" if method == "embedding" else "top_fts",
            "samples": k,
            "method_kwargs": {"keys": keys, "query": query},
        }

        if method == "embedding" and "embedding_model" in retrieval_cfg:
            sample_config["method_kwargs"][
                "embedding_model"
            ] = retrieval_cfg["embedding_model"]

        sample_op = SampleOperation(
            self.runner, sample_config, self.default_model, self.max_threads
        )
        retrieved, cost = sample_op.execute(candidate_chunks, is_build=False)

        render_key = retrieval_cfg.get("content_key", content_key)
        rendered_lines: list[str] = []
        op_name = sample_config["name"]
        rank_field = f"_{op_name}_rank"
        score_field = f"_{op_name}_score"
        for item in retrieved:
            rank_part = f" Rank {item.get(rank_field)}" if rank_field in item else ""
            score_part = (
                f" score={item.get(score_field):.4f}"
                if score_field in item and isinstance(item.get(score_field), (int, float))
                else ""
            )
            prefix = f"[Chunk {item.get(order_key, '?')} (Retrieved){rank_part}{score_part}]"
            rendered_lines.extend((prefix, f"{item.get(render_key, '')}"))

        return rendered_lines, cost

    def render_hierarchy_headers(
        self,
        current_chunk: dict,
        chunks: list[dict],
        doc_header_key: str,
    ) -> str:
        """
        Render headers for the current chunk's hierarchy.

        Args:
            current_chunk (dict): The current chunk being processed.
            chunks (list[dict]): List of chunks up to and including the current chunk.
            doc_header_key (str): The key for the headers in the current chunk.
        Returns:
            str: Renderted headers in the current chunk's hierarchy.
        """
        current_hierarchy = {}

        if doc_header_key is None:
            return ""

        # Find the largest/highest level in the current chunk
        current_chunk_headers = current_chunk.get(doc_header_key, [])

        # If there are no headers in the current chunk, return an empty string
        if not current_chunk_headers:
            return ""

        highest_level = float("inf")  # Initialize with positive infinity
        for header_info in current_chunk_headers:
            try:
                level = header_info.get("level")
                if level is not None and level < highest_level:
                    highest_level = level
            except Exception as e:
                self.runner.console.log(f"[red]Error processing header: {e}[/red]")
                self.runner.console.log(f"[red]Header: {header_info}[/red]")
                return ""

        # If no headers found in the current chunk, set highest_level to None
        if highest_level == float("inf"):
            highest_level = None

        for chunk in chunks:
            for header_info in chunk.get(doc_header_key, []):
                try:
                    header = header_info["header"]
                    level = header_info["level"]
                    if header and level:
                        current_hierarchy[level] = header
                    # Clear lower levels when a higher level header is found
                    for lower_level in range(level + 1, len(current_hierarchy) + 1):
                        if lower_level in current_hierarchy:
                            current_hierarchy[lower_level] = None
                except Exception as e:
                    self.runner.console.log(f"[red]Error processing header: {e}[/red]")
                    self.runner.console.log(f"[red]Header: {header_info}[/red]")
                    return ""

        rendered_headers = [
            f"{'#' * level} {header}"
            for level, header in sorted(current_hierarchy.items())
            if header is not None and (highest_level is None or level < highest_level)
        ]
        rendered_headers = " > ".join(rendered_headers)
        return f"_Current Section:_ {rendered_headers}" if rendered_headers else ""
