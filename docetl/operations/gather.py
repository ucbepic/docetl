from typing import Any

from jinja2.exceptions import UndefinedError
from pydantic import field_validator

from docetl.operations.base import BaseOperation, Cardinality
from docetl.operations.utils.validation import lookup_field


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
        doc_header_key: str | None = None
        main_chunk_start: str | None = None
        main_chunk_end: str | None = None

        @field_validator("peripheral_chunks")
        def validate_peripheral_chunks(cls, v):
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

    @classmethod
    def transform_schema(cls, schema, config):
        result = super().transform_schema(schema, config)
        if config.get("content_key"):
            result[f"{config['content_key']}_rendered"] = "string"
        return result

    # ── plan traits ────────────────────────────────────────────────
    # Not row-local (each chunk's rendering reads neighboring chunks)
    # and not order-preserving (output is regrouped by document).

    @classmethod
    def cardinality(cls, config):
        return Cardinality.ONE_TO_ONE

    @classmethod
    def fields_read(cls, config):
        if not config.get("content_key"):
            return None
        fields = {config["content_key"]}
        for key in ("doc_id_key", "order_key", "doc_header_key"):
            if config.get(key):
                fields.add(config[key])
        return frozenset(fields)

    @classmethod
    def fields_written(cls, config):
        if not config.get("content_key"):
            return None
        return frozenset({f"{config['content_key']}_rendered"})

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
            doc_id = lookup_field(item, doc_id_key)
            if doc_id not in grouped_chunks:
                grouped_chunks[doc_id] = []
            grouped_chunks[doc_id].append(item)

        # Process each group of chunks
        for chunks in grouped_chunks.values():
            # Sort chunks by their order within the document
            chunks.sort(key=lambda x: x[order_key])
            hierarchy = {}

            # Process each chunk with its peripheral context and headers
            for i, chunk in enumerate(chunks):
                hierarchy, header_context = self._advance_header_hierarchy(
                    hierarchy, chunk, doc_header_key
                )
                rendered_chunk = self.render_chunk_with_context(
                    chunks,
                    i,
                    peripheral_config,
                    content_key,
                    order_key,
                    main_chunk_start,
                    main_chunk_end,
                    doc_header_key,
                    header_context=header_context,
                )

                result = chunk.copy()
                result[f"{content_key}_rendered"] = rendered_chunk
                results.append(result)

        return results, cost

    def render_chunk_with_context(
        self,
        chunks: list[dict],
        current_index: int,
        peripheral_config: dict,
        content_key: str,
        order_key: str,
        main_chunk_start: str,
        main_chunk_end: str,
        doc_header_key: str,
        header_context: str | None = None,
    ) -> str:
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
            header_context (str, optional): Precomputed inherited headers for this chunk.

        Returns:
            str: Renderted chunk with context and headers.
        """

        if header_context is None:
            header_context = self.render_hierarchy_headers(
                chunks[current_index], chunks[: current_index + 1], doc_header_key
            )

        # Header context is independent of whether neighboring text is requested.
        if not peripheral_config:
            content = chunks[current_index][content_key]
            return f"{header_context}\n{content}" if header_context else content

        combined_parts = ["--- Previous Context ---"]

        combined_parts.extend(
            self.process_peripheral_chunks(
                chunks[:current_index],
                peripheral_config.get("previous", {}),
                content_key,
                order_key,
            )
        )
        combined_parts.append("--- End Previous Context ---\n")

        # Process main chunk
        main_chunk = chunks[current_index]
        if header_context:
            combined_parts.append(header_context)
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
        combined_parts.append("--- End Next Context ---")

        return "\n".join(combined_parts)

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
        hierarchy = {}
        prefix = chunks[:-1] if chunks and chunks[-1] is current_chunk else chunks
        for chunk in prefix:
            hierarchy, _ = self._advance_header_hierarchy(
                hierarchy, chunk, doc_header_key
            )
        return self._advance_header_hierarchy(hierarchy, current_chunk, doc_header_key)[
            1
        ]

    def _advance_header_hierarchy(
        self, hierarchy: dict[int, str], chunk: dict, doc_header_key: str | None
    ) -> tuple[dict[int, str], str]:
        """Render inherited ancestors, then atomically apply this chunk's headings."""
        if not doc_header_key:
            return hierarchy, ""
        try:
            headings = []
            headers = lookup_field(chunk, doc_header_key)
            if headers is None:
                headers = []
            if not isinstance(headers, (list, tuple)):
                raise TypeError("Headers must be a list of heading entries")
            for entry in headers:
                level, title = entry["level"], entry["header"]
                if not isinstance(level, int) or isinstance(level, bool) or level < 1:
                    raise ValueError("Header levels must be positive integers")
                if not isinstance(title, str):
                    raise TypeError("Header text must be a string")
                if title:
                    headings.append((level, title))

            highest_level = min((level for level, _ in headings), default=None)
            ancestors = " > ".join(
                f"{'#' * level} {title}"
                for level, title in sorted(hierarchy.items())
                if highest_level is None or level < highest_level
            )
            updated = dict(hierarchy)
            for level, title in headings:
                updated = {
                    depth: name for depth, name in updated.items() if depth < level
                }
                updated[level] = title
            context = f"_Current Section:_ {ancestors}" if ancestors else ""
            return updated, context
        except (KeyError, TypeError, ValueError, UndefinedError) as error:
            self.console.log(f"[red]Error processing headers: {error}[/red]")
            # A malformed heading can change the section; do not inherit stale ancestry.
            return {}, ""
