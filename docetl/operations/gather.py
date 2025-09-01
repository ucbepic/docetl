from typing import Any

from pydantic import field_validator

import numpy as np
from docetl.operations.base import BaseOperation
from docetl.operations.clustering_utils import get_embeddings_for_clustering


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
            # Validate general retrieval spec if provided
            if "general" in v and v["general"] is not None:
                general_cfg = v["general"]
                if not isinstance(general_cfg, dict):
                    raise TypeError("'peripheral_chunks.general' must be a dictionary")
                # Required fields
                for req in ["method", "k", "method_kwargs"]:
                    if req not in general_cfg:
                        raise ValueError(
                            f"Missing '{req}' in peripheral_chunks.general configuration"
                        )
                # Validate method
                if general_cfg["method"] not in {"embedding", "fts"}:
                    raise ValueError(
                        "peripheral_chunks.general.method must be 'embedding' or 'fts'"
                    )
                # Validate k
                k = general_cfg["k"]
                if not isinstance(k, (int, float)) or k <= 0:
                    raise ValueError(
                        "peripheral_chunks.general.k must be a positive number"
                    )
                # Validate method_kwargs
                mk = general_cfg["method_kwargs"]
                if not isinstance(mk, dict):
                    raise TypeError("peripheral_chunks.general.method_kwargs must be a dict")
                # Required for similarity computation
                if "keys_for_similarity" not in mk:
                    raise ValueError(
                        "Missing 'keys_for_similarity' in peripheral_chunks.general.method_kwargs"
                    )
                kfs = mk["keys_for_similarity"]
                if not isinstance(kfs, list) or not all(isinstance(x, str) for x in kfs):
                    raise TypeError(
                        "peripheral_chunks.general.method_kwargs.keys_for_similarity must be a list of strings"
                    )
                # Optional content_key and embedding_model types
                if "content_key" in general_cfg and not isinstance(general_cfg["content_key"], str):
                    raise TypeError(
                        "peripheral_chunks.general.content_key must be a string"
                    )
                if general_cfg["method"] == "embedding":
                    if "embedding_model" in mk and not isinstance(mk["embedding_model"], str):
                        raise TypeError(
                            "peripheral_chunks.general.method_kwargs.embedding_model must be a string"
                        )
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
        retrieval_config = {}
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
        # Retrieved "general" context (position-agnostic) - include before main
        if peripheral_config.get("general"):
            candidates_prev = chunks[:current_index]
            retrieved_lines_prev, c = self._retrieve_topk_similar(
                candidate_chunks=candidates_prev,
                retrieval_cfg=peripheral_config["general"],
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
        # Retrieved "general" context (position-agnostic) - include after main
        if peripheral_config.get("general"):
            candidates_next = chunks[current_index + 1 :]
            retrieved_lines_next, c = self._retrieve_topk_similar(
                candidate_chunks=candidates_next,
                retrieval_cfg=peripheral_config["general"],
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
        if peripheral_config.get("general"):
            general_candidates = chunks[:current_index] + chunks[current_index + 1 :]
            gen_lines, c = self._retrieve_topk_similar(
                candidate_chunks=general_candidates,
                retrieval_cfg=peripheral_config["general"],
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

    def _retrieve_topk_similar(
        self,
        candidate_chunks: list[dict],
        retrieval_cfg: dict,
        main_chunk: dict,
        content_key: str,
        order_key: str,
    ) -> tuple[list[str], float]:
        """
        Compute top-k similar chunks to the main chunk from candidates using
        either embedding similarity or FTS (BM25-like), without using SampleOperation.
        """
        if not candidate_chunks:
            return [], 0.0

        method = retrieval_cfg.get("method", "embedding")
        k_conf = retrieval_cfg.get("k", 1)
        method_kwargs = retrieval_cfg.get("method_kwargs", {})
        keys_for_similarity = method_kwargs.get("keys_for_similarity", [])
        render_key = retrieval_cfg.get("content_key", content_key)

        # Determine k number
        if isinstance(k_conf, float):
            k = max(1, int(k_conf * len(candidate_chunks)))
        else:
            k = min(int(k_conf), len(candidate_chunks))

        # Build text for candidates and main
        def join_keys(doc: dict) -> str:
            return " ".join(str(doc.get(key, "")) for key in keys_for_similarity)

        if method == "embedding":
            embedding_config = {
                "embedding_keys": keys_for_similarity,
                "embedding_model": method_kwargs.get("embedding_model", "text-embedding-3-small"),
            }
            # Get embeddings for candidates and main
            all_docs = candidate_chunks + [main_chunk]
            embeddings, cost = get_embeddings_for_clustering(
                all_docs, embedding_config, self.runner.api
            )
            embeddings = np.array(embeddings)
            cand_embeddings = embeddings[:-1]
            main_embedding = embeddings[-1]

            # Normalize and compute cosine similarity
            cand_norm = cand_embeddings / np.linalg.norm(cand_embeddings, axis=1, keepdims=True)
            main_norm = main_embedding / np.linalg.norm(main_embedding)
            scores = cand_norm.dot(main_norm)

            # Top-k indices
            top_idx = np.argsort(scores)[-k:][::-1]
            selected = [(candidate_chunks[i], float(scores[i])) for i in top_idx]
        else:
            # FTS BM25 with fallback
            try:
                from rank_bm25 import BM25Okapi
                import re

                def preprocess(text: str) -> list[str]:
                    text = text.lower()
                    text = re.sub(r"[^a-z0-9\s]", " ", text)
                    text = re.sub(r"\s+", " ", text).strip()
                    return text.split()

                tokenized_docs = [preprocess(join_keys(doc)) for doc in candidate_chunks]
                bm25 = BM25Okapi(tokenized_docs)
                main_tokens = preprocess(join_keys(main_chunk))
                scores = bm25.get_scores(main_tokens)
                top_idx = np.argsort(scores)[-k:][::-1]
                selected = [(candidate_chunks[i], float(scores[i])) for i in top_idx]
                cost = 0.0
            except ImportError:
                # TF-IDF cosine fallback
                import re
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity

                def normalize_text(text: str) -> str:
                    text = text.lower()
                    text = re.sub(r"\s+", " ", text)
                    text = re.sub(r"[^a-z0-9\s]", " ", text)
                    return text.strip()

                documents = [normalize_text(join_keys(doc)) for doc in candidate_chunks]
                main_text = normalize_text(join_keys(main_chunk))
                if not any(documents):
                    return [], 0.0
                try:
                    vec = TfidfVectorizer(
                        lowercase=True,
                        stop_words="english",
                        ngram_range=(1, 1),
                        max_features=10000,
                        token_pattern=r"\b[a-z0-9]+\b",
                        min_df=1,
                        max_df=0.95,
                    )
                    tfidf = vec.fit_transform(documents)
                    main_vec = vec.transform([main_text])
                    sims = cosine_similarity(main_vec, tfidf).flatten()
                    top_idx = np.argsort(sims)[-k:][::-1]
                    selected = [(candidate_chunks[i], float(sims[i])) for i in top_idx]
                except ValueError:
                    return [], 0.0
                cost = 0.0

        # Render
        lines: list[str] = []
        for rank, (item, score) in enumerate(selected, 1):
            prefix = f"[Chunk {item.get(order_key, '?')} (Retrieved) Rank {rank} score={score:.4f}]"
            lines.extend((prefix, f"{item.get(render_key, '')}"))

        return lines, cost

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
