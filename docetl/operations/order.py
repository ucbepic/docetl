"""
The `OrderOperation` class is a subclass of `BaseOperation` that performs an ordering operation on a dataset.
It uses a hybrid approach combining embedding similarity and LLM-based ranking to efficiently order documents
according to specified criteria, in either ascending or descending direction.
"""

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import Field
from rich.prompt import Confirm
from sklearn.metrics.pairwise import cosine_similarity

from docetl.operations.base import BaseOperation
from docetl.operations.utils import RichLoopBar, rich_as_completed
from docetl.utils import completion_cost


class OrderOperation(BaseOperation):
    class schema(BaseOperation.schema):
        type: str = "order"
        prompt: str
        input_keys: List[str]
        direction: Literal["asc", "desc"]
        model: Optional[str] = None
        embedding_model: Optional[str] = None
        num_grounding_examples: int = 5
        batch_size: int = 10
        chunk_size: int = 5
        timeout: Optional[int] = None
        verbose: bool = False
        litellm_completion_kwargs: Dict[str, Any] = Field(default_factory=dict)

    def syntax_check(self) -> None:
        """
        Checks the configuration of the OrderOperation for required keys and valid structure.

        This method performs the following checks:
        1. Verifies the presence of required keys: 'prompt', 'input_keys', and 'direction'.
        2. Ensures that 'input_keys' is a list of strings.
        3. Validates that 'direction' is either 'asc' or 'desc'.
        4. Optionally checks if 'model' and 'embedding_model' are strings (if present).
        5. Validates that numerical parameters (num_grounding_examples, batch_size, chunk_size) are positive integers.

        Raises:
            ValueError: If required keys are missing or values are invalid.
            TypeError: If the types of configuration values are incorrect.
        """
        required_keys = ["prompt", "input_keys", "direction"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(
                    f"Missing required key '{key}' in OrderOperation configuration"
                )

        # Check if input_keys is a list of strings
        if not isinstance(self.config["input_keys"], list):
            raise TypeError("'input_keys' must be a list")
        if not all(isinstance(key, str) for key in self.config["input_keys"]):
            raise TypeError("All items in 'input_keys' must be strings")

        # Check if direction is valid
        if self.config["direction"] not in ["asc", "desc"]:
            raise ValueError("'direction' must be either 'asc' or 'desc'")

        # Check if model is specified (optional)
        if "model" in self.config and not isinstance(self.config["model"], str):
            raise TypeError("'model' in configuration must be a string")

        # Check if embedding_model is specified (optional)
        if "embedding_model" in self.config and not isinstance(
            self.config["embedding_model"], str
        ):
            raise TypeError("'embedding_model' in configuration must be a string")

        # Check numerical parameters
        for param in ["num_grounding_examples", "batch_size", "chunk_size"]:
            if param in self.config:
                if not isinstance(self.config[param], int):
                    raise TypeError(f"'{param}' must be an integer")
                if self.config[param] <= 0:
                    raise ValueError(f"'{param}' must be a positive integer")

    def _extract_document_content(self, document: Dict, input_keys: List[str]) -> str:
        """
        Extracts relevant content from a document based on specified keys.

        Args:
            document (Dict): The document to extract content from.
            input_keys (List[str]): List of keys to extract from the document.

        Returns:
            str: The extracted content as a string.
        """
        content_parts = []
        for key in input_keys:
            if key in document:
                content_parts.append(f"{key}: {document[key]}")
        return "\n".join(content_parts)

    def _batch_rank_documents(
        self,
        batch: List[Dict],
        criteria: str,
        direction: str,
        model: str,
        timeout_seconds: int = 120,
    ) -> Tuple[List[int], float]:
        """
        Uses an LLM to rank a batch of documents according to the given criteria and direction.

        Args:
            batch (List[Dict]): A batch of documents to rank.
            criteria (str): The ranking criteria.
            direction (str): The direction of ordering ('asc' or 'desc').
            model (str): The LLM model to use.
            timeout_seconds (int): Timeout for the LLM call.

        Returns:
            Tuple[List[int], float]: A tuple containing the ranked indices and the cost.
        """
        # Construct the prompt
        document_texts = []
        for i, doc in enumerate(batch):
            content = self._extract_document_content(doc, self.config["input_keys"])
            document_texts.append(f"Document {i+1}:\n{content}")

        documents_text = "\n\n".join(document_texts)

        order_direction = (
            "ascending (from lowest to highest)"
            if direction == "asc"
            else "descending (from highest to lowest)"
        )

        prompt = f"""
        Your task is to order the following documents based on this criteria:

        {criteria}

        The ordering should be in {order_direction} order.

        Here are the documents:

        {documents_text}

        Return a numbered list of document numbers, ordered according to the criteria and direction,
        with the {'least' if direction == 'asc' else 'most'} matching document first.
        Only return the document numbers in order, separated by commas.
        For example: {'1, 2, 4, 5, 3' if direction == 'asc' else '3, 5, 4, 2, 1'}
        """

        # Call the LLM
        response = self.runner.api.call_llm(
            model,
            "rank",
            [{"role": "user", "content": prompt}],
            {"ranking": "string"},
            timeout_seconds=timeout_seconds,
            max_retries_per_timeout=self.config.get("max_retries_per_timeout", 2),
            bypass_cache=self.config.get("bypass_cache", False),
            litellm_completion_kwargs=self.config.get("litellm_completion_kwargs", {}),
            op_config=self.config,
        )

        # Parse the response
        try:
            output = self.runner.api.parse_llm_response(
                response.response,
                {"ranking": "string"},
            )[0]

            # Extract the ranking (document numbers, 1-indexed)
            ranking_text = output["ranking"].strip()
            # Handle various formatting issues
            for char in ["[", "]", "(", ")", "{", "}"]:
                ranking_text = ranking_text.replace(char, "")

            # Try to extract a list of integers
            try:
                # Split by commas and clean up
                ranking_parts = [part.strip() for part in ranking_text.split(",")]
                # Convert to integers and adjust to 0-indexed
                ranking = [int(part) - 1 for part in ranking_parts if part.isdigit()]

                # Check if we have the right number of documents
                if len(ranking) != len(batch):
                    # If not complete, fall back to the original order
                    self.console.log(
                        f"[yellow]Warning: LLM returned incomplete ranking ({len(ranking)} of {len(batch)} documents). Using embedding similarity as fallback.[/yellow]"
                    )
                    return None, response.total_cost

                # Check if the ranking contains valid indices
                if any(idx < 0 or idx >= len(batch) for idx in ranking):
                    self.console.log(
                        f"[yellow]Warning: LLM returned invalid document indices. Using embedding similarity as fallback.[/yellow]"
                    )
                    return None, response.total_cost

                return ranking, response.total_cost
            except Exception as e:
                self.console.log(
                    f"[yellow]Error parsing ranking: {str(e)}. Using embedding similarity as fallback.[/yellow]"
                )
                return None, response.total_cost

        except Exception as e:
            self.console.log(
                f"[yellow]Error parsing LLM response: {str(e)}. Using embedding similarity as fallback.[/yellow]"
            )
            return None, response.total_cost

    def _calculate_position_change(
        self, original_positions: List[int], new_positions: List[int]
    ) -> Dict[str, Any]:
        """
        Calculates statistics about how much document positions changed between orderings.

        Args:
            original_positions: List of document indices in original order
            new_positions: List of document indices in new order

        Returns:
            Dict with statistics about position changes
        """
        if len(original_positions) != len(new_positions):
            return {
                "average_position_change": "N/A - different lengths",
                "max_position_change": "N/A - different lengths",
                "position_changes": "N/A - different lengths",
                "percent_changed": "N/A - different lengths",
            }

        # Create mapping from document index to position in original order
        original_pos_map = {
            doc_idx: pos for pos, doc_idx in enumerate(original_positions)
        }

        position_changes = []
        total_change = 0
        max_change = 0
        num_changed = 0

        for new_pos, doc_idx in enumerate(new_positions):
            if doc_idx in original_pos_map:
                orig_pos = original_pos_map[doc_idx]
                change = abs(new_pos - orig_pos)
                position_changes.append((doc_idx, orig_pos, new_pos, change))
                total_change += change
                max_change = max(max_change, change)
                if change > 0:
                    num_changed += 1

        if not position_changes:
            return {
                "average_position_change": 0,
                "max_position_change": 0,
                "position_changes": [],
                "percent_changed": 0,
            }

        avg_change = total_change / len(position_changes)
        percent_changed = (num_changed / len(position_changes)) * 100

        return {
            "average_position_change": avg_change,
            "max_position_change": max_change,
            "position_changes": position_changes,
            "percent_changed": percent_changed,
        }

    def execute(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        """
        Executes the order operation on the provided dataset.

        Args:
            input_data (List[Dict]): The dataset to order.

        Returns:
            Tuple[List[Dict], float]: A tuple containing the ordered results and the total cost of the operation.

        This method implements a hybrid ordering approach:
        1. Generates embeddings for the ordering criteria and each document
        2. Uses embedding similarity as an initial ranking
        3. Performs LLM-based ranking on batches of documents
        4. Uses embedding-guided merging to produce the final ordering
        5. Applies the specified direction (asc/desc) to the final result
        """
        if len(input_data) == 0:
            return [], 0

        if len(input_data) == 1:
            return input_data, 0

        criteria = self.config["prompt"]
        input_keys = self.config["input_keys"]
        direction = self.config["direction"]
        model = self.config.get("model", self.default_model)
        embedding_model = self.config.get("embedding_model", "text-embedding-3-small")
        batch_size = self.config.get("batch_size", 10)
        chunk_size = self.config.get("chunk_size", 5)
        num_grounding_examples = self.config.get("num_grounding_examples", 5)
        verbose = self.config.get("verbose", False)

        if self.status:
            self.status.stop()

        total_cost = 0

        # Warn if ordering a large number of documents
        if len(input_data) > 100:
            if not Confirm.ask(
                f"[yellow]Warning: Ordering {len(input_data)} documents may be expensive. "
                f"Do you want to continue?[/yellow]",
                console=self.runner.console,
            ):
                raise ValueError("Operation cancelled by user.")

        # Step 1: Generate embeddings for criteria and documents
        self.console.log("Generating embeddings for criteria and documents...")

        # Generate embedding for criteria
        criteria_embedding_response = self.runner.api.gen_embedding(
            model=embedding_model, input=[criteria]
        )
        criteria_embedding = criteria_embedding_response["data"][0]["embedding"]
        total_cost += completion_cost(criteria_embedding_response)

        # Generate embeddings for documents
        document_contents = [
            self._extract_document_content(doc, input_keys) for doc in input_data
        ]

        # Generate embeddings in batches to avoid API limits
        document_embeddings = []
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            embedding_batch_size = 100  # Batch size for embedding API calls
            embedding_futures = []

            for i in range(0, len(document_contents), embedding_batch_size):
                batch = document_contents[i : i + embedding_batch_size]
                embedding_futures.append(
                    executor.submit(
                        self.runner.api.gen_embedding,
                        model=embedding_model,
                        input=batch,
                    )
                )

            for future in rich_as_completed(
                embedding_futures,
                total=len(embedding_futures),
                desc="Generating document embeddings",
                console=self.console,
            ):
                response = future.result()
                for data in response["data"]:
                    document_embeddings.append(data["embedding"])
                total_cost += completion_cost(response)

        # Step 2: Calculate similarity scores and sort documents
        similarities = cosine_similarity([criteria_embedding], document_embeddings)[0]

        # Create document indices sorted by similarity
        # For ascending order, we want the least similar documents first
        # For descending order, we want the most similar documents first
        reverse_order = direction == "desc"
        similarity_indices = list(range(len(input_data)))
        similarity_indices.sort(key=lambda i: similarities[i], reverse=reverse_order)

        # Store original order for comparison
        original_order = similarity_indices.copy()

        # Only log if verbose is enabled
        if verbose:
            self.console.log(
                "[bold green]Initial embedding-based ordering created.[/bold green]"
            )

        # Step 3: Distribute documents into batches for LLM ranking
        num_batches = math.ceil(len(input_data) / batch_size)
        batches = []

        # Create batches with a mix of documents
        for i in range(batch_size):
            for j in range(i, len(similarity_indices), batch_size):
                if j < len(similarity_indices):
                    if len(batches) <= j // batch_size:
                        batches.append([])
                    batches[j // batch_size].append(similarity_indices[j])

        # Make sure all documents are included in batches
        all_indices = set()
        for batch in batches:
            all_indices.update(batch)

        missing_indices = set(range(len(input_data))) - all_indices
        for idx in missing_indices:
            # Add missing indices to the smallest batch
            smallest_batch = min(batches, key=len)
            smallest_batch.append(idx)

        # Step 4: Perform LLM ranking on each batch
        batch_rankings = []

        pbar = RichLoopBar(
            range(len(batches)),
            desc="Ranking batches with LLM",
            console=self.console,
        )

        for i in pbar:
            batch_indices = batches[i]
            batch_docs = [input_data[idx] for idx in batch_indices]

            # Use LLM to rank the batch
            ranking, cost = self._batch_rank_documents(
                batch_docs,
                criteria,
                direction,
                model,
                timeout_seconds=self.config.get("timeout", 120),
            )
            total_cost += cost

            if ranking is None:
                # Fall back to similarity ranking if LLM ranking failed
                batch_similarities = [similarities[idx] for idx in batch_indices]
                ranking_by_similarity = sorted(
                    range(len(batch_indices)),
                    key=lambda i: batch_similarities[i],
                    reverse=reverse_order,
                )
                batch_rankings.append([batch_indices[i] for i in ranking_by_similarity])
            else:
                # Map the LLM ranking back to the original indices
                batch_rankings.append([batch_indices[i] for i in ranking])

        # Prepare flattened LLM ranking for stats calculation and logging
        flattened_llm_ranking = []
        for batch_rank in batch_rankings:
            flattened_llm_ranking.extend(batch_rank)

        # Remove duplicates while preserving order
        seen = set()
        llm_ranking = []
        for idx in flattened_llm_ranking:
            if idx not in seen:
                llm_ranking.append(idx)
                seen.add(idx)

        # Add any missing indices
        for idx in range(len(input_data)):
            if idx not in seen:
                llm_ranking.append(idx)

        # Calculate changes from embedding to LLM ranking (always calculate for possible future use)
        llm_changes = self._calculate_position_change(original_order, llm_ranking)

        # Only log if verbose is enabled
        if verbose:
            self.console.log(
                "[bold green]LLM-based batch ranking completed.[/bold green]"
            )
            self.console.log(
                f"Position changes: Avg={llm_changes['average_position_change']:.2f}, Max={llm_changes['max_position_change']}, Changed={llm_changes['percent_changed']:.2f}%"
            )

        # Step 5: Create initial global ranking based on embedding similarity
        initial_global_ranking = similarity_indices.copy()

        # Step 6: Split into smaller chunks for LLM verification
        chunks = [
            initial_global_ranking[i : i + chunk_size]
            for i in range(0, len(initial_global_ranking), chunk_size)
        ]

        final_ranking = []

        # For small datasets, we can skip this step
        if len(input_data) <= num_grounding_examples * 2:
            # Just use the batch rankings to create the final ranking
            # Flatten batch rankings and remove duplicates while preserving order
            seen = set()
            for rank in batch_rankings:
                for idx in rank:
                    if idx not in seen:
                        final_ranking.append(idx)
                        seen.add(idx)

            # Add any missing indices
            for idx in range(len(input_data)):
                if idx not in seen:
                    final_ranking.append(idx)
        else:
            # For larger datasets, verify and merge chunks
            pbar = RichLoopBar(
                range(len(chunks)),
                desc="Verifying chunk ordering with LLM",
                console=self.console,
            )

            for i in pbar:
                chunk_indices = chunks[i]
                chunk_docs = [input_data[idx] for idx in chunk_indices]

                # For the first chunk, include some grounding examples from the batch rankings
                if i == 0 and num_grounding_examples > 0:
                    grounding_indices = []
                    for rank in batch_rankings:
                        grounding_indices.extend(rank[:num_grounding_examples])
                        if len(grounding_indices) >= num_grounding_examples:
                            break

                    # Remove duplicates while preserving order
                    seen = set()
                    unique_grounding = []
                    for idx in grounding_indices:
                        if idx not in seen and idx not in chunk_indices:
                            unique_grounding.append(idx)
                            seen.add(idx)
                    grounding_indices = unique_grounding[:num_grounding_examples]

                    grounding_docs = [input_data[idx] for idx in grounding_indices]
                    verification_docs = grounding_docs + chunk_docs

                    # Use LLM to verify/correct the ordering
                    ranking, cost = self._batch_rank_documents(
                        verification_docs,
                        criteria,
                        direction,
                        model,
                        timeout_seconds=self.config.get("timeout", 120),
                    )
                    total_cost += cost

                    if ranking is None:
                        # Fall back to the original chunk order
                        chunk_verified = chunk_indices
                    else:
                        # Extract only the chunk part (after grounding examples)
                        ranking_without_grounding = [
                            idx - len(grounding_indices)
                            for idx in ranking
                            if idx >= len(grounding_indices)
                            and idx < len(verification_docs)
                        ]
                        chunk_verified = [
                            chunk_indices[idx] for idx in ranking_without_grounding
                        ]
                else:
                    # For subsequent chunks, use LLM to verify the ordering
                    ranking, cost = self._batch_rank_documents(
                        chunk_docs,
                        criteria,
                        direction,
                        model,
                        timeout_seconds=self.config.get("timeout", 120),
                    )
                    total_cost += cost

                    if ranking is None:
                        # Fall back to the original chunk order
                        chunk_verified = chunk_indices
                    else:
                        # Map back to original indices
                        chunk_verified = [chunk_indices[idx] for idx in ranking]

                # Add verified chunk to final ranking
                final_ranking.extend(chunk_verified)

        # Check if all indices are in the final ranking
        included_indices = set(final_ranking)
        all_indices = set(range(len(input_data)))
        missing_indices = all_indices - included_indices

        # Add any missing indices to the end
        final_ranking.extend(missing_indices)

        # Calculate changes from LLM ranking to final ranking
        final_changes = self._calculate_position_change(llm_ranking, final_ranking)

        # Calculate total changes from original to final
        total_changes = self._calculate_position_change(original_order, final_ranking)

        # Only log if verbose is enabled
        if verbose:
            self.console.log("[bold green]Final ranking complete.[/bold green]")
            self.console.log(
                f"Final position changes: Avg={total_changes['average_position_change']:.2f}, Max={total_changes['max_position_change']}, Changed={total_changes['percent_changed']:.2f}%"
            )

        # Step 7: Reorder the input data based on the final ranking
        result = [input_data[idx] for idx in final_ranking]

        # Add rank information to each document
        for i, item in enumerate(result):
            item["_rank"] = i + 1

        if self.status:
            self.status.start()

        return result, total_cost
