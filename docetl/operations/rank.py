import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from pydantic import Field

from docetl.operations.base import BaseOperation
from docetl.operations.utils import rich_as_completed
from docetl.operations.utils.progress import RichLoopBar
from docetl.utils import completion_cost


class RankOperation(BaseOperation):
    class schema(BaseOperation.schema):
        type: str = "order"
        prompt: str
        input_keys: list[str] = Field(default_factory=list)
        direction: Literal["asc", "desc"]
        model: str | None = None
        embedding_model: str | None = None
        batch_size: int = Field(10, gt=0)
        initial_ordering_method: Literal[
            "embedding", "likert", "calibrated_embedding"
        ] = "embedding"
        k: int | None = Field(None, gt=0)
        rerank_call_budget: int = Field(100, gt=0)
        num_top_items_per_window: int = Field(3, gt=0)
        overlap_fraction: float = Field(0.5, ge=0, le=1)
        timeout: int | None = Field(None, gt=0)
        num_calibration_docs: int = Field(10, gt=0)
        verbose: bool = False
        litellm_completion_kwargs: dict[str, Any] = Field(default_factory=dict)

    def _extract_document_content(self, document: dict, input_keys: list[str]) -> str:
        """
        Extracts relevant content from a document based on specified keys.

        Args:
            document (dict): The document to extract content from.
            input_keys (list[str]): List of keys to extract from the document.

        Returns:
            str: The extracted content as a string.
        """
        if not input_keys:
            input_keys = document.keys()
        content_parts = []
        for key in input_keys:
            if key in document:
                content_parts.append(f"{key}: {document[key]}")
        return "\n".join(content_parts)

    def _batch_rank_documents(
        self,
        batch: list[dict],
        criteria: str,
        direction: str,
        model: str,
        timeout_seconds: int = 120,
        batch_label: str = "Batch",  # Added parameter for identifying which batch is being processed
    ) -> tuple[list[int], float]:
        """
        Uses an LLM to rank a batch of documents according to the given criteria and direction.

        Args:
            batch (list[dict]): A batch of documents to rank.
            criteria (str): The ranking criteria.
            direction (str): The direction of ordering ('asc' or 'desc').
            model (str): The LLM model to use.
            timeout_seconds (int): Timeout for the LLM call.
            batch_label (str): Label to identify which batch is being processed in logs.

        Returns:
            tuple[list[int], float]: A tuple containing the ranked indices and the cost.
        """
        # Construct the prompt
        document_texts = []
        for i, doc in enumerate(batch):
            document_texts.append(
                f"Document {i+1}:\n{self._extract_document_content(doc, self.config['input_keys'])}"
            )

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
        Only return the document numbers in order, as a list.
        For example: {'1, 2, 4, 5, 3' if direction == 'asc' else '3, 5, 4, 2, 1'}
        """

        # Call the LLM
        response = self.runner.api.call_llm(
            model,
            "rank",
            [{"role": "user", "content": prompt}],
            {"ranking": "list[int]"},
            timeout_seconds=timeout_seconds,
            max_retries_per_timeout=self.config.get("max_retries_per_timeout", 2),
            bypass_cache=self.config.get("bypass_cache", self.bypass_cache),
            litellm_completion_kwargs=self.config.get("litellm_completion_kwargs", {}),
            op_config=self.config,
        )

        # Parse the response
        try:
            output = self.runner.api.parse_llm_response(
                response.response,
                {"ranking": "list[int]"},
            )[0]

            # Extract the ranking (document numbers, 1-indexed)
            try:
                # Convert to 0-indexed
                ranking = [num - 1 for num in output["ranking"]]

                # See which indexes are missing and add them to the end of the ranking
                # Deduplicate the ranking
                deduped_ranking = []
                for num in ranking:
                    if num not in deduped_ranking:
                        deduped_ranking.append(num)
                ranking = deduped_ranking

                # Check if we have the right number of documents
                if len(ranking) != len(batch):
                    # If not complete, fall back to the original order
                    self.console.log(
                        f"[yellow]Warning: LLM returned incomplete ranking for {batch_label} ({len(ranking)} of {len(batch)} documents).[/yellow]"
                    )
                    missing_indexes = set(range(len(batch))) - set(ranking)
                    ranking.extend(list(missing_indexes))

                # Check if the ranking contains valid indices
                if any(idx < 0 or idx >= len(batch) for idx in ranking):
                    self.console.log(
                        f"[yellow]Warning: LLM returned invalid document indices for {batch_label}. Using embedding similarity as fallback.[/yellow]"
                    )
                    return None, response.total_cost

                return ranking, response.total_cost
            except Exception as e:
                self.console.log(
                    f"[yellow]Error parsing ranking for {batch_label}: {str(e)}. Using embedding similarity as fallback.[/yellow]"
                )
                return None, response.total_cost

        except Exception as e:
            self.console.log(
                f"[yellow]Error parsing LLM response for {batch_label}: {str(e)}. Using embedding similarity as fallback.[/yellow]"
            )
            return None, response.total_cost

    def _execute_comparison_qurk(
        self, input_data: list[dict], sample: bool = False
    ) -> tuple[list[dict], float]:
        """
        Implements the comparison-based approach from the human-powered sort paper.
        Uses random batches of S items and head-to-head counting to break ties.

        Args:
            input_data (list[dict]): The dataset to order.

        Returns:
            tuple[list[dict], float]: A tuple containing the ordered results and the total cost.
        """
        if len(input_data) <= 1:
            return input_data, 0

        criteria = self.config["prompt"]
        input_keys = self.config["input_keys"]
        direction = self.config["direction"].lower()
        model = self.config.get("model", self.default_model)
        batch_size = self.config.get("batch_size", 10)  # S in the paper

        [self._extract_document_content(doc, input_keys) for doc in input_data]

        # Initialize head-to-head win counts
        win_counts = {idx: 0 for idx in range(len(input_data))}

        # Determine number of batches based on N*(N-1)/(S*(S-1))
        total_comparisons = len(input_data) * (len(input_data) - 1)
        batch_comparisons = batch_size * (batch_size - 1)
        num_batches = max(
            1, (total_comparisons // batch_comparisons)
        )  # Reduced from paper for cost

        total_cost = 0
        random.seed(42)

        # Process random batches using ThreadPoolExecutor for parallel execution
        self.console.log(
            f"[bold]Processing {num_batches} random comparison batches in parallel[/bold]"
        )

        # Function to process a single batch
        def process_batch(batch_num: int) -> tuple[dict[int, int], float]:
            # Select random batch of documents
            batch_indices = random.sample(
                range(len(input_data)), min(batch_size, len(input_data))
            )
            batch_docs = [input_data[idx] for idx in batch_indices]

            # Use LLM to rank the batch
            ranking, cost = self._batch_rank_documents(
                batch_docs,
                criteria,
                direction,
                model,
                timeout_seconds=self.config.get("timeout", 120),
                batch_label=f"Batch {batch_num+1}/{num_batches}",
            )

            # Initialize local win counts
            local_win_counts = {idx: 0 for idx in range(len(input_data))}

            if ranking is not None:
                # Update win counts based on pairwise comparisons in this batch
                for i in range(len(ranking)):
                    for j in range(i + 1, len(ranking)):
                        local_win_counts[batch_indices[ranking[i]]] += 1

            return local_win_counts, cost

        # Use ThreadPoolExecutor to process batches in parallel
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            # Submit all batch processing tasks
            futures = []
            for batch_num in range(num_batches):
                future = executor.submit(process_batch, batch_num)
                futures.append(future)

            # Process results as they complete
            for future in rich_as_completed(
                futures,
                total=len(futures),
                desc="Processing comparison batches",
                console=self.console,
            ):
                try:
                    local_win_counts, cost = future.result()
                    # Merge local win counts into global win counts
                    for idx, count in local_win_counts.items():
                        win_counts[idx] += count
                    total_cost += cost
                except Exception as e:
                    self.console.log(f"[red]Error in batch processing: {str(e)}[/red]")
                    raise e

        # Create final ranking based on win counts
        final_ranking = sorted(
            range(len(input_data)),
            key=lambda idx: win_counts[idx],
            reverse=True,  # Higher win count = higher rank
        )

        # Reorder the input data based on the final ranking
        result = [input_data[idx] for idx in final_ranking]

        # Add rank information to each document
        results_with_rank = []
        for i, item in enumerate(result):
            item["_rank"] = i + 1
            results_with_rank.append(item.copy())

        return results_with_rank, total_cost

    def _execute_rating_embedding_qurk(
        self, input_data: list[dict]
    ) -> tuple[list[dict], float]:
        """
        Implements the rating-based approach from the human-powered sort paper.
        Each document is rated independently and then sorted by the average rating.

        Args:
            input_data (list[dict]): The dataset to order.

        Returns:
            tuple[list[dict], float]: A tuple containing the ordered results and the total cost.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        if len(input_data) <= 1:
            return input_data, 0

        criteria = self.config["prompt"]
        input_keys = self.config["input_keys"]
        direction = self.config["direction"].lower()
        self.config.get("model", self.default_model)
        embedding_model = self.config.get("embedding_model", "text-embedding-3-small")

        total_cost = 0
        ratings = {}

        # Calculate similarity to criteria as initial ratings
        # (Substituting embeddings for human ratings)
        criteria_embedding_response = self.runner.api.gen_embedding(
            model=embedding_model, input=[criteria]
        )
        criteria_embedding = criteria_embedding_response["data"][0]["embedding"]
        total_cost += completion_cost(criteria_embedding_response)

        document_contents = [
            self._extract_document_content(doc, input_keys) for doc in input_data
        ]

        # Process documents in batches of 1000
        document_embeddings = []
        batch_size = 1000

        for i in range(0, len(document_contents), batch_size):
            batch = document_contents[i : i + batch_size]
            batch_embeddings_response = self.runner.api.gen_embedding(
                model=embedding_model, input=batch
            )
            total_cost += completion_cost(batch_embeddings_response)

            # Extract embeddings from the batch response
            batch_embeddings = [
                data["embedding"] for data in batch_embeddings_response["data"]
            ]
            document_embeddings.extend(batch_embeddings)

        # Calculate similarity scores
        similarities = cosine_similarity([criteria_embedding], document_embeddings)[0]

        # Assign ratings
        for idx, similarity in enumerate(similarities):
            ratings[idx] = similarity

        # Create final ranking based on ratings
        reverse_order = direction == "desc"
        final_ranking = sorted(
            range(len(input_data)), key=lambda idx: ratings[idx], reverse=reverse_order
        )

        # Reorder the input data based on the final ranking
        result = [input_data[idx] for idx in final_ranking]

        # Add rank information to each document
        results_with_rank = []
        for i, item in enumerate(result):
            item["_rank"] = i + 1
            results_with_rank.append(item.copy())

        return results_with_rank, total_cost

    def _execute_sliding_window_qurk(
        self,
        input_data: list[dict],
        initial_ordering_method: str = "embedding",
        k: int | None = None,
    ) -> tuple[list[dict], float]:
        """
        Implements the sliding window approach from the human-powered sort paper.
        Starts with an initial ordering based on the specified method, then applies successive windows for reranking.

        Args:
            input_data (list[dict]): The dataset to order.
            initial_ordering_method (str): Method to use for initial ordering ("embedding" or "likert").
            k (int | None): Number of top elements to focus on. If None, set to len(input_data).

        Returns:
            tuple[list[dict], float]: A tuple containing the ordered results and the total cost.
        """
        if len(input_data) <= 1:
            return input_data, 0

        # If k is None, set it to the length of input_data
        if k is None:
            k = len(input_data)
        else:
            k = min(k, len(input_data))  # Ensure k doesn't exceed input_data length

        # Validate initial ordering method
        if initial_ordering_method not in ["embedding", "likert"]:
            raise ValueError(
                "initial_ordering_method must be either 'embedding' or 'likert'"
            )

        criteria = self.config["prompt"]
        input_keys = self.config["input_keys"]
        direction = self.config["direction"].lower()
        model = self.config.get("model", self.default_model)
        window_size = self.config.get("batch_size", 10)  # S in the paper
        window_step = window_size // 2

        total_cost = 0
        [self._extract_document_content(doc, input_keys) for doc in input_data]

        # Get initial ordering using the specified method
        if self.config.get("verbose", False):
            self.console.log(
                f"[bold blue]Step 1: Initial Ordering using {initial_ordering_method} method[/bold blue]"
            )

        if initial_ordering_method == "embedding":
            # Use embedding-based rating for initial ordering
            initial_results, initial_cost = self._execute_rating_embedding_qurk(
                input_data
            )
            total_cost += initial_cost
        else:  # likert
            # Use Likert scale rating for initial ordering
            initial_results, initial_cost = self._execute_likert_rating_qurk(input_data)
            total_cost += initial_cost

        # If we couldn't match the documents, fall back to their indices
        if len(initial_results) != len(input_data):
            self.console.log(
                "[yellow]Warning: Couldn't match all documents from initial ordering. Using indices as fallback.[/yellow]"
            )
            initial_results = input_data

        if self.config.get("verbose", False):
            self.console.log(
                f"[bold blue]Step 2: Applying Sliding Window Refinement (focusing on top {k} elements)[/bold blue]"
            )

        # Apply sliding windows for refinement
        max_windows = len(input_data) // window_step
        if self.config.get("verbose", False):
            self.console.log(
                f"Applying sliding windows with size {window_size} and step {window_step} until top {k} elements are processed"
            )

        current_results = initial_results.copy()

        for window_num in range(max_windows):
            # Calculate window start position
            start_pos = (window_num * window_step) % len(current_results)
            end_pos = min(start_pos + window_size, len(current_results))

            # Extract window indices and documents
            window_indices = list(range(start_pos, end_pos))
            window_docs = [current_results[idx] for idx in window_indices]

            # Rerank window using LLM
            new_ranking, cost = self._batch_rank_documents(
                window_docs,
                criteria,
                direction,
                model,
                timeout_seconds=self.config.get("timeout", 120),
                batch_label=f"Window {window_num+1}/{max_windows}",
            )
            total_cost += cost

            if new_ranking is not None:
                # Apply new ordering to the window
                reordered_window = [window_indices[i] for i in new_ranking]

                # Update the current ranking
                current_results[start_pos:end_pos] = [
                    current_results[i] for i in reordered_window
                ]

                if self.config.get("verbose", False):
                    self.console.log(
                        f"Applied reordering to window {start_pos}-{end_pos}"
                    )

            # Check if we've seen enough elements
            if start_pos >= k:
                self.console.log(
                    f"All top {k} elements have been processed in windows. Stopping."
                )
                break

        # Final result
        result = current_results

        # Add rank information to each document
        results_with_rank = []
        for i, item in enumerate(result):
            item_copy = item.copy()
            item_copy["_rank"] = i + 1
            results_with_rank.append(item_copy)

        return results_with_rank, total_cost

    def _execute_likert_rating_qurk(
        self, input_data: list[dict]
    ) -> tuple[list[dict], float]:
        """
        Implements a rating-based approach using a 7-point Likert scale with LLM ratings.
        Each document is rated by the LLM on a 1-7 scale and then sorted by the rating.
        Uses a thread pool to parallelize LLM calls for better performance.

        Args:
            input_data (list[dict]): The dataset to order.

        Returns:
            tuple[list[dict], float]: A tuple containing the ordered results and the total cost.
        """
        if len(input_data) <= 1:
            return input_data, 0

        criteria = self.config["prompt"]
        input_keys = self.config["input_keys"]
        direction = self.config["direction"].lower()
        model = self.config.get("model", self.default_model)
        batch_size = self.config.get(
            "batch_size", 10
        )  # Rate more documents per batch for efficiency
        num_calibration_docs = self.config.get("num_calibration_docs", 10)
        max_workers = self.max_threads

        total_cost = 0
        ratings = {}

        # Select a random sample of 10 documents to provide context
        random.seed(42)
        context_size = min(num_calibration_docs, len(input_data))
        context_indices = random.sample(range(len(input_data)), context_size)
        context_docs = [input_data[idx] for idx in context_indices]

        # Create context text
        context_texts = []
        for i, doc in enumerate(context_docs):
            content = self._extract_document_content(doc, input_keys)
            context_texts.append(f"Example {i+1}:\n{content}")

        context_text = "\n\n".join(context_texts)

        # Function to process a batch of documents
        def process_batch(batch_indices):
            batch_docs = [input_data[idx] for idx in batch_indices]

            # Construct document texts
            document_texts = []
            for j, doc in enumerate(batch_docs):
                content = self._extract_document_content(doc, input_keys)
                document_texts.append(f"Document {j+1}:\n{content}")

            documents_text = "\n\n".join(document_texts)

            # Construct the prompt for Likert scale rating
            prompt = f"""
            Your task is to rate each document based on this criteria:

            {criteria}

            Rate each document on a 7-point Likert scale where:
            1 = Strongly disagree that the document matches the criteria
            2 = Disagree
            3 = Somewhat disagree
            4 = Neither agree nor disagree
            5 = Somewhat agree
            6 = Agree
            7 = Strongly agree that the document matches the criteria

            For context, here are some example documents from the dataset:
            {context_text}

            Now, please rate the following documents:
            {documents_text}

            Provide an integer rating from 1-7 for each document, in the same order as presented.
            Your response should be a list of {len(batch_docs)} integers.
            """

            # Call the LLM with list[int] output format
            response = self.runner.api.call_llm(
                model,
                "rate",
                [{"role": "user", "content": prompt}],
                {"ratings": "list[int]"},  # Specify output as list of integers
                timeout_seconds=self.config.get("timeout", 120),
                max_retries_per_timeout=self.config.get("max_retries_per_timeout", 2),
                bypass_cache=self.config.get("bypass_cache", self.bypass_cache),
                litellm_completion_kwargs=self.config.get(
                    "litellm_completion_kwargs", {}
                ),
                op_config=self.config,
            )

            batch_result = {
                "cost": response.total_cost,
                "ratings": [],
                "indices": batch_indices,
            }

            # Parse the response
            try:
                output = self.runner.api.parse_llm_response(
                    response.response,
                    {"ratings": "list[int]"},
                )[0]

                batch_ratings = output["ratings"]

                # Validate ratings
                for j, rating in enumerate(batch_ratings):
                    if j < len(batch_docs) and 1 <= rating <= 7:
                        batch_result["ratings"].append((batch_indices[j], rating))

            except Exception as e:
                self.console.log(
                    f"[yellow]Error parsing ratings response: {str(e)}[/yellow]"
                )

            return batch_result

        # Create batches
        batches = []
        for i in range(0, len(input_data), batch_size):
            batch_indices = list(range(i, min(i + batch_size, len(input_data))))
            batches.append(batch_indices)

        # Process batches in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]

            pbar = RichLoopBar(
                range(len(futures)),
                desc=f"Processing {self.config['name']} (first coarse pass of likert rating)",
                console=self.console,
            )

            # Collect results as they complete
            for i in pbar:
                future = futures[i]
                result = future.result()
                total_cost += result["cost"]

                # Store ratings
                for idx, rating in result["ratings"]:
                    ratings[idx] = rating

        # Handle any documents that weren't rated
        for idx in range(len(input_data)):
            if idx not in ratings:
                # Assign a neutral rating
                ratings[idx] = 4

        # Create final ranking based on ratings
        reverse_order = direction == "desc"
        final_ranking = sorted(
            range(len(input_data)), key=lambda idx: ratings[idx], reverse=reverse_order
        )

        # Reorder the input data based on the final ranking
        result = [input_data[idx] for idx in final_ranking]

        # Add rank information to each document
        results_with_rank = []
        for i, item in enumerate(result):
            item_copy = item.copy()
            item_copy["_rank"] = i + 1
            results_with_rank.append(item_copy)

        return results_with_rank, total_cost

    def execute(self, input_data: list[dict]) -> tuple[list[dict], float]:
        """
        Starts with an initial ordering based on the specified method, then applies "picky" windows.
        A "picky" window is one that is very large, but the LLM only has to pick a handful of documents from it.
        The window will then be reranked such that the LLM's picks are at the top.
        We start out with small picky windows, then increase the size as we progress.

        Args:
            input_data (list[dict]): The dataset to order.

        Returns:
            tuple[list[dict], float]: A tuple containing the ordered results and the total cost.
        """
        if len(input_data) <= 1:
            return input_data, 0

        initial_ordering_method = self.config.get("initial_ordering_method", "likert")
        k = self.config.get("k", None)
        budget = (
            self.config.get("rerank_call_budget", None)
            or self.config.get("call_budget", None)
            or 10
        )

        # If k is None, set it to the length of input_data
        if k is None:
            k = len(input_data)
        else:
            k = min(k, len(input_data))  # Ensure k doesn't exceed input_data length

        # Validate initial ordering method
        if initial_ordering_method not in [
            "embedding",
            "likert",
            "calibrated_embedding",
        ]:
            raise ValueError(
                "initial_ordering_method must be either 'embedding' or 'likert' or 'calibrated_embedding'"
            )
        num_top_items = self.config.get("num_top_items_per_window", 3)
        overlap_fraction = self.config.get("overlap_fraction", 0.5)
        verbose = self.config.get("verbose", False)

        total_cost = 0

        # Get initial ordering using the specified method
        if verbose:
            self.console.log(
                f"[bold blue]Step 1: Initial Ordering using {initial_ordering_method} method[/bold blue]"
            )

        if initial_ordering_method == "embedding":
            # Use embedding-based rating for initial ordering
            initial_results, initial_cost = self._execute_rating_embedding_qurk(
                input_data
            )
            total_cost += initial_cost
        elif initial_ordering_method == "calibrated_embedding":
            # Use calibrated embedding for initial ordering
            initial_results, initial_cost = self._execute_calibrated_embedding_sort(
                input_data
            )
            total_cost += initial_cost
        else:  # likert
            # Use Likert scale rating for initial ordering
            initial_results, initial_cost = self._execute_likert_rating_qurk(input_data)
            total_cost += initial_cost

        assert len(initial_results) == len(
            input_data
        ), "Initial results must be the same length as the input data"

        # We will do a sliding window approach over the dataset, starting backwards
        # Calculate the step size to ensure we have exactly budget windows to cover k items
        num_windows = budget  # Set number of windows to match budget exactly
        # We need (budget-1) steps to create budget windows
        step_size = max(1, int(k / (budget - 1 or 1)))  # Avoid division by zero
        # Window size needs to be larger than step size to create overlap
        window_size = min(
            k,
            (
                max(num_top_items, int(step_size / (1 - overlap_fraction)))
                if overlap_fraction < 1
                else step_size * 2
            ),
        )

        current_results = initial_results.copy()

        if verbose:
            self.console.log(
                f"[bold blue]Step 2: Creating {num_windows} Windows For {budget} Calls To Refine; window size: {window_size}, step size: {step_size}[/bold blue]"
            )

        # Slide through starting at the end of the k documents
        iter_num = 0

        # Add a unique identifier to track each document
        for i, doc in enumerate(current_results):
            if "_docetl_id" not in doc:
                doc["_docetl_id"] = f"doc_{i}"

        # Create a map of document IDs to positions for tracking
        doc_id_to_position = {
            doc["_docetl_id"]: i for i, doc in enumerate(current_results)
        }

        pbar = RichLoopBar(
            range(k - 1, 0, -step_size),
            desc=f"Processing {self.config['name']} (sliding window refinement)",
            console=self.console,
        )

        for i in pbar:
            iter_num += 1
            # Get the window
            end_idx = i
            start_idx = max(0, end_idx - window_size)
            if start_idx == 0:
                end_idx = window_size

            window_indices = list(range(start_idx, end_idx))

            # Print the iteration number
            if verbose:
                self.console.log(
                    f"[bold blue]Iteration {iter_num} of {num_windows}; window: {window_indices}[/bold blue]"
                )

            # Skip if window is too small
            if len(window_indices) < num_top_items:
                if verbose:
                    self.console.log(
                        f"[yellow]Window too small: {len(window_indices)} < {num_top_items}. Skipping.[/yellow]"
                    )
                continue

            # Extract the window docs for the LLM to evaluate
            window_docs = [current_results[idx] for idx in window_indices]

            # Ask the LLM to pick num_top_items documents from the window
            picked_window_indices, picked_window_cost = self._execute_picky_window(
                window_docs, num_top_items
            )
            total_cost += picked_window_cost

            # Convert window-relative indices to document IDs
            picked_doc_ids = [
                window_docs[idx]["_docetl_id"]
                for idx in picked_window_indices
                if idx < len(window_docs)
            ]

            # Make sure we have unique picks
            picked_doc_ids = list(
                dict.fromkeys(picked_doc_ids)
            )  # Remove duplicates while preserving order

            # Limit to the requested number of top items
            picked_doc_ids = picked_doc_ids[:num_top_items]

            if verbose:
                self.console.log(f"Selected document IDs: {picked_doc_ids}")

            # Move the picked documents to the beginning of the window
            # This ensures each document is in exactly one place
            for target_idx, doc_id in enumerate(picked_doc_ids):
                if target_idx >= len(window_indices):
                    break

                # Get the current position of this document
                current_pos = doc_id_to_position[doc_id]

                # If it's already at the target position, skip
                if current_pos == window_indices[target_idx]:
                    continue

                # Find which document is currently at our target position
                target_pos = window_indices[target_idx]
                doc_at_target = current_results[target_pos]

                # Swap the positions
                current_results[current_pos], current_results[target_pos] = (
                    doc_at_target,
                    current_results[current_pos],
                )

                # Update the position map for both documents
                doc_id_to_position[doc_id] = target_pos
                doc_id_to_position[doc_at_target["_docetl_id"]] = current_pos

            # Verify uniqueness after each iteration
            doc_ids = [doc["_docetl_id"] for doc in current_results]
            assert len(set(doc_ids)) == len(
                doc_ids
            ), f"Duplicate document IDs found after iteration {iter_num}"
            assert len(doc_ids) == len(
                input_data
            ), f"Number of documents changed after iteration {iter_num}"

            if start_idx == 0:
                break

        # Return the final results but change the _rank to be the index of the document in the original input data
        assert len(current_results) == len(
            input_data
        ), "Current results must be the same length as the input data"
        final_results = []
        for i, doc in enumerate(current_results):
            doc["_rank"] = i + 1
            # Remove the _docetl_id
            doc.pop("_docetl_id", None)

            final_results.append(doc)
        return final_results, total_cost

    def _execute_picky_window(
        self, window_docs: list[dict], num_top_items: int
    ) -> list[int]:
        """
        Asks the LLM to pick the top N items from a window of documents and returns
        indices to those documents within the window.

        Args:
            window_docs (list[dict]): The window of documents to process.
            num_top_items (int): Number of top items to pick from the window.

        Returns:
            list[int]: The indices of picked documents (relative to the window).
        """
        # Get the criteria and model from config
        criteria = self.config["prompt"]
        direction = self.config["direction"]
        model = self.config.get("model", self.default_model)
        input_keys = self.config["input_keys"]

        # Construct document texts for the prompt with letter identifiers
        document_texts = []
        for i, doc in enumerate(window_docs):
            doc_content = self._extract_document_content(doc, input_keys)
            document_texts.append(f"Document {i}:\n{doc_content}")

        documents_text = "\n\n".join(document_texts)

        # Construct the prompt for picking top items only
        top_or_bottom = "top" if direction == "desc" else "bottom"
        prompt = f"""You are tasked with ranking documents based on specific criteria.

        CRITERIA:
        {criteria}

        RANKING DIRECTION:
        - We are ordering by '{direction}' (where 'desc' means highest-ranked first, 'asc' means lowest-ranked first)
        - You need to select the {top_or_bottom} {num_top_items} documents that best match the criteria

        DOCUMENTS TO EVALUATE:
        {documents_text}

        INSTRUCTIONS:
        1. Evaluate each document against the criteria
        2. Select the {num_top_items} documents that best match the criteria
        3. Return ONLY a list of document numbers in {'ascending' if direction == 'asc' else 'descending'} order of relevance

        RESPONSE FORMAT:
        Return only a list of integers representing document numbers, like: [3, 1, 4, 0, 2]
        """

        # Call the LLM
        response = self.runner.api.call_llm(
            model,
            "pick_top",
            [{"role": "user", "content": prompt}],
            {f"{top_or_bottom}_picks": "list[int]"},
            timeout_seconds=self.config.get("timeout", 120),
            max_retries_per_timeout=self.config.get("max_retries_per_timeout", 2),
            bypass_cache=self.config.get("bypass_cache", self.bypass_cache),
            litellm_completion_kwargs=self.config.get("litellm_completion_kwargs", {}),
            op_config=self.config,
        )

        # Parse the response
        try:
            output = self.runner.api.parse_llm_response(
                response.response,
                {f"{top_or_bottom}_picks": "list[int]"},
            )[0]

            # Extract the top picks (document indices)
            try:
                # Get the picks from the response
                picks = output[f"{top_or_bottom}_picks"]

                # Filter out invalid indices and ensure they're within bounds
                valid_picks = [idx for idx in picks if 0 <= idx < len(window_docs)]

                # If we didn't get enough valid picks, log a warning
                if len(valid_picks) < min(num_top_items, len(window_docs)):
                    self.console.log(
                        f"[yellow]Warning: LLM returned only {len(valid_picks)} valid picks out of {num_top_items} requested.[/yellow]"
                    )

                return valid_picks, response.total_cost

            except Exception as e:
                self.console.log(
                    f"[yellow]Error parsing top picks: {str(e)}. Returning default order.[/yellow]"
                )
                # Fall back to first N indices in window
                return list(range(min(num_top_items, len(window_docs)))), 0

        except Exception as e:
            self.console.log(
                f"[yellow]Error parsing LLM response: {str(e)}. Returning default order.[/yellow]"
            )
            # Fall back to first N indices in window
            return list(range(min(num_top_items, len(window_docs)))), 0

    def _execute_calibrated_embedding_sort(
        self, input_data: list[dict]
    ) -> tuple[list[dict], float]:
        if len(input_data) <= 1:
            return input_data, 0

        from sklearn.metrics.pairwise import cosine_similarity

        input_keys = self.config["input_keys"]
        embedding_model = self.config.get("embedding_model", "text-embedding-3-small")
        total_cost = 0

        document_contents = [
            self._extract_document_content(doc, input_keys) for doc in input_data
        ]

        # First, do an all-pairs comparison with the qurk baseline on a sample of 20 documents
        # Get a random sample of 20 documents
        sample_size = min(20, len(input_data))
        random.seed(42)
        sample_indices = random.sample(range(len(input_data)), sample_size)
        sample_docs = [input_data[i] for i in sample_indices]

        # Run the all-pairs comparison with the qurk baseline
        qurk_results, qurk_cost = self._execute_comparison_qurk(
            sample_docs, sample=True
        )
        total_cost += qurk_cost

        # Create embeddings for the qurk_results for calibration
        sorted_sample_docs = [
            self._extract_document_content(doc, input_keys) for doc in qurk_results
        ]
        sorted_sample_embeddings = self.runner.api.gen_embedding(
            model=embedding_model, input=sorted_sample_docs
        )
        total_cost += completion_cost(sorted_sample_embeddings)

        # Process documents in batches of 1000
        document_embeddings = []
        batch_size = 1000

        for i in range(0, len(document_contents), batch_size):
            batch = document_contents[i : i + batch_size]
            batch_embeddings_response = self.runner.api.gen_embedding(
                model=embedding_model, input=batch
            )
            total_cost += completion_cost(batch_embeddings_response)

            # Extract embeddings from the batch response
            batch_embeddings = [
                data["embedding"] for data in batch_embeddings_response["data"]
            ]
            document_embeddings.extend(batch_embeddings)

        # Calculate cosine similarity between all document embeddings and sorted sample embeddings in a vectorized way
        import numpy as np

        doc_embeddings_array = np.array(document_embeddings)
        sample_embeddings_array = np.array(
            [data["embedding"] for data in sorted_sample_embeddings["data"]]
        )

        # Calculate cosine similarity matrix between all document embeddings and all sample embeddings
        # Shape: (num_documents, num_samples)
        similarity_matrix = cosine_similarity(
            doc_embeddings_array, sample_embeddings_array
        )

        # For each document, find the index of the most similar sample and its similarity score
        max_similarity_indices = np.argmax(similarity_matrix, axis=1)
        max_similarity_scores = np.max(similarity_matrix, axis=1)

        # Create a list of (original_doc_index, sample_idx, similarity_score) tuples
        doc_similarity_info = [
            (i, max_similarity_indices[i], max_similarity_scores[i])
            for i in range(len(input_data))
        ]

        # Sort by sample index (ascending) and then by similarity (descending) for tie-breaking
        doc_similarity_info.sort(key=lambda x: (x[1], -x[2]))

        # Return the ordered data, making sure to include all original documents
        ordered_data = []
        for i, (orig_idx, _, _) in enumerate(doc_similarity_info):
            new_doc = input_data[
                orig_idx
            ].copy()  # Create a copy to avoid modifying the original
            new_doc["_rank"] = i + 1
            ordered_data.append(new_doc)

        # Verify we have all documents
        assert len(ordered_data) == len(
            input_data
        ), f"Expected {len(input_data)} documents but got {len(ordered_data)}"

        return ordered_data, total_cost
