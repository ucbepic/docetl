import math
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import Field
from sklearn.metrics.pairwise import cosine_similarity

from docetl.operations.base import BaseOperation
from docetl.operations.utils import rich_as_completed
from docetl.utils import completion_cost


class RankOperation(BaseOperation):
    class schema(BaseOperation.schema):
        type: str = "order"
        prompt: str
        input_keys: List[str]
        direction: Literal["asc", "desc"]
        model: Optional[str] = None
        embedding_model: Optional[str] = None
        batch_size: int = 10
        initial_ordering_method: str = "embedding"
        k: Optional[int] = None
        call_budget: int = 100
        num_top_items_per_window: int = 3
        overlap_fraction: float = 0.5
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
        5. Validates that numerical parameters (batch_size) are positive integers.

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
        for param in ["batch_size"]:
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
        batch_label: str = "Batch",  # Added parameter for identifying which batch is being processed
    ) -> Tuple[List[int], float]:
        """
        Uses an LLM to rank a batch of documents according to the given criteria and direction.

        Args:
            batch (List[Dict]): A batch of documents to rank.
            criteria (str): The ranking criteria.
            direction (str): The direction of ordering ('asc' or 'desc').
            model (str): The LLM model to use.
            timeout_seconds (int): Timeout for the LLM call.
            batch_label (str): Label to identify which batch is being processed in logs.

        Returns:
            Tuple[List[int], float]: A tuple containing the ranked indices and the cost.
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
            bypass_cache=self.config.get("bypass_cache", False),
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
                ranking = list(set(ranking))

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
        self, input_data: List[Dict]
    ) -> Tuple[List[Dict], float]:
        """
        Implements the comparison-based approach from the human-powered sort paper.
        Uses random batches of S items and head-to-head counting to break ties.

        Args:
            input_data (List[Dict]): The dataset to order.

        Returns:
            Tuple[List[Dict], float]: A tuple containing the ordered results and the total cost.
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
        def process_batch(batch_num: int) -> Tuple[Dict[int, int], float]:
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
        self, input_data: List[Dict]
    ) -> Tuple[List[Dict], float]:
        """
        Implements the rating-based approach from the human-powered sort paper.
        Each document is rated independently and then sorted by the average rating.

        Args:
            input_data (List[Dict]): The dataset to order.

        Returns:
            Tuple[List[Dict], float]: A tuple containing the ordered results and the total cost.
        """
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
        input_data: List[Dict],
        initial_ordering_method: str = "embedding",
        k: Optional[int] = None,
    ) -> Tuple[List[Dict], float]:
        """
        Implements the sliding window approach from the human-powered sort paper.
        Starts with an initial ordering based on the specified method, then applies successive windows for reranking.

        Args:
            input_data (List[Dict]): The dataset to order.
            initial_ordering_method (str): Method to use for initial ordering ("embedding" or "likert").
            k (Optional[int]): Number of top elements to focus on. If None, set to len(input_data).

        Returns:
            Tuple[List[Dict], float]: A tuple containing the ordered results and the total cost.
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
        self, input_data: List[Dict]
    ) -> Tuple[List[Dict], float]:
        """
        Implements a rating-based approach using a 7-point Likert scale with LLM ratings.
        Each document is rated by the LLM on a 1-7 scale and then sorted by the rating.
        Uses a thread pool to parallelize LLM calls for better performance.

        Args:
            input_data (List[Dict]): The dataset to order.

        Returns:
            Tuple[List[Dict], float]: A tuple containing the ordered results and the total cost.
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
        max_workers = self.max_threads

        total_cost = 0
        ratings = {}

        # Select a random sample of documents to provide context
        random.seed(42)
        context_size = min(10, len(input_data))
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
                bypass_cache=self.config.get("bypass_cache", False),
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

            # Collect results as they complete
            for future in futures:
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

    def _execute_tournament_window(
        self, input_data: List[Dict], call_budget: int
    ) -> Tuple[List[Dict], float]:
        """
        Implements a tournament-style windowed approach for finding the top K documents.
        Starts with an embedding-based sort, then runs successive rounds of disjoint window
        reranking, keeping the top half of documents in each round.

        The function tracks all pairwise comparisons throughout the tournament
        and uses them to create a full ordering at the end.

        Args:
            input_data (List[Dict]): The dataset to order.
            call_budget (int): Maximum number of LLM calls allowed.

        Returns:
            Tuple[List[Dict], float]: A tuple containing the ordered results and the total cost.
        """
        import numpy as np

        if len(input_data) <= 1:
            return input_data, 0

        # Validate parameters
        if call_budget <= 0:
            raise ValueError("call_budget must be positive")

        criteria = self.config["prompt"]
        input_keys = self.config["input_keys"]
        direction = self.config["direction"].lower()
        model = self.config.get("model", self.default_model)
        window_size = self.config.get("batch_size", 10)  # Default window size
        verbose = self.config.get("verbose", False)
        k = self.config.get("k", None) or len(input_data)

        total_cost = 0

        # Step 1: Get initial ordering using embedding-based rating
        self.console.log(
            "[bold blue]Step 1: Initial Ordering using Embeddings[/bold blue]"
        )

        # Use embedding-based rating for initial ordering
        initial_results, initial_cost = self._execute_rating_embedding_qurk(input_data)
        total_cost += initial_cost

        # Extract the initial ranking
        current_ranking = []
        for doc in initial_results:
            # Find the index of this document in the original input data
            for i, original_doc in enumerate(input_data):
                if doc.get("_rank") == original_doc.get("_rank") or doc == original_doc:
                    current_ranking.append(i)
                    break

        # If we couldn't match the documents, fall back to their indices
        if len(current_ranking) != len(input_data):
            self.console.log(
                "[yellow]Warning: Couldn't match all documents from initial ordering. Using indices as fallback.[/yellow]"
            )
            current_ranking = list(range(len(input_data)))

        # Step 2: Tournament-style rounds
        self.console.log("[bold blue]Step 2: Tournament Rounds[/bold blue]")

        document_contents = [
            self._extract_document_content(doc, input_keys) for doc in input_data
        ]

        # Initialize pairwise comparison tracking
        n_documents = len(input_data)
        comparison_counts = np.zeros((n_documents, n_documents))
        comparison_wins = np.zeros((n_documents, n_documents))

        # Add initial pairwise comparisons from embedding ranking
        # Every document beats all documents ranked below it with 0.5 weight
        for i in range(len(current_ranking)):
            doc_i = current_ranking[i]
            for j in range(i + 1, len(current_ranking)):
                doc_j = current_ranking[j]
                comparison_counts[doc_i, doc_j] += 0.5
                comparison_counts[doc_j, doc_i] += 0.5
                comparison_wins[doc_i, doc_j] += 0.5  # i beats j with weight 0.5

        # Function to rank a window of documents using LLM and update pairwise comparisons
        def rank_window(window_indices, window_name):
            window_docs = [document_contents[idx] for idx in window_indices]

            ranking, cost = self._batch_rank_documents(
                window_docs,
                criteria,
                direction,
                model,
                timeout_seconds=self.config.get("timeout", 120),
                batch_label=window_name,
            )

            if ranking is None:
                # Fall back to the current ordering
                return window_indices, cost, None
            else:
                # Map the LLM ranking back to the original indices
                ranked_indices = [window_indices[i] for i in ranking]
                return ranked_indices, cost, ranking

        # Track actual LLM calls used
        calls_used = 0

        # Determine rounds and strategy based on call budget
        n_documents = len(input_data)

        # Calculate the advancement rate - how many documents move forward from each window
        advancement_rate = 0.5  # Default - top half advances

        # The number of LLM calls follows a geometric series: 2^0, 2^1, 2^2, ..., 2^(r-1)
        # Sum of geometric series: (1 - 2^r) / (1 - 2) = 2^r - 1
        # Therefore, if budget = 2^r - 1, then r = log2(budget + 1)
        max_rounds = math.floor(math.log2(call_budget + 1))

        # Calculate how many documents we can process in the first round
        # Each window in first round can handle window_size documents
        # First round has 2^(max_rounds-1) windows
        first_round_windows = 2 ** (max_rounds - 1)
        first_round_docs = first_round_windows * window_size

        # Adjust if we have more docs than we can handle in our calculated rounds
        if n_documents > first_round_docs:
            if verbose:
                self.console.log(
                    "[yellow]Warning: Not all documents can be processed with the given budget.[/yellow]"
                )
                self.console.log(
                    f"Processing {first_round_docs} of {n_documents} documents."
                )
            # We'll need to preselect documents based on embedding ranking
            current_ranking = current_ranking[:first_round_docs]

        # Create projected calls array
        projected_calls = [2 ** (max_rounds - 1 - i) for i in range(max_rounds)]

        total_projected_calls = sum(projected_calls)

        if verbose:
            self.console.log("[bold]Tournament approach parameters:[/bold]")
            self.console.log(f"Documents: {n_documents}")
            self.console.log(f"Window size: {window_size}")
            self.console.log(f"Projected calls per round: {projected_calls}")
            self.console.log(
                f"Total projected calls: {total_projected_calls} (budget: {call_budget})"
            )

        # Run each round of the tournament based on the projected strategy
        rounds_to_run = len(projected_calls)
        docs_in_round = n_documents

        for round_num in range(rounds_to_run):
            self.console.log(f"[bold]Round {round_num+1}/{rounds_to_run}[/bold]")

            # Create windows for this round, ensuring each window is at most window_size
            windows = []
            for i in range(0, len(current_ranking), window_size):
                window = current_ranking[i : i + window_size]
                if window:  # Only add non-empty windows
                    windows.append((len(windows), window))

            # Calculate how many windows we can process with remaining budget
            windows_to_process = min(len(windows), call_budget - calls_used)

            if windows_to_process <= 0 or docs_in_round <= k:
                if verbose:
                    if docs_in_round <= k:
                        self.console.log(
                            f"Already have {docs_in_round} ≤ {k} documents. Stopping."
                        )
                    else:
                        self.console.log(
                            f"Call budget exhausted. Stopping with {calls_used} calls used."
                        )
                break

            self.console.log(
                f"Processing {windows_to_process} windows in round {round_num+1}"
            )

            # If we can't process all windows, only process windows_to_process
            windows = windows[:windows_to_process]

            # Process windows in parallel
            new_ranking = []
            window_rankings = []

            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                # Submit all window processing tasks
                futures = []
                window_map = {}  # Map futures to their windows
                for i, window in windows:
                    window_name = f"Round {round_num+1} - Window {i+1}/{len(windows)}"
                    future = executor.submit(rank_window, window, window_name)
                    futures.append(future)
                    window_map[future] = (
                        window  # Store the window associated with this future
                    )

                # Process results as they complete
                for future in rich_as_completed(
                    futures,
                    total=len(futures),
                    desc=f"Processing Round {round_num+1} windows",
                    console=self.console,
                ):
                    try:
                        ranked_window, cost, llm_ranking = future.result()

                        # Update pairwise comparisons if we have a valid ranking
                        if llm_ranking is not None:
                            window = window_map[future]  # Get the original window

                            # Update pairwise comparisons for this window
                            for i in range(len(llm_ranking)):
                                doc_i = window[llm_ranking[i]]
                                # i beats all documents ranked below it
                                for j in range(i + 1, len(llm_ranking)):
                                    doc_j = window[llm_ranking[j]]
                                    comparison_counts[doc_i, doc_j] += 1
                                    comparison_counts[doc_j, doc_i] += 1
                                    comparison_wins[doc_i, doc_j] += 1  # i beats j

                        window_rankings.append(ranked_window)
                        total_cost += cost
                        calls_used += 1
                    except Exception as e:
                        self.console.log(
                            f"[red]Error in window processing: {str(e)}[/red]"
                        )

            # Collect ranked documents from each window
            for window_ranking in window_rankings:
                # Add half of the window (or at least enough to eventually reach k)
                docs_to_take = max(1, math.ceil(len(window_ranking) * advancement_rate))
                new_ranking.extend(window_ranking[:docs_to_take])

            # Update for next round
            current_ranking = new_ranking
            docs_in_round = len(current_ranking)

            if verbose:
                self.console.log(
                    f"Round {round_num+1} complete. Keeping top {len(current_ranking)} documents."
                )
                self.console.log(f"Calls used so far: {calls_used}/{call_budget}")

            # If we've reached our target window size, stop
            if docs_in_round < window_size:
                break

        # Step 3: Generate full ordering using pairwise comparisons
        self.console.log(
            "[bold blue]Step 3: Generating full ordering using pairwise comparisons[/bold blue]"
        )

        # Calculate win rate for each document
        win_rates = np.zeros(n_documents)

        for i in range(n_documents):
            total_comparisons = np.sum(comparison_counts[i, :]) + np.sum(
                comparison_counts[:, i]
            )
            if total_comparisons > 0:
                win_rates[i] = np.sum(comparison_wins[i, :]) / total_comparisons

        # For any documents that have no comparisons, use their position in the initial ranking
        for i, idx in enumerate(current_ranking):
            if (
                np.sum(comparison_counts[idx, :]) + np.sum(comparison_counts[:, idx])
                == 0
            ):
                # Give it a win rate based on its position in the initial ranking
                # Scale it to be lower than the lowest compared document
                min_compared_win_rate = (
                    np.min(win_rates[win_rates > 0]) if np.any(win_rates > 0) else 0.5
                )
                win_rates[idx] = min_compared_win_rate * (1 - i / len(current_ranking))

        # Sort all documents by win rate
        full_ranking = np.argsort(-win_rates)

        # Make sure the top from the tournament are at the top
        # We want to ensure that the tournament winners take precedence
        top_k_set = set(current_ranking[: window_size // 2])
        non_top_k = [idx for idx in full_ranking if idx not in top_k_set]

        # Final ranking: top k from tournament followed by remaining docs sorted by win rate
        final_ranking = list(current_ranking[: window_size // 2]) + [
            idx for idx in non_top_k
        ]

        # Reorder the input data based on the final ranking
        result = [input_data[idx] for idx in final_ranking]

        # Add rank information to each document
        results_with_rank = []
        for i, item in enumerate(result):
            item_copy = item.copy()
            item_copy["_rank"] = i + 1
            results_with_rank.append(item_copy)

        if verbose:
            self.console.log("[bold green]Tournament complete![/bold green]")
            self.console.log(f"Total calls used: {calls_used}/{call_budget}")
            self.console.log(f"Result contains {len(results_with_rank)} documents")

        return results_with_rank, total_cost

    def execute(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        """
        Starts with an initial ordering based on the specified method, then applies "picky" windows.
        A "picky" window is one that is very large, but the LLM only has to pick a handful of documents from it.
        The window will then be reranked such that the LLM's picks are at the top.
        We start out with small picky windows, then increase the size as we progress.

        Args:
            input_data (List[Dict]): The dataset to order.
            initial_ordering_method (str): Method to use for initial ordering ("embedding" or "likert").
            k (Optional[int]): Number of top elements to focus on. If None, set to len(input_data).
            call_budget (Optional[int]): Maximum number of LLM calls allowed. If None, set to 100.

        Returns:
            Tuple[List[Dict], float]: A tuple containing the ordered results and the total cost.
        """
        if len(input_data) <= 1:
            return input_data, 0

        initial_ordering_method = self.config.get(
            "initial_ordering_method", "embedding"
        )
        k = self.config.get("k", None)
        budget = self.config.get("call_budget", None) or 100

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

        for i in range(k - 1, 0, -step_size):
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
        self, window_docs: List[Dict], num_top_items: int
    ) -> List[int]:
        """
        Asks the LLM to pick the top N items from a window of documents and returns
        indices to those documents within the window.

        Args:
            window_docs (List[Dict]): The window of documents to process.
            num_top_items (int): Number of top items to pick from the window.

        Returns:
            List[int]: The indices of picked documents (relative to the window).
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
        prompt = f"""
        {criteria}

        We are only interested in identifying the {top_or_bottom} {num_top_items} documents based on this criteria.
        The direction we are ordering by is {direction}, so we are looking for the {top_or_bottom} {num_top_items} documents.

        Here are the documents:

        {documents_text}

        Return ONLY the document numbers of the {top_or_bottom} {num_top_items} documents that best match the criteria.
        Provide them in order from {'least' if direction == 'asc' else 'most'} to {'most' if direction == 'asc' else 'least'} matching.
        Return only the document numbers as a list.
        Format example: [3, 1, 4, 0, 2]
        """

        # Call the LLM
        response = self.runner.api.call_llm(
            model,
            "pick_top",
            [{"role": "user", "content": prompt}],
            {f"{top_or_bottom}_picks": "list[int]"},
            timeout_seconds=self.config.get("timeout", 120),
            max_retries_per_timeout=self.config.get("max_retries_per_timeout", 2),
            bypass_cache=self.config.get("bypass_cache", False),
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
