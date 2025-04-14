import math
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import Field
from rich.progress import Progress
from rich.prompt import Confirm
from rich.table import Table
from sklearn.metrics.pairwise import cosine_similarity

from docetl.operations.base import BaseOperation
from docetl.operations.utils import rich_as_completed
from docetl.utils import completion_cost


class OrderOperation(BaseOperation):
    class schema(BaseOperation.schema):
        type: str = "order"
        prompt: str
        input_keys: List[str]
        direction: Literal["asc", "desc"]
        model: Optional[str] = None
        embedding_model: Optional[str] = None
        batch_size: int = 10
        num_passes: int = 1
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
            document_texts.append(f"Document {i+1}:\n{doc}")

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
                        f"[yellow]Warning: LLM returned incomplete ranking for {batch_label} ({len(ranking)} of {len(batch)} documents). Using embedding similarity as fallback.[/yellow]"
                    )
                    return None, response.total_cost

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

    def _log_position_score_table(self, position_scores, position_counts, top_n=20):
        """
        Creates and displays a rich table showing position scores for the top and bottom documents.

        Args:
            position_scores: Dictionary mapping document indices to their accumulated scores
            position_counts: Dictionary mapping document indices to the count of batches they appeared in
            top_n: Number of top and bottom documents to display
        """
        # Create a table for position scores
        table = Table(title="Document Position Scores")
        table.add_column("Document Index", justify="right", style="cyan")
        table.add_column("Total Score", justify="right")
        table.add_column("Batch Count", justify="right")
        table.add_column("Average Score", justify="right", style="green")

        # Calculate average scores
        avg_scores = {}
        for idx in position_scores:
            if position_counts[idx] > 0:
                avg_scores[idx] = position_scores[idx] / position_counts[idx]
            else:
                avg_scores[idx] = 0

        # Get indices of documents with highest and lowest scores
        sorted_indices = sorted(
            avg_scores.keys(), key=lambda x: avg_scores[x], reverse=True
        )
        top_indices = sorted_indices[:top_n]
        bottom_indices = sorted_indices[-top_n:] if len(sorted_indices) > top_n else []

        # Add top documents to table
        for idx in top_indices:
            table.add_row(
                str(idx),
                f"{position_scores[idx]:.2f}",
                str(position_counts[idx]),
                f"{avg_scores[idx]:.2f} ↑",
            )

        # Add separator if showing both top and bottom
        if bottom_indices and bottom_indices[0] != top_indices[-1]:
            table.add_row("...", "...", "...", "...")

        # Add bottom documents to table
        for idx in reversed(bottom_indices):
            if idx not in top_indices:  # Avoid duplicates for small datasets
                table.add_row(
                    str(idx),
                    f"{position_scores[idx]:.2f}",
                    str(position_counts[idx]),
                    f"{avg_scores[idx]:.2f} ↓",
                )

        self.console.print(table)

    def _select_diverse_sample(
        self, documents: List[Dict], sample_size: int
    ) -> List[int]:
        """
        Selects a diverse sample of document indices based on embedding similarity.

        Args:
            documents: List of documents to sample from
            sample_size: Number of documents to sample

        Returns:
            List of selected document indices
        """
        if len(documents) <= sample_size:
            return list(range(len(documents)))

        # Extract document contents
        input_keys = self.config["input_keys"]
        document_contents = [
            self._extract_document_content(doc, input_keys) for doc in documents
        ]

        # Generate document embeddings
        embedding_model = self.config.get("embedding_model", "text-embedding-3-small")
        document_embeddings_response = self.runner.api.gen_embedding(
            model=embedding_model, input=document_contents
        )
        document_embeddings = [
            data["embedding"] for data in document_embeddings_response["data"]
        ]

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(document_embeddings)

        # Initialize with a random document
        selected_indices = [random.randint(0, len(documents) - 1)]

        # Greedily select the most dissimilar documents
        for _ in range(sample_size - 1):
            # Calculate average similarity to already selected documents
            avg_similarities = []
            for i in range(len(documents)):
                if i in selected_indices:
                    avg_similarities.append(float("inf"))  # Already selected
                else:
                    avg_similarity = sum(
                        similarity_matrix[i][j] for j in selected_indices
                    ) / len(selected_indices)
                    avg_similarities.append(avg_similarity)

            # Select the document with minimum average similarity
            next_index = avg_similarities.index(min(avg_similarities))
            selected_indices.append(next_index)

        return selected_indices

    def _log_verification_results(
        self, original_ranking, verified_ranking, segment_name
    ):
        """
        Logs detailed information about verification results for top/bottom segments.

        Args:
            original_ranking: Original ranking of the segment
            verified_ranking: New verified ranking
            segment_name: Name of the segment (e.g., "Top" or "Bottom")
        """
        if original_ranking == verified_ranking:
            self.console.log(
                f"[green]{segment_name} segment verification: No changes needed[/green]"
            )
            return

        # Calculate stats about changes
        change_info = self._calculate_position_change(
            original_ranking, verified_ranking
        )

        # Create a table showing the changes
        table = Table(title=f"{segment_name} Segment Verification Results")
        table.add_column("Document", justify="right", style="cyan")
        table.add_column("Original Position", justify="right")
        table.add_column("Verified Position", justify="right")
        table.add_column("Change", justify="right", style="yellow")

        # Add rows for each document position change
        for doc_idx, orig_pos, new_pos, change in change_info["position_changes"]:
            change_indicator = ""
            if change > 0:
                if orig_pos > new_pos:
                    change_indicator = f"↑ {change}"  # Moved up
                else:
                    change_indicator = f"↓ {change}"  # Moved down

            table.add_row(str(doc_idx), str(orig_pos), str(new_pos), change_indicator)

        # Log the summary
        self.console.log(f"[bold]{segment_name} segment verification summary:[/bold]")
        self.console.log(
            f"Average position change: {change_info['average_position_change']:.2f}"
        )
        self.console.log(f"Max position change: {change_info['max_position_change']}")
        self.console.log(
            f"Percent of documents changed: {change_info['percent_changed']:.1f}%"
        )

        # Print the table if in debug mode
        if self.config.get("verbose", False):
            self.console.print(table)

    def _calculate_head_to_head_ranking(
        self,
        comparisons,
        document_count,
        previous_ranking=None,
        reverse_order=True,
        previous_ranking_weight=0.5,
    ):
        """
        Calculate ranking based on head-to-head win/loss comparisons.

        Args:
            comparisons: List of tuples (winner_idx, loser_idx, weight) representing comparisons
            document_count: Total number of documents
            previous_ranking: Optional previous ranking to use for missing comparisons
            similarities: Kept for compatibility but not used
            reverse_order: Whether higher win percentage should be ranked higher
            previous_ranking_weight: Weight for previous ranking-based comparisons

        Returns:
            Tuple of (ranking, win_percentages)
        """
        # Initialize win/loss tracking dictionaries
        win_counts = {idx: 0 for idx in range(document_count)}
        comparison_counts = {idx: 0 for idx in range(document_count)}
        compared_pairs = set()

        # Process all direct pairwise comparisons
        for winner_idx, loser_idx, weight in comparisons:
            win_counts[
                winner_idx
            ] += 1.0  # Use fixed weight of 1.0 for direct comparisons
            comparison_counts[winner_idx] += 1.0
            comparison_counts[loser_idx] += 1.0
            compared_pairs.add((winner_idx, loser_idx))

        # Handle missing comparisons using previous ranking if available
        if previous_ranking is not None:
            # Create a mapping from document index to position in previous ranking
            prev_rank_map = {
                doc_idx: pos for pos, doc_idx in enumerate(previous_ranking)
            }

            # Process all possible pairs
            for idx1 in range(document_count):
                for idx2 in range(idx1 + 1, document_count):
                    # Skip pairs that were directly compared
                    if (idx1, idx2) in compared_pairs or (idx2, idx1) in compared_pairs:
                        continue

                    # Both documents exist in previous ranking
                    if idx1 in prev_rank_map and idx2 in prev_rank_map:
                        prev_pos1 = prev_rank_map[idx1]
                        prev_pos2 = prev_rank_map[idx2]

                        # Determine winner based on previous ranking
                        if prev_pos1 < prev_pos2:  # Lower position = higher rank
                            winner, loser = idx1, idx2
                        else:
                            winner, loser = idx2, idx1

                        # Add with lower weight than direct comparisons
                        win_counts[winner] += previous_ranking_weight
                        comparison_counts[winner] += previous_ranking_weight
                        comparison_counts[loser] += previous_ranking_weight

        # Calculate win percentage for each document
        win_percentages = {}
        for idx in range(document_count):
            if comparison_counts[idx] > 0:
                win_percentages[idx] = win_counts[idx] / comparison_counts[idx]
            else:
                # Fallback for documents with no comparisons at all
                win_percentages[idx] = 0.5

        # Create final ranking based on win percentages
        ranking = sorted(
            range(document_count),
            key=lambda idx: win_percentages[idx],
            reverse=reverse_order,  # Higher win percentage = higher rank
        )

        return ranking, win_percentages

    def execute(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        """
        Executes an improved order operation on the provided dataset.

        This implementation uses a tournament-based approach with LLM ranking
        that more effectively preserves relative ordering relationships.

        Args:
            input_data (List[Dict]): The dataset to order.

        Returns:
            Tuple[List[Dict], float]: A tuple containing the ordered results and the total cost.
        """
        if len(input_data) <= 1:
            return input_data, 0

        random.seed(42)

        criteria = self.config["prompt"]
        input_keys = self.config["input_keys"]
        direction = self.config["direction"].lower()
        model = self.config.get("model", self.default_model)
        embedding_model = self.config.get("embedding_model", "text-embedding-3-small")
        batch_size = self.config.get("batch_size", 10)
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

        # Step 0: Create a calibration set
        self.console.log("[bold blue]Step 0: Creating Calibration Set[/bold blue]")

        # Select a small representative sample (8-10 docs)
        calibration_size = min(batch_size, len(input_data))
        calibration_indices = self._select_diverse_sample(input_data, calibration_size)
        calibration_docs = [input_data[idx] for idx in calibration_indices]

        # Get "ground truth" ordering for calibration set using comparison approach
        calibration_results, calibration_cost = self._execute_comparison_qurk(
            calibration_docs
        )
        total_cost += calibration_cost

        # Map back to original indices and extract ordering
        calibration_ordering = []
        for doc in calibration_results:
            for i, orig_doc in enumerate(calibration_docs):
                if doc == orig_doc:
                    calibration_ordering.append(calibration_indices[i])
                    break

        # Step 1: Generate Likert ratings for pre-sorting
        self.console.log(
            "[bold blue]Step 1: Generating Likert Ratings for Pre-sorting[/bold blue]"
        )

        document_contents = [
            self._extract_document_content(doc, input_keys) for doc in input_data
        ]

        # Create context text
        context_texts = []
        for i, doc in enumerate(calibration_docs):
            doc_text = self._extract_document_content(doc, input_keys)
            context_texts.append(f"Example {i+1}:\n{doc_text}")

        context_text = "\n\n".join(context_texts)

        # Process documents in batches for Likert ratings
        likert_ratings = {}

        with Progress(console=self.console) as progress:
            rating_task = progress.add_task(
                "Generating Likert ratings...",
                total=math.ceil(len(input_data) / batch_size),
            )

            for i in range(0, len(input_data), batch_size):
                batch_indices = list(range(i, min(i + batch_size, len(input_data))))
                batch_docs = [document_contents[idx] for idx in batch_indices]

                # Construct document texts
                document_texts = []
                for j, doc in enumerate(batch_docs):
                    document_texts.append(f"Document {j+1}:\n{doc}")

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

                For context, here are some example documents from the dataset, which are already sorted by the ground truth ordering:
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
                    {"ratings": "list[int]"},
                    timeout_seconds=self.config.get("timeout", 120),
                    max_retries_per_timeout=self.config.get(
                        "max_retries_per_timeout", 2
                    ),
                    bypass_cache=self.config.get("bypass_cache", False),
                    litellm_completion_kwargs=self.config.get(
                        "litellm_completion_kwargs", {}
                    ),
                    op_config=self.config,
                )

                total_cost += response.total_cost

                # Parse the response
                try:
                    output = self.runner.api.parse_llm_response(
                        response.response,
                        {"ratings": "list[int]"},
                    )[0]

                    batch_ratings = output["ratings"]

                    # Validate and store ratings
                    for j, rating in enumerate(batch_ratings):
                        if j < len(batch_docs) and 1 <= rating <= 7:
                            doc_idx = batch_indices[j]
                            likert_ratings[doc_idx] = rating

                except Exception as e:
                    self.console.log(
                        f"[yellow]Error parsing ratings response: {str(e)}[/yellow]"
                    )

                progress.update(rating_task, advance=1)

        # Handle any documents that weren't rated
        for idx in range(len(input_data)):
            if idx not in likert_ratings:
                # Assign a neutral rating
                likert_ratings[idx] = 4

        # Also generate embeddings for tie-breaking
        with Progress(console=self.console) as progress:
            embedding_task = progress.add_task(
                "Generating embeddings for tie-breaking...", total=2
            )

            # Generate embedding for criteria
            progress.update(
                embedding_task, description="Generating criteria embedding..."
            )
            embedding_model = self.config.get(
                "embedding_model", "text-embedding-3-small"
            )
            criteria_embedding_response = self.runner.api.gen_embedding(
                model=embedding_model, input=[criteria]
            )
            criteria_embedding = criteria_embedding_response["data"][0]["embedding"]
            total_cost += completion_cost(criteria_embedding_response)
            progress.update(embedding_task, advance=1)

            # Generate embeddings for documents
            progress.update(
                embedding_task, description="Generating document embeddings..."
            )
            document_embeddings_response = self.runner.api.gen_embedding(
                model=embedding_model, input=document_contents
            )
            total_cost += completion_cost(document_embeddings_response)

            document_embeddings = [
                data["embedding"] for data in document_embeddings_response["data"]
            ]
            progress.update(embedding_task, advance=1)

        # Calculate similarity scores
        similarities = cosine_similarity([criteria_embedding], document_embeddings)[0]

        # Combine Likert ratings and similarity scores for a hybrid score
        hybrid_scores = {}
        for idx in range(len(input_data)):
            # Normalize Likert rating to [0,1] range
            likert_norm = (likert_ratings[idx] - 1) / 6.0  # Scales 1-7 to 0-1

            # Weighted combination (adjust weights as needed)
            likert_weight = 0.7  # Primary signal
            sim_weight = 0.3  # Secondary signal for tie-breaking

            hybrid_scores[idx] = (likert_norm * likert_weight) + (
                similarities[idx] * sim_weight
            )

        reverse_order = direction == "desc"
        initial_ranking = sorted(
            range(len(input_data)),
            key=lambda i: hybrid_scores[i],
            reverse=reverse_order,
        )

        # Step 2: Tournament-style LLM ranking with windows
        self.console.log(
            "[bold blue]Step 2: Tournament-style window Creation[/bold blue]"
        )

        # Function to rank a window of documents using LLM
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
                # Fall back to similarity ranking
                window_similarities = [similarities[idx] for idx in window_indices]
                ranking_by_similarity = sorted(
                    range(len(window_indices)),
                    key=lambda i: window_similarities[i],
                    reverse=reverse_order,
                )
                return [window_indices[i] for i in ranking_by_similarity], cost
            else:
                # Map the LLM ranking back to the original indices
                return [window_indices[i] for i in ranking], cost

        # Initialize window rankings
        batch_rankings = []
        batch_costs = []

        # Tournament rounds approach for window creation
        current_ranking = initial_ranking.copy()
        round_num = 0

        # Determine tournament parameters
        min_window_overlap = batch_size // 2
        max_rounds = min(
            5, int(math.log2(len(input_data) / batch_size)) + 1
        )  # Adaptive max rounds

        self.console.log(
            f"[bold]Planning {max_rounds} rounds with window size {batch_size}[/bold]"
        )

        # Track all comparisons across rounds for final statistics
        all_comparisons = []

        while round_num < max_rounds:
            round_num += 1
            self.console.log(f"[bold]Round {round_num}[/bold]")

            # Create overlapping windows
            windows = []
            window_step = max(1, batch_size - min_window_overlap)

            for start_idx in range(0, len(current_ranking), window_step):
                end_idx = min(start_idx + batch_size, len(current_ranking))
                window = current_ranking[start_idx:end_idx]

                # Ensure minimum window size
                if len(window) < 3:
                    continue

                windows.append((start_idx, window))

                # If this is the last window and doesn't include the last document,
                # add an additional window at the end
                if (
                    end_idx < len(current_ranking)
                    and start_idx + window_step >= len(current_ranking) - batch_size
                ):
                    last_start = len(current_ranking) - batch_size
                    if last_start > start_idx:  # Avoid duplicate window
                        last_window = current_ranking[last_start : len(current_ranking)]
                        windows.append((last_start, last_window))
                        break

            self.console.log(
                f"Created {len(windows)} windows with size ~{batch_size} and overlap ~{min_window_overlap}"
            )

            # Process windows in parallel
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                # Submit all window processing tasks
                futures = []
                window_start_indices = []
                for i, (start_idx, window) in enumerate(windows):
                    window_name = f"Round {round_num} - window {i+1}/{len(windows)}"
                    future = executor.submit(rank_window, window, window_name)
                    futures.append(future)
                    window_start_indices.append(start_idx)

                # Process results as they complete
                for i, future in enumerate(
                    rich_as_completed(
                        futures,
                        total=len(futures),
                        desc=f"Processing Round {round_num} windows",
                        console=self.console,
                    )
                ):
                    try:
                        ranked_window, cost = future.result()
                        batch_rankings.append(ranked_window)
                        batch_costs.append(cost)
                        total_cost += cost
                    except Exception as e:
                        self.console.log(
                            f"[red]Error in window processing: {str(e)}[/red]"
                        )

            # Extract comparisons from this round's windows
            comparisons_round = []
            for window_ranking in batch_rankings[
                -len(windows) :
            ]:  # Only use this round's windows
                for higher_pos, higher_idx in enumerate(window_ranking):
                    for lower_pos, lower_idx in enumerate(
                        window_ranking[higher_pos + 1 :], higher_pos + 1
                    ):
                        # Record comparison (winner, loser, weight)
                        comparison = (higher_idx, lower_idx, 1.0)
                        comparisons_round.append(comparison)
                        all_comparisons.append(
                            comparison
                        )  # Also add to all comparisons for final stats

            # For intermediate rounds: update based on current comparisons + previous ranking
            # For the final round: compute the final ranking using all comparisons
            if round_num == max_rounds:
                self.console.log(
                    "[bold blue]Final Round: Computing Final Ranking[/bold blue]"
                )

                # Calculate final ranking using all accumulated comparisons
                current_ranking, win_percentages_round = (
                    self._calculate_head_to_head_ranking(
                        comparisons=all_comparisons,  # Use ALL comparisons across ALL rounds for final ranking
                        document_count=len(input_data),
                        previous_ranking=None,  # Don't use previous ranking for final result
                        reverse_order=True,
                    )
                )
            else:
                # Update ranking for next round
                current_ranking, win_percentages_round = (
                    self._calculate_head_to_head_ranking(
                        comparisons=comparisons_round,  # Only use this round's comparisons
                        document_count=len(input_data),
                        previous_ranking=current_ranking,  # Use previous ranking to inform missing comparisons
                        reverse_order=True,
                    )
                )

            # If this is the final round, log statistics
            if round_num == max_rounds and verbose:
                self.console.log(
                    f"[bold]Final round completed. Using head-to-head ranking from {len(all_comparisons)} total comparisons.[/bold]"
                )

                # Calculate coverage statistics
                total_possible = len(input_data) * (len(input_data) - 1) // 2
                direct_comparisons = set(
                    (winner, loser) for winner, loser, _ in all_comparisons
                )
                coverage = (len(direct_comparisons) / total_possible) * 100
                self.console.log(
                    f"Comparison coverage: {coverage:.1f}% ({len(direct_comparisons)}/{total_possible} pairs)"
                )

                # Reconstruct win counts and comparison counts for display purposes
                win_counts = {idx: 0 for idx in range(len(input_data))}
                comparison_counts = {idx: 0 for idx in range(len(input_data))}

                for winner_idx, loser_idx, weight in all_comparisons:
                    win_counts[winner_idx] += weight
                    comparison_counts[winner_idx] += weight
                    comparison_counts[loser_idx] += weight

                # Display win records for top documents
                wins_table = Table(title="Document Win Records")
                wins_table.add_column("Document", justify="right", style="cyan")
                wins_table.add_column("Wins", justify="right")
                wins_table.add_column("Comparisons", justify="right")
                wins_table.add_column("Win %", justify="right", style="green")

                # Get top 15 documents by win percentage
                top_docs = sorted(
                    range(len(input_data)),
                    key=lambda idx: win_percentages_round[idx],
                    reverse=True,
                )[:15]

                for idx in top_docs:
                    wins_table.add_row(
                        str(idx),
                        f"{win_counts[idx]:.1f}",
                        f"{comparison_counts[idx]:.1f}",
                        f"{win_percentages_round[idx]:.1%}",
                    )

                self.console.print(wins_table)

        # Use the final ranking from the last round
        final_ranking = current_ranking

        # Log ranking changes compared to embedding-only ranking
        if verbose:
            changes = self._calculate_position_change(initial_ranking, final_ranking)
            self.console.log(
                "[bold]Position changes from initial embedding ranking:[/bold]"
            )
            self.console.log(
                f"  Average position change: {changes['average_position_change']:.2f}"
            )
            self.console.log(
                f"  Maximum position change: {changes['max_position_change']}"
            )
            self.console.log(
                f"  Documents changed position: {changes['percent_changed']:.1f}%"
            )

            # Show top position movers
            significant_changes = sorted(
                changes["position_changes"], key=lambda x: x[3], reverse=True
            )[:5]
            if significant_changes:
                self.console.log("[bold]Top position changes:[/bold]")
                for doc_idx, orig_pos, new_pos, change in significant_changes:
                    direction_indicator = "↑" if orig_pos > new_pos else "↓"
                    self.console.log(
                        f"  Doc {doc_idx}: {orig_pos} → {new_pos} ({direction_indicator}{change} positions)"
                    )

        # Step 4: Reorder the input data based on the final ranking
        result = [input_data[idx] for idx in final_ranking]

        # Add rank information to each document
        results_with_rank = []
        for i, item in enumerate(result):
            item["_rank"] = i + 1
            results_with_rank.append(item.copy())

        if self.status:
            self.status.start()

        return results_with_rank, total_cost

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

        document_contents = [
            self._extract_document_content(doc, input_keys) for doc in input_data
        ]

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
            batch_docs = [document_contents[idx] for idx in batch_indices]

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
        model = self.config.get("model", self.default_model)
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

        document_embeddings_response = self.runner.api.gen_embedding(
            model=embedding_model, input=document_contents
        )
        total_cost += completion_cost(document_embeddings_response)

        document_embeddings = [
            data["embedding"] for data in document_embeddings_response["data"]
        ]

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
        self, input_data: List[Dict], initial_ordering_method: str = "embedding"
    ) -> Tuple[List[Dict], float]:
        """
        Implements the sliding window approach from the human-powered sort paper.
        Starts with an initial ordering based on the specified method, then applies successive windows for reranking.

        Args:
            input_data (List[Dict]): The dataset to order.
            initial_ordering_method (str): Method to use for initial ordering ("embedding" or "likert").

        Returns:
            Tuple[List[Dict], float]: A tuple containing the ordered results and the total cost.
        """
        if len(input_data) <= 1:
            return input_data, 0

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
        document_contents = [
            self._extract_document_content(doc, input_keys) for doc in input_data
        ]

        # Get initial ordering using the specified method
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

        # Extract the ordering from the initial results
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

        self.console.log(
            "[bold blue]Step 2: Applying Sliding Window Refinement[/bold blue]"
        )

        # Apply sliding windows for refinement
        max_windows = len(input_data) // window_step
        self.console.log(
            f"Applying {max_windows} sliding windows with size {window_size} and step {window_step}"
        )

        for window_num in range(max_windows):
            # Calculate window start position
            start_pos = (window_num * window_step) % len(current_ranking)
            end_pos = min(start_pos + window_size, len(current_ranking))

            # Extract window indices and documents
            window_indices = current_ranking[start_pos:end_pos]
            window_docs = [document_contents[idx] for idx in window_indices]

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
                current_ranking[start_pos:end_pos] = reordered_window

                if self.config.get("verbose", False):
                    self.console.log(
                        f"Applied reordering to window {start_pos}-{end_pos}"
                    )

        # Final result
        result = [input_data[idx] for idx in current_ranking]

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

        # Process documents in batches
        for i in range(0, len(input_data), batch_size):
            batch_indices = list(range(i, min(i + batch_size, len(input_data))))
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

            total_cost += response.total_cost

            # Parse the response
            try:
                output = self.runner.api.parse_llm_response(
                    response.response,
                    {"ratings": "list[int]"},
                )[0]

                batch_ratings = output["ratings"]

                # Validate and store ratings
                for j, rating in enumerate(batch_ratings):
                    if j < len(batch_docs) and 1 <= rating <= 7:
                        doc_idx = batch_indices[j]
                        ratings[doc_idx] = rating

            except Exception as e:
                self.console.log(
                    f"[yellow]Error parsing ratings response: {str(e)}[/yellow]"
                )

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
            item["_rank"] = i + 1
            results_with_rank.append(item.copy())

        return results_with_rank, total_cost
