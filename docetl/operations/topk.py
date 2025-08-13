"""TopK operation for retrieving top documents by similarity."""

from typing import Literal, Union

from pydantic import Field, field_validator, model_validator

from docetl.operations.base import BaseOperation
from docetl.operations.rank import RankOperation
from docetl.operations.sample import SampleOperation


class TopKOperation(BaseOperation):
    """
    Retrieves top-k items based on similarity to a query using embeddings or full-text search.

    This operation is a specialized interface for the sample operation's top_embedding
    and top_fts methods, providing a cleaner API for retrieval use cases.

    Attributes:
        method: "embedding" for semantic similarity or "fts" for full-text search
        k: Number of items to retrieve (int or float for percentage)
        keys: List of keys to use for similarity matching
        query: Query string (supports Jinja templates with {{ input.field }})
        stratify_key: Optional key(s) for stratified retrieval
        embedding_model: Model to use for embeddings (only for embedding method)
    """

    class schema(BaseOperation.schema):
        type: str = "topk"
        method: Literal["embedding", "fts", "llm_compare"]
        k: Union[int, float] = Field(..., description="Number of items to retrieve")
        keys: list[str] = Field(
            ..., description="Keys to use for similarity matching or comparison"
        )
        query: str = Field(
            ...,
            description="Query string for similarity or comparison criteria (supports Jinja templates)",
        )
        stratify_key: Union[str, list[str]] | None = Field(
            None, description="Key(s) for stratified retrieval"
        )
        embedding_model: str | None = Field(
            "text-embedding-3-small",
            description="Embedding model (for embedding and llm_compare methods)",
        )
        model: str | None = Field(None, description="LLM model for llm_compare method")
        batch_size: int | None = Field(
            10, description="Batch size for LLM comparisons in llm_compare method"
        )

        @field_validator("k")
        def validate_k(cls, v):
            if isinstance(v, (int, float)):
                if v <= 0:
                    raise ValueError("'k' must be a positive number")
            else:
                raise TypeError("'k' must be an integer or float")
            return v

        @field_validator("keys")
        def validate_keys(cls, v):
            if not v:
                raise ValueError("'keys' cannot be empty")
            if not all(isinstance(key, str) for key in v):
                raise TypeError("All items in 'keys' must be strings")
            return v

        @field_validator("query")
        def validate_query(cls, v):
            if not v or not v.strip():
                raise ValueError("'query' cannot be empty")
            return v

        @field_validator("stratify_key")
        def validate_stratify_key(cls, v):
            if v is None:
                return v

            if isinstance(v, str):
                pass  # Single key is valid
            elif isinstance(v, list):
                if not v:
                    raise ValueError("'stratify_key' list cannot be empty")
                if not all(isinstance(key, str) for key in v):
                    raise TypeError("All items in 'stratify_key' must be strings")
            else:
                raise TypeError("'stratify_key' must be a string or list of strings")

            return v

        @model_validator(mode="after")
        def validate_method_specific_fields(self):
            """Validate method-specific fields."""
            # FTS doesn't need embedding_model
            if (
                self.method == "fts"
                and self.embedding_model != "text-embedding-3-small"
            ):
                print("Warning: 'embedding_model' is ignored when method='fts'")

            # llm_compare specific validations
            if self.method == "llm_compare":
                # Needs model
                if not self.model:
                    raise ValueError(
                        "'model' must be specified when method='llm_compare'"
                    )

                # Doesn't support stratification (rank operates on entire dataset)
                if self.stratify_key:
                    raise ValueError(
                        "'stratify_key' is not supported with method='llm_compare'"
                    )

                # Doesn't support Jinja templates (needs consistent criteria)
                if "{{" in self.query or "}}" in self.query:
                    raise ValueError(
                        "'query' cannot contain Jinja templates ({{ }}) when method='llm_compare'. "
                        "The ranking criteria must be consistent across all documents."
                    )

            return self

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def execute(
        self, input_data: list[dict], is_build: bool = False
    ) -> tuple[list[dict], float]:
        """
        Execute the topk operation by delegating to appropriate operation.

        Args:
            input_data: List of dictionaries to retrieve from
            is_build: Whether this is a build phase execution

        Returns:
            Tuple of (top_k_items, cost)
        """
        # Handle llm_compare method using RankOperation
        if self.config["method"] == "llm_compare":
            return self._execute_llm_compare(input_data, is_build)

        # Handle embedding and fts methods using SampleOperation
        # Build sample configuration
        sample_config = {
            "name": self.config.get("name", "topk_operation"),
            "type": "sample",
            "method": f"top_{self.config['method']}",
            "samples": self.config["k"],
            "method_kwargs": {
                "keys": self.config["keys"],
                "query": self.config["query"],
            },
        }

        # Add stratification if specified
        if self.config.get("stratify_key"):
            sample_config["stratify_key"] = self.config["stratify_key"]
            # When stratifying, we want to retrieve top-k from each group
            sample_config["samples_per_group"] = True

        # Add embedding model for embedding method
        if self.config["method"] == "embedding":
            sample_config["method_kwargs"]["embedding_model"] = self.config.get(
                "embedding_model", "text-embedding-3-small"
            )

        # Add any additional config like random_state, bypass_cache
        if "random_state" in self.config:
            sample_config["random_state"] = self.config["random_state"]
        if "bypass_cache" in self.config:
            sample_config["bypass_cache"] = self.config["bypass_cache"]

        # Create and execute sample operation
        sample_op = SampleOperation(
            self.runner, sample_config, self.default_model, self.max_threads
        )

        return sample_op.execute(input_data, is_build)

    def _execute_llm_compare(
        self, input_data: list[dict], is_build: bool = False
    ) -> tuple[list[dict], float]:
        """
        Execute LLM-based comparison using RankOperation.

        Args:
            input_data: List of dictionaries to rank
            is_build: Whether this is a build phase execution

        Returns:
            Tuple of (top_k_items, cost)
        """
        # Build rank configuration
        rank_config = {
            "name": self.config.get("name", "topk_llm_operation"),
            "type": "order",
            "prompt": self.config["query"],  # Use query as the ranking criteria
            "input_keys": self.config["keys"],
            "direction": "desc",  # We want highest ranking items first for top-k
            "model": self.config["model"],
            "k": (
                self.config["k"]
                if isinstance(self.config["k"], int)
                else int(self.config["k"] * len(input_data))
            ),
            "initial_ordering_method": "embedding",
            "embedding_model": self.config.get(
                "embedding_model", "text-embedding-3-small"
            ),
            "batch_size": self.config.get("batch_size", 10),
            "rerank_call_budget": 100,  # Default budget for reranking
        }

        # Add optional configs
        if "bypass_cache" in self.config:
            rank_config["bypass_cache"] = self.config["bypass_cache"]
        if "timeout" in self.config:
            rank_config["timeout"] = self.config["timeout"]

        # Create and execute rank operation
        rank_op = RankOperation(
            self.runner, rank_config, self.default_model, self.max_threads
        )

        # Get ranked results
        ranked_results, cost = rank_op.execute(input_data)

        # Return only top k items
        k = self.config["k"]
        if isinstance(k, float):
            k = int(k * len(input_data))

        return ranked_results[:k], cost
