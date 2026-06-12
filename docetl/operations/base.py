"""
The BaseOperation class is an abstract base class for all operations in the docetl framework. It provides a common structure and interface for various data processing operations.
"""

import traceback
from abc import ABC, ABCMeta, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator
from rich.console import Console
from rich.status import Status

from docetl.console import DOCETL_CONSOLE


class Cardinality(str, Enum):
    """How an operation's output row count relates to its input row count.

    ONE_TO_ONE is an *at-most* contract, not an exactness guarantee: the
    operation intends one output per input, decides each row locally,
    never inserts or reorders rows — but may still drop a row on a
    per-row failure (an exhausted LLM timeout silently drops the row
    today rather than raising). That is sufficient for selection swaps
    (filter pushdown commutes with row-local drops: both orders keep
    exactly the rows that pass the predicate AND survive the op) but NOT
    for count- or position-sensitive rewrites like a positional head —
    which is why LimitPushdown is not a default rule.
    """

    ONE_TO_ONE = "one_to_one"  # at most one output row per input row,
    # decided row-locally; never inserts or reorders (see class docstring)
    SELECTION = "selection"  # a subset of the input rows (any added
    # annotation fields show up in fields_written)
    MANY_TO_ONE = "many_to_one"  # group-by style aggregation
    ONE_TO_MANY = "one_to_many"  # each input row expands to >= 0 rows
    MANY_TO_MANY = "many_to_many"  # anything else / unknown


class GleaningConfig(BaseModel):
    """Configuration for gleaning (iterative improvement) in operations."""

    num_rounds: int = Field(
        ..., gt=0, description="Number of gleaning rounds to perform"
    )
    validation_prompt: str = Field(
        ..., min_length=1, description="Prompt used to validate and improve outputs"
    )
    if_condition: str | None = Field(
        None,
        alias="if",
        description="Optional condition to determine when to perform gleaning",
    )

    @field_validator("validation_prompt")
    @classmethod
    def validate_validation_prompt(cls, v: str) -> str:
        """Ensure validation_prompt is not just whitespace."""
        if not v.strip():
            raise ValueError("'validation_prompt' cannot be empty or only whitespace")
        return v


class BaseOperationMeta(ABCMeta):
    def __new__(cls, *arg, **kw):
        self = ABCMeta.__new__(cls, *arg, **kw)
        self.schema.__name__ = self.__name__
        return self


class BaseOperation(ABC, metaclass=BaseOperationMeta):
    def __init__(
        self,
        runner,
        config: dict,
        default_model: str,
        max_threads: int,
        console: Console | None = None,
        status: Status | None = None,
        is_build: bool = False,
        **kwargs,
    ):
        """
        Initialize the BaseOperation.

        Args:
            config (dict): Configuration dictionary for the operation.
            default_model (str): Default language model to use.
            max_threads (int): Maximum number of threads for parallel processing.
            console (Console | None): Rich console for outputting logs. Defaults to None.
            status (Status | None): Rich status for displaying progress. Defaults to None.
        """
        assert "name" in config, "Operation must have a name"
        assert "type" in config, "Operation must have a type"
        self.runner = runner
        self.config = config
        self.default_model = default_model
        self.max_threads = max_threads
        self.console = console or DOCETL_CONSOLE
        self.manually_fix_errors = self.config.get("manually_fix_errors", False)
        self.status = status
        self.num_retries_on_validate_failure = self.config.get(
            "num_retries_on_validate_failure", 2
        )
        self.is_build = is_build
        self.bypass_cache = self.runner.config.get("bypass_cache", False)
        self.syntax_check()

    # This must be overridden in a subclass
    class schema(BaseModel, extra="allow"):
        name: str
        type: str
        skip_on_error: bool = False
        gleaning: GleaningConfig | None = None
        retriever: str | None = None

    @classmethod
    def transform_schema(
        cls, schema: dict[str, str], config: dict[str, Any]
    ) -> dict[str, str]:
        """Return the output schema after this operation runs on rows
        with *schema*, given its *config*.

        Best-effort and purely static — used for inspection (e.g.
        ``Frame.schema()``) without executing anything. The default merges
        the operation's declared ``output.schema`` and applies
        ``drop_keys``; operations with structural effects (split, unnest,
        gather, extract, ...) override this to declare the keys they add
        or reshape.
        """
        result = dict(schema)
        output_schema = (config.get("output") or {}).get("schema") or {}
        if isinstance(output_schema, dict):
            result.update(output_schema)
        for key in config.get("drop_keys") or []:
            result.pop(key, None)
        return result

    # ── plan traits ────────────────────────────────────────────────
    # Static, config-dependent semantics consumed by docetl.plan for
    # validation and equivalence-preserving rewrites. Like
    # transform_schema, these are classmethods so they work without
    # instantiating (and thus without executing) anything. The defaults
    # are maximally conservative: an operation that doesn't override
    # them can never qualify for a rewrite. Overrides must stay sound —
    # when a trait can't be determined from the config, return the
    # unknown/conservative value, never a guess.

    @classmethod
    def cardinality(cls, config: dict[str, Any]) -> Cardinality:
        """Output-vs-input row count relationship for this config."""
        return Cardinality.MANY_TO_MANY

    @classmethod
    def fields_read(cls, config: dict[str, Any]) -> "frozenset[str] | None":
        """Input fields this operation reads, or None if unknown (treat as
        the whole row)."""
        return None

    @classmethod
    def fields_written(cls, config: dict[str, Any]) -> "frozenset[str] | None":
        """Fields this operation adds, overwrites, or removes (``drop_keys``
        count), or None if unknown (treat as possibly any field)."""
        return None

    @classmethod
    def fields_removed(cls, config: dict[str, Any]) -> "frozenset[str]":
        """Fields this operation *definitely* removes from every row it
        emits (``drop_keys``, a filter's popped decision key). Used for
        the only sound missing-field check in an open world: reading a
        field that was definitely removed upstream."""
        return frozenset(config.get("drop_keys") or [])

    @classmethod
    def is_llm(cls, config: dict[str, Any]) -> bool:
        """Whether executing this config costs LLM calls."""
        return False

    @classmethod
    def is_row_local(cls, config: dict[str, Any]) -> bool:
        """Whether each output row depends only on its own input row
        (plus static config/external services keyed on that row)."""
        return False

    @classmethod
    def preserves_order(cls, config: dict[str, Any]) -> bool:
        """Whether output rows appear in input-row order."""
        return False

    @abstractmethod
    def execute(self, input_data: list[dict]) -> tuple[list[dict], float]:
        """
        Execute the operation on the input data.

        This method should be implemented by subclasses to perform the
        actual operation on the input data.

        Args:
            input_data (list[dict]): List of input data items.

        Returns:
            tuple[list[dict], float]: A tuple containing the processed data
            and the total cost of the operation.
        """
        pass

    def syntax_check(self, context: dict[str, Any] | None = None) -> None:
        """Perform syntax checks on the operation configuration."""
        # Validate the configuration using Pydantic
        self.schema.model_validate(self.config, context=context)

    def _maybe_build_retrieval_context(self, context: dict[str, Any]) -> str:
        """Build retrieval context string if a retriever is configured."""
        retriever_name = self.config.get("retriever")
        if not retriever_name:
            return ""
        retrievers = getattr(self.runner, "retrievers", {})
        if retriever_name not in retrievers:
            raise ValueError(
                f"Retriever '{retriever_name}' not found in configuration."
            )
        retriever = retrievers[retriever_name]
        try:
            result = retriever.retrieve(context)
            return result.rendered_context or ""
        except Exception as e:
            # Soft-fail to avoid blocking the op
            self.console.log(
                f"[yellow]Warning: retrieval failed for '{retriever_name}': {e}[/yellow]"
            )
            # Print traceback to help debug
            self.console.log(traceback.format_exc())
            return "No extra context available."
