"""
The BaseOperation class is an abstract base class for all operations in the docetl framework. It provides a common structure and interface for various data processing operations.
"""

from abc import ABC, ABCMeta, abstractmethod

from pydantic import BaseModel, Field, field_validator
from rich.console import Console
from rich.status import Status

from docetl.console import DOCETL_CONSOLE


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

    def syntax_check(self) -> None:
        """Perform syntax checks on the operation configuration."""
        # Validate the configuration using Pydantic
        self.schema(**self.config)
