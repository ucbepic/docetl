from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievalResult:
    """Container for retrieval outputs."""

    docs: list[dict]
    rendered_context: str
    meta: dict[str, Any]


class Retriever(ABC):
    """Abstract base class for retrievers."""

    name: str

    def __init__(self, runner, name: str, config: dict[str, Any]):
        self.runner = runner
        self.name = name
        self.config = config

    @abstractmethod
    def ensure_index(self) -> None:
        """Create or verify the underlying index/table."""
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, context: dict[str, Any]) -> RetrievalResult:
        """Execute retrieval based on the provided Jinja context."""
        raise NotImplementedError
