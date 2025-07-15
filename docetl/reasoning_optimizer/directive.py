from copy import deepcopy
from pydantic import BaseModel, Field
from typing import Dict, List
import os
from litellm import completion
import re
from abc import ABC, abstractmethod

class Directive(BaseModel, ABC):
    name: str = Field(..., description="The name of the directive")
    formal_description: str = Field(..., description="A description of the directive; e.g., map => map -> map")
    nl_description: str = Field(..., description="An english description of the directive")
    when_to_use: str = Field(..., description="When to use the directive")
    instantiate_schema_type: BaseModel = Field(..., description="The schema the agent must conform to when instantiating the directive")
    example: str = Field(..., description="An example of the directive being used")

    def to_string_for_plan(self) -> str:
        """Serialize directive for prompts."""
        parts = [
            f"### {self.name}",
            f"**Format:** {self.formal_description}",
            f"**Description:** {self.nl_description}",
            f"**When to Use:** {self.when_to_use}",
        ]
        return "\n\n".join(parts)

    @abstractmethod
    def to_string_for_instantiate(self, *args, **kwargs) -> str:
        pass

    @abstractmethod
    def llm_instantiate(
        self,
        *args,
        **kwargs
    ):
        pass

    @abstractmethod
    def apply(self, *args, **kwargs) -> list:
        pass

    @abstractmethod
    def instantiate(self, *args, **kwargs) -> list:
        pass