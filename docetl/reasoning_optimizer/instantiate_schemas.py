from pydantic import BaseModel, Field
from typing import List, Dict
from pydantic import field_validator
import re

class MapOpConfig(BaseModel):
    """
    Configuration for a Map operator in a data processing chain.

    Attributes:
        name (str): The name of the Map operator.
        prompt (str): Jinja prompt template for the inserted Map operator. Must refer to the input document keys as {{ input.key }}.
        output_keys (List[str]): The keys of the output of the Map operator, to be referenced in the downstream operator's prompt.
        model (str): The model to use for the Map operator.
    """

    name: str = Field(..., description="The name of the Map operator")
    prompt: str = Field(
        ...,
        description="Jinja prompt template for the inserted Map operator. Must refer to the input document keys as {{ input.key }}."
    )
    output_keys: List[str] = Field(
        ...,
        description="The keys of the output of the Map operator, to be referenced in the downstream operator's prompt. Can be a single key or a list of keys. Can be new keys or existing keys from the map operator we are rewriting."
    )
    model: str = Field(
        default="gpt-4o-mini",
        description="The model to use for the Map operator."
    )

    @classmethod
    def validate_prompt_contains_input_key(cls, value: str) -> str:
        """
        Validator to ensure the prompt contains at least one {{ input.key }} reference.

        Args:
            value (str): The prompt string.

        Returns:
            str: The validated prompt string.

        Raises:
            ValueError: If no {{ input.key }} pattern is found in the prompt.
        """
        # Matches {{ input.key }} for any key (non-whitespace, non-})
        if not re.search(r"\{\{\s*input\.[^}\s]+\s*\}\}", value):
            raise ValueError("The prompt must contain at least one '{{ input.key }}' reference.")
        return value

    @field_validator("prompt")
    @classmethod
    def check_prompt(cls, v: str) -> str:
        return cls.validate_prompt_contains_input_key(v)
    
class ChainingInstantiateSchema(BaseModel):
    """
    Schema for chaining multiple Map operators in a data processing pipeline.

    Validates:
    - For each required input key, at least one MapOpConfig prompt must reference {{ input.key }}.
    - The output_keys of the final MapOpConfig must match the expected output_keys.
    """

    new_ops: List[MapOpConfig] = Field(
        ...,
        description="The new Map operators to insert in the chain."
    )

    @classmethod
    def validate_chain(
        cls,
        new_ops: List[MapOpConfig],
        required_input_keys: List[str],
        expected_output_keys: List[str]
    ) -> None:
        """
        Validates that for each required input key, at least one prompt in new_ops contains {{ input.key }},
        and that the final op's output_keys match expected_output_keys.

        Args:
            new_ops (List[MapOpConfig]): The list of MapOpConfig objects.
            required_input_keys (List[str]): The list of input keys that must be referenced in prompts.
            expected_output_keys (List[str]): The expected output keys for the final op.

        Raises:
            ValueError: If any required input key is not referenced in any prompt,
                        or if the final op's output_keys do not match expected_output_keys.
        """

        # Check each required input key is referenced in at least one prompt
        for key in required_input_keys:
            pattern = r"\{\{\s*input\." + re.escape(key) + r"\s*\}\}"
            if not any(re.search(pattern, op.prompt) for op in new_ops):
                raise ValueError(
                    f"Input key '{key}' must be referenced as '{{{{ input.{key} }}}}' in at least one prompt."
                )

        # Check that the final op's output_keys match expected_output_keys (order-insensitive)
        final_output_keys = list(new_ops[-1].output_keys) if new_ops else []
        if set(final_output_keys) != set(expected_output_keys):
            raise ValueError(
                f"The output_keys of the final op ({final_output_keys}) do not match the expected output_keys ({expected_output_keys})."
            )

class GleaningConfig(BaseModel):
    """
    Configuration for gleaning.

    Attributes:
        validation_prompt (str): Instructions for the LLM to evaluate and improve the output. The validation prompt doesn't need any variables, since it's appended to the chat thread.
        num_rounds (int): The maximum number of refinement iterations..
        model (str): The model to use for validation.
    """

    validation_prompt: str = Field(..., description="The prompt to evaluate and improve the output of the upstream operator.")
    num_rounds: int = Field(..., description="The maximum number of refinement iterations.")
    model: str = Field(default="gpt-4o-mini", description="The LLM model to use.")

class GleaningInstantiateSchema(BaseModel):
    """
    Schema for gleaning operations in a data processing pipeline.
    """

    gleaning_config: GleaningConfig = Field(
        ...,
        description="The gleaning configuration to apply to the target operation."
    )
