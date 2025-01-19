import hashlib
import json
from typing import Any, Dict, List

from outlines import generate, models
from pydantic import create_model

from .llm import InvalidOutputError, LLMResult


class OutlinesBackend:
    """Backend for handling Outlines (local) models in DocETL operations."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Outlines backend.

        Args:
            config: Optional configuration dictionary containing global settings
        """
        self._models = {}
        self._processors = {}
        self.config = config or {}

    def setup_model(self, model_path: str, output_schema: Dict[str, Any] = None):
        """Initialize Outlines model and processor if needed.

        Args:
            model_path: Path to the model, without the 'outlines/' prefix
            output_schema: Schema for the expected output
        """
        if model_path not in self._models:
            model_kwargs = {k: v for k, v in self.config.items() if k in ["max_tokens"]}
            self._models[model_path] = models.transformers(model_path, **model_kwargs)

            if output_schema:
                field_definitions = {
                    k: (eval(v) if isinstance(v, str) else v, ...)
                    for k, v in output_schema.items()
                }
                output_model = create_model("OutputModel", **field_definitions)
                self._processors[model_path] = generate.json(
                    self._models[model_path], output_model
                )

    def process_messages(
        self,
        model_path: str,
        messages: List[Dict[str, str]],
        output_schema: Dict[str, Any],
    ) -> LLMResult:
        """Process messages through Outlines model.

        Args:
            model_path: Path to the model, without the 'outlines/' prefix
            messages: List of message dictionaries with 'role' and 'content'
            output_schema: Schema for the expected output

        Returns:
            LLMResult containing the model's response in LiteLLM format
        """
        try:
            self.setup_model(model_path, output_schema)

            prompt = "\n".join(
                f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages
            )

            result = self._processors[model_path](prompt)

            response = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "send_output",
                                        "arguments": json.dumps(result.model_dump()),
                                    },
                                    "id": "call_"
                                    + hashlib.md5(
                                        json.dumps(result.model_dump()).encode()
                                    ).hexdigest(),
                                    "type": "function",
                                }
                            ],
                        },
                        "finish_reason": "stop",
                        "index": 0,
                    }
                ],
                "model": f"outlines/{model_path}",
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }

            return LLMResult(response=response, total_cost=0.0, validated=True)

        except Exception as e:
            import traceback

            traceback.print_exc()
            raise InvalidOutputError(
                message=str(e),
                output=str(e),
                expected_schema=output_schema,
                messages=messages,
            )
