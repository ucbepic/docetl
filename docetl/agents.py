"""Agent and tool helpers for Python-only agentic DocETL operations."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Union, get_args, get_origin


JsonSchema = dict[str, Any]


@dataclass(frozen=True)
class Tool:
    """A Python callable exposed to an agent as a structured function tool."""

    function: Callable[..., Any]
    name: str
    description: str
    parameters: JsonSchema
    timeout: float | None = None

    def cache_identity(self) -> dict[str, Any]:
        """Return a stable identity for caching and hashing operation configs."""
        return {
            "kind": "docetl_tool",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "source_hash": _get_callable_hash(self.function),
        }


@dataclass(frozen=True)
class Agent:
    """Configuration for an agentic DocETL operation.

    The LiteLLM model is specified on the operation (`model=`) or through
    `docetl.default_model`; the agent defines tools and loop controls.
    """

    tools: list[Any] = field(default_factory=list)
    max_turns: int = 5
    max_tool_calls: int = 20
    tool_timeout: float | None = 30.0
    continue_on_tool_error: bool = True
    cache: bool = False
    instructions: str | None = None
    model_settings: dict[str, Any] = field(default_factory=dict)
    run_config: dict[str, Any] = field(default_factory=dict)

    def cache_identity(self) -> dict[str, Any]:
        """Return a stable identity for caching and hashing operation configs."""
        return {
            "kind": "docetl_agent",
            "max_turns": self.max_turns,
            "max_tool_calls": self.max_tool_calls,
            "tool_timeout": self.tool_timeout,
            "continue_on_tool_error": self.continue_on_tool_error,
            "cache": self.cache,
            "instructions": self.instructions,
            "model_settings": self.model_settings,
            "run_config": self.run_config,
            "tools": [_get_tool_identity(tool_item) for tool_item in self.tools],
        }


def as_tool(
    function: Callable[..., Any] | Tool,
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: JsonSchema | None = None,
    timeout: float | None = None,
) -> Tool:
    """Convert a typed Python callable into a DocETL tool."""
    if isinstance(function, Tool):
        return function
    tool_name = name or function.__name__
    tool_description = description or inspect.getdoc(function) or tool_name
    tool_parameters = parameters or _get_parameters_schema(function)
    return Tool(
        function=function,
        name=tool_name,
        description=tool_description,
        parameters=tool_parameters,
        timeout=timeout,
    )


def tool(
    function: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: JsonSchema | None = None,
    timeout: float | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Decorate or wrap a Python callable as a DocETL agent tool."""
    if function is not None:
        return as_tool(
            function,
            name=name,
            description=description,
            parameters=parameters,
            timeout=timeout,
        )

    def decorate(inner: Callable[..., Any]) -> Tool:
        return as_tool(
            inner,
            name=name,
            description=description,
            parameters=parameters,
            timeout=timeout,
        )

    return decorate


def normalize_agent(agent_config: Agent | dict[str, Any]) -> Agent:
    """Normalize user-provided agent configuration into an Agent instance."""
    if isinstance(agent_config, Agent):
        return agent_config
    if isinstance(agent_config, dict):
        return Agent(**agent_config)
    raise TypeError("agent must be a docetl.Agent or a dictionary of Agent fields")


def get_agent_tool_names(agent_config: Agent | dict[str, Any] | None) -> list[str]:
    """Return display names for tools configured on an agent."""
    if agent_config is None:
        return []
    agent = normalize_agent(agent_config)
    return [_get_tool_name(tool_item) for tool_item in agent.tools]


def _get_parameters_schema(function: Callable[..., Any]) -> JsonSchema:
    signature = inspect.signature(function)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, parameter in signature.parameters.items():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        annotation = (
            str
            if parameter.annotation is inspect.Parameter.empty
            else parameter.annotation
        )
        properties[name] = _annotation_to_json_schema(annotation)
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _annotation_to_json_schema(annotation: Any) -> JsonSchema:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if annotation is Any or annotation is inspect.Parameter.empty:
        return {}
    if origin is Union:
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            schema = _annotation_to_json_schema(non_none_args[0])
            schema["nullable"] = True
            return schema
        return {"anyOf": [_annotation_to_json_schema(arg) for arg in non_none_args]}
    if origin is Literal:
        values = list(args)
        value_types = {type(value) for value in values}
        schema_type = "string"
        if value_types <= {int}:
            schema_type = "integer"
        elif value_types <= {float, int}:
            schema_type = "number"
        elif value_types <= {bool}:
            schema_type = "boolean"
        return {"type": schema_type, "enum": values}
    if annotation in {str, "str", "string"}:
        return {"type": "string"}
    if annotation in {int, "int", "integer"}:
        return {"type": "integer"}
    if annotation in {float, "float", "number"}:
        return {"type": "number"}
    if annotation in {bool, "bool", "boolean"}:
        return {"type": "boolean"}
    if origin in {list, tuple, set}:
        item_annotation = args[0] if args else Any
        return {"type": "array", "items": _annotation_to_json_schema(item_annotation)}
    if origin is dict:
        return {"type": "object", "additionalProperties": True}
    return {"type": "string"}


def _get_callable_hash(function: Callable[..., Any]) -> str:
    try:
        source = inspect.getsource(function)
    except Exception:
        source = repr(function)
    signature = str(inspect.signature(function))
    payload = f"{function.__module__}.{function.__qualname__}{signature}\n{source}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _get_tool_identity(tool_item: Any) -> Any:
    if hasattr(tool_item, "cache_identity"):
        return tool_item.cache_identity()
    if callable(tool_item):
        return as_tool(tool_item).cache_identity()
    try:
        json.dumps(tool_item, sort_keys=True)
        return tool_item
    except TypeError:
        return repr(tool_item)


def _get_tool_name(tool_item: Any) -> str:
    if isinstance(tool_item, Tool):
        return tool_item.name
    if callable(tool_item):
        return getattr(tool_item, "__name__", repr(tool_item))
    name = getattr(tool_item, "name", None)
    if isinstance(name, str) and name:
        return name
    return type(tool_item).__name__
