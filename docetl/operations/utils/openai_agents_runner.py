"""OpenAI Agents SDK bridge for agentic DocETL operations."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Awaitable
from typing import Any

from pydantic import BaseModel, create_model

from docetl.agents import Agent, Tool, as_tool, normalize_agent


class AgentExecutionError(Exception):
    """Raised when an agentic operation cannot complete successfully."""


def run_openai_agent(
    *,
    runner: Any,
    model: str,
    op_type: str,
    messages: list[dict[str, Any]],
    output_schema: dict[str, Any],
    agent_config: Agent | dict[str, Any],
    system_prompt: str,
    scratchpad: str | None = None,
    litellm_completion_kwargs: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], float]:
    """Run an OpenAI Agents SDK agent backed by a LiteLLM model."""
    agent = normalize_agent(agent_config)
    prompt = _render_messages_for_agent(messages)
    if scratchpad:
        prompt = f"{prompt}\n\nCurrent scratchpad:\n{scratchpad}"
    coro = _run_openai_agent_async(
        runner=runner,
        model=model,
        op_type=op_type,
        prompt=prompt,
        output_schema=output_schema,
        agent=agent,
        system_prompt=system_prompt,
        scratchpad=scratchpad,
        litellm_completion_kwargs=litellm_completion_kwargs or {},
    )
    return _run_async_safely(coro)


async def _run_openai_agent_async(
    *,
    runner: Any,
    model: str,
    op_type: str,
    prompt: str,
    output_schema: dict[str, Any],
    agent: Agent,
    system_prompt: str,
    scratchpad: str | None,
    litellm_completion_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    try:
        from agents import Agent as OpenAIAgent
        from agents import ModelSettings, Runner
        from agents.extensions.models.litellm_model import LitellmModel
    except ImportError as exc:
        raise AgentExecutionError(
            "Agentic operations require the OpenAI Agents SDK. Install "
            "`openai-agents[litellm]` to use docetl.Agent."
        ) from exc
    tool_call_counter = {"count": 0}
    tools = _build_sdk_tools(agent, tool_call_counter)
    output_type = _create_output_model(output_schema, scratchpad is not None)
    instructions = _build_agent_instructions(system_prompt, agent, output_schema)
    model_settings = _build_model_settings(
        ModelSettings, agent, litellm_completion_kwargs
    )
    sdk_agent = OpenAIAgent(
        name=f"docetl_{op_type}_agent",
        instructions=instructions,
        model=LitellmModel(model=model),
        tools=tools,
        output_type=output_type,
        model_settings=model_settings,
    )
    result = await Runner.run(
        sdk_agent,
        prompt,
        max_turns=agent.max_turns,
        run_config=_build_run_config(agent),
    )
    if tool_call_counter["count"] > agent.max_tool_calls:
        raise AgentExecutionError(
            f"Agent exceeded max_tool_calls={agent.max_tool_calls}."
        )
    output = _coerce_final_output(result.final_output, output_schema)
    cost = _record_usage_and_cost(runner, model, result)
    return output, cost


def _build_sdk_tools(agent: Agent, counter: dict[str, int]) -> list[Any]:
    tools: list[Any] = []
    for tool_item in agent.tools:
        if isinstance(tool_item, Tool) or callable(tool_item):
            tools.append(_build_function_tool(as_tool(tool_item), agent, counter))
            continue
        if _is_legacy_tool_dict(tool_item):
            tools.append(
                _build_function_tool(_legacy_tool_to_tool(tool_item), agent, counter)
            )
            continue
        tools.append(tool_item)
    return tools


def _build_function_tool(tool: Tool, agent: Agent, counter: dict[str, int]) -> Any:
    from agents import FunctionTool

    async def invoke_tool(_context: Any, input_json: str) -> Any:
        counter["count"] += 1
        if counter["count"] > agent.max_tool_calls:
            raise AgentExecutionError(
                f"Agent exceeded max_tool_calls={agent.max_tool_calls}."
            )
        arguments = json.loads(input_json or "{}")
        try:
            result = tool.function(**arguments)
            if isinstance(result, Awaitable):
                result = await result
            return _json_safe(result)
        except Exception as exc:
            if not agent.continue_on_tool_error:
                raise
            return {"error": str(exc), "tool": tool.name}

    return FunctionTool(
        name=tool.name,
        description=tool.description,
        params_json_schema=tool.parameters,
        on_invoke_tool=invoke_tool,
        strict_json_schema=True,
        timeout_seconds=tool.timeout or agent.tool_timeout,
        timeout_behavior="error_as_result",
    )


def _legacy_tool_to_tool(tool_config: dict[str, Any]) -> Tool:
    function_config = tool_config["function"]
    function_name = function_config["name"]
    local_scope: dict[str, Any] = {}
    exec(tool_config["code"].strip(), {}, local_scope)
    function = local_scope[function_name]
    return Tool(
        function=function,
        name=function_name,
        description=function_config["description"],
        parameters=function_config["parameters"],
        timeout=tool_config.get("timeout"),
    )


def _is_legacy_tool_dict(tool_item: Any) -> bool:
    return (
        isinstance(tool_item, dict)
        and "code" in tool_item
        and isinstance(tool_item.get("function"), dict)
    )


def _create_output_model(
    output_schema: dict[str, Any], has_scratchpad: bool
) -> type[BaseModel]:
    fields: dict[str, tuple[Any, Any]] = {}
    for field_name, field_type in output_schema.items():
        fields[field_name] = (_schema_type_to_python_type(field_type), ...)
    if has_scratchpad:
        fields["updated_scratchpad"] = (str | None, None)
    return create_model("DocETLAgentOutput", **fields)


def _schema_type_to_python_type(value: Any) -> Any:
    if not isinstance(value, str):
        return Any
    normalized = value.strip().lower()
    if normalized in {"str", "string", "text", "varchar"}:
        return str
    if normalized in {"int", "integer"}:
        return int
    if normalized in {"float", "decimal", "number"}:
        return float
    if normalized in {"bool", "boolean"}:
        return bool
    if normalized.startswith("list[") and normalized.endswith("]"):
        inner = normalized[5:-1]
        return list[_schema_type_to_python_type(inner)]
    if normalized in {"dict", "object", "json"}:
        return dict[str, Any]
    return Any


def _build_agent_instructions(
    system_prompt: str, agent: Agent, output_schema: dict[str, Any]
) -> str:
    schema_text = json.dumps(output_schema, sort_keys=True)
    custom = (
        f"\n\nAdditional agent instructions:\n{agent.instructions}"
        if agent.instructions
        else ""
    )
    return (
        f"{system_prompt}\n\n"
        "You are running inside a DocETL operation. Use the available tools when "
        "they help answer the user's task. Return only the final structured "
        f"output matching this DocETL schema: {schema_text}."
        f"{custom}"
    )


def _build_model_settings(
    model_settings_cls: type[Any],
    agent: Agent,
    litellm_completion_kwargs: dict[str, Any],
) -> Any:
    settings = dict(agent.model_settings)
    extra_args = dict(settings.pop("extra_args", {}))
    for key, value in litellm_completion_kwargs.items():
        if key in {
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "max_tokens",
            "tool_choice",
            "parallel_tool_calls",
            "metadata",
            "store",
            "include_usage",
        }:
            settings.setdefault(key, value)
        else:
            extra_args[key] = value
    settings.setdefault("include_usage", True)
    if extra_args:
        settings["extra_args"] = extra_args
    return model_settings_cls(**settings)


def _build_run_config(agent: Agent) -> Any:
    if not agent.run_config:
        return None
    try:
        from agents import RunConfig
    except ImportError:
        return None
    return RunConfig(**agent.run_config)


def _render_messages_for_agent(messages: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, default=str)
        rendered.append(f"{role.upper()}:\n{content}")
    return "\n\n".join(rendered)


def _coerce_final_output(
    final_output: Any, output_schema: dict[str, Any]
) -> dict[str, Any]:
    if isinstance(final_output, BaseModel):
        return final_output.model_dump()
    if isinstance(final_output, dict):
        return final_output
    if isinstance(final_output, str):
        try:
            parsed = json.loads(final_output)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        if len(output_schema) == 1:
            return {next(iter(output_schema)): final_output}
    if hasattr(final_output, "__dict__"):
        return dict(final_output.__dict__)
    raise AgentExecutionError(f"Could not parse agent final output: {final_output!r}")


def _record_usage_and_cost(runner: Any, model: str, result: Any) -> float:
    usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
    if usage is None:
        return 0.0
    prompt_tokens = getattr(usage, "input_tokens", 0) or 0
    completion_tokens = getattr(usage, "output_tokens", 0) or 0
    runner.total_token_usage[model]["prompt_tokens"] += prompt_tokens
    runner.total_token_usage[model]["completion_tokens"] += completion_tokens
    cached_tokens = (
        getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0
    )
    if cached_tokens:
        runner.total_token_usage[model]["cached_tokens"] = (
            runner.total_token_usage[model].get("cached_tokens", 0) + cached_tokens
        )
    try:
        from litellm import cost_per_token

        prompt_cost, completion_cost = cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return float(prompt_cost + completion_cost)
    except Exception:
        return 0.0


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return json.loads(json.dumps(value, default=str))


def _run_async_safely(
    coro: Awaitable[tuple[dict[str, Any], float]],
) -> tuple[dict[str, Any], float]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    result: list[tuple[dict[str, Any], float] | BaseException] = []

    def run_in_thread() -> None:
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:
            result.append(exc)

    thread = threading.Thread(target=run_in_thread)
    thread.start()
    thread.join()
    if result and isinstance(result[0], BaseException):
        raise result[0]
    if not result:
        raise AgentExecutionError("Agent execution thread finished without a result.")
    return result[0]
