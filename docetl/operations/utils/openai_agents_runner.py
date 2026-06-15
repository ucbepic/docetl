"""OpenAI Agents SDK bridge for agentic DocETL operations."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Awaitable
from typing import Any

from pydantic import BaseModel, create_model

from docetl.agents import Agent, AgentTool, Tool, as_tool, normalize_agent

_OPENAI_HOSTED_TOOL_TYPES = frozenset(
    {
        "CodeInterpreterTool",
        "FileSearchTool",
        "HostedMCPTool",
        "ImageGenerationTool",
        "ShellTool",
        "ToolSearchTool",
        "WebSearchTool",
    }
)


class AgentExecutionError(Exception):
    """Raised when an agentic operation cannot complete successfully."""


class AgentResult(dict):
    """Dictionary result with attached cost metadata for APIWrapper accounting."""

    _docetl_total_cost: float

    def __init__(self, output: dict[str, Any], total_cost: float):
        super().__init__(output)
        self._docetl_total_cost = total_cost


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
) -> AgentResult:
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
    output, cost = _run_async_safely(coro)
    return AgentResult(output, cost)


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
        from agents.models.openai_responses import OpenAIResponsesModel
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise AgentExecutionError(
            "Agentic operations require the OpenAI Agents SDK. Install "
            "`openai-agents[litellm]` to use docetl.Agent."
        ) from exc
    openai_client = AsyncOpenAI() if _has_hosted_tool(agent) else None
    tool_call_counter = {"count": 0}
    tools = _build_sdk_tools(
        agent=agent,
        counter=tool_call_counter,
        model=model,
        op_type=op_type,
        system_prompt=system_prompt,
        litellm_completion_kwargs=litellm_completion_kwargs,
        openai_agent_cls=OpenAIAgent,
        model_settings_cls=ModelSettings,
        litellm_model_cls=LitellmModel,
        openai_responses_model_cls=OpenAIResponsesModel,
        openai_client=openai_client,
    )
    output_type = _create_output_model(output_schema, scratchpad is not None)
    instructions = _build_agent_instructions(system_prompt, agent, output_schema)
    model_settings = _build_model_settings(
        ModelSettings, agent, litellm_completion_kwargs
    )
    sdk_model = _build_sdk_model(
        agent=agent,
        model=model,
        litellm_model_cls=LitellmModel,
        openai_responses_model_cls=OpenAIResponsesModel,
        openai_client=openai_client,
    )
    sdk_agent = OpenAIAgent(
        name=f"docetl_{op_type}_agent",
        instructions=instructions,
        model=sdk_model,
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


def _build_sdk_tools(
    *,
    agent: Agent,
    counter: dict[str, int],
    model: str,
    op_type: str,
    system_prompt: str,
    litellm_completion_kwargs: dict[str, Any],
    openai_agent_cls: type[Any],
    model_settings_cls: type[Any],
    litellm_model_cls: type[Any],
    openai_responses_model_cls: type[Any],
    openai_client: Any | None,
) -> list[Any]:
    tools: list[Any] = []
    for tool_item in agent.tools:
        if isinstance(tool_item, AgentTool):
            tools.append(
                _build_agent_tool(
                    agent_tool=tool_item,
                    model=model,
                    op_type=op_type,
                    system_prompt=system_prompt,
                    litellm_completion_kwargs=litellm_completion_kwargs,
                    openai_agent_cls=openai_agent_cls,
                    model_settings_cls=model_settings_cls,
                    litellm_model_cls=litellm_model_cls,
                    openai_responses_model_cls=openai_responses_model_cls,
                    openai_client=openai_client,
                )
            )
            continue
        if isinstance(tool_item, Tool) or callable(tool_item):
            tools.append(_build_function_tool(as_tool(tool_item), agent, counter))
            continue
        tools.append(tool_item)
    return tools


def _build_agent_tool(
    *,
    agent_tool: AgentTool,
    model: str,
    op_type: str,
    system_prompt: str,
    litellm_completion_kwargs: dict[str, Any],
    openai_agent_cls: type[Any],
    model_settings_cls: type[Any],
    litellm_model_cls: type[Any],
    openai_responses_model_cls: type[Any],
    openai_client: Any | None,
) -> Any:
    subagent = agent_tool.agent
    subagent_counter = {"count": 0}
    subagent_tools = _build_sdk_tools(
        agent=subagent,
        counter=subagent_counter,
        model=model,
        op_type=f"{op_type}_{agent_tool.name}",
        system_prompt=system_prompt,
        litellm_completion_kwargs=litellm_completion_kwargs,
        openai_agent_cls=openai_agent_cls,
        model_settings_cls=model_settings_cls,
        litellm_model_cls=litellm_model_cls,
        openai_responses_model_cls=openai_responses_model_cls,
        openai_client=openai_client,
    )
    instructions = _build_subagent_instructions(system_prompt, agent_tool)
    output_type = (
        _create_output_model(agent_tool.output_schema, False)
        if agent_tool.output_schema
        else None
    )
    sdk_model = _build_sdk_model(
        agent=subagent,
        model=model,
        litellm_model_cls=litellm_model_cls,
        openai_responses_model_cls=openai_responses_model_cls,
        openai_client=openai_client,
    )
    sdk_agent = openai_agent_cls(
        name=f"docetl_{op_type}_{agent_tool.name}",
        instructions=instructions,
        model=sdk_model,
        tools=subagent_tools,
        output_type=output_type,
        model_settings=_build_model_settings(
            model_settings_cls, subagent, litellm_completion_kwargs
        ),
    )
    return sdk_agent.as_tool(
        tool_name=agent_tool.name,
        tool_description=agent_tool.description,
        max_turns=agent_tool.max_turns or subagent.max_turns,
        run_config=_build_run_config(subagent),
    )


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
        "they help answer the user's task. Your final answer must be exactly one "
        "JSON object matching this DocETL schema, with no prose, Markdown, or "
        f"code fences: {schema_text}."
        f"{custom}"
    )


def _build_subagent_instructions(system_prompt: str, agent_tool: AgentTool) -> str:
    custom = (
        f"\n\nSpecialist instructions:\n{agent_tool.agent.instructions}"
        if agent_tool.agent.instructions
        else ""
    )
    schema_text = (
        "\n\nReturn exactly one JSON object matching this schema, with no prose, "
        "Markdown, or code fences: "
        + json.dumps(agent_tool.output_schema, sort_keys=True)
        if agent_tool.output_schema
        else ""
    )
    return (
        f"{system_prompt}\n\n"
        "You are a specialist agent invoked as a tool inside a DocETL operation. "
        "Complete the bounded task from the manager agent and return only the "
        "useful result for that task."
        f"{schema_text}"
        f"{custom}"
    )


def _build_sdk_model(
    *,
    agent: Agent,
    model: str,
    litellm_model_cls: type[Any],
    openai_responses_model_cls: type[Any],
    openai_client: Any | None,
) -> Any:
    if _has_direct_hosted_tool(agent):
        openai_model = _normalize_openai_model_name(model)
        if openai_client is None:
            raise AgentExecutionError(
                "OpenAI hosted tools require an OpenAI client, but none was created."
            )
        return openai_responses_model_cls(
            model=openai_model,
            openai_client=openai_client,
        )
    return litellm_model_cls(model=model)


def _has_hosted_tool(agent: Agent) -> bool:
    for tool_item in agent.tools:
        if isinstance(tool_item, AgentTool):
            if _has_hosted_tool(tool_item.agent):
                return True
            continue
        if type(tool_item).__name__ in _OPENAI_HOSTED_TOOL_TYPES:
            return True
    return False


def _has_direct_hosted_tool(agent: Agent) -> bool:
    return any(
        type(tool_item).__name__ in _OPENAI_HOSTED_TOOL_TYPES
        for tool_item in agent.tools
    )


def _normalize_openai_model_name(model: str) -> str:
    if model.startswith("openai/"):
        openai_model = model.split("/", 1)[1]
        if openai_model:
            return openai_model
        raise AgentExecutionError(
            "OpenAI hosted tools require a non-empty OpenAI model name."
        )
    if "/" in model:
        raise AgentExecutionError(
            "OpenAI hosted tools require an OpenAI Responses-compatible model. "
            f"Got LiteLLM model '{model}'. Use a plain OpenAI model name such as "
            "'gpt-4o-mini', or remove hosted OpenAI tools for non-OpenAI providers."
        )
    return model


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
