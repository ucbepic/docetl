"""Convenience helpers for common OpenAI Agents SDK tools."""

from __future__ import annotations

from typing import Any, Literal


def bash(
    *,
    name: str = "bash",
    network: Literal["disabled", "allowlist"] = "disabled",
    allowed_domains: list[str] | None = None,
    memory_limit: Literal["1g", "4g", "16g", "64g"] | None = None,
    file_ids: list[str] | None = None,
    container_id: str | None = None,
) -> Any:
    """Create a hosted sandbox shell tool for DocETL agents.

    Reuse the returned tool object across manager and specialist agents when
    they should work in the same sandbox tool environment.
    """
    try:
        from agents import ShellTool
    except ImportError as exc:
        raise ImportError(
            "docetl.tools.bash requires the OpenAI Agents SDK. Install "
            "`openai-agents[litellm]` to use hosted shell tools."
        ) from exc
    if container_id:
        return ShellTool(
            name=name,
            environment={"type": "container_reference", "container_id": container_id},
        )
    environment: dict[str, Any] = {"type": "container_auto"}
    if file_ids:
        environment["file_ids"] = file_ids
    if memory_limit:
        environment["memory_limit"] = memory_limit
    if network == "allowlist":
        environment["network_policy"] = {
            "type": "allowlist",
            "allowed_domains": allowed_domains or [],
        }
    else:
        environment["network_policy"] = {"type": "disabled"}
    return ShellTool(name=name, environment=environment)
