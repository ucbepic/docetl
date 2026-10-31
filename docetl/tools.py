"""Convenience helpers for common OpenAI Agents SDK tools."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, ClassVar, Literal

NetworkMode = Literal["disabled", "allowlist"]
MemoryLimit = Literal["1g", "4g", "16g", "64g"]


@dataclass(frozen=True)
class ParallelSearch:
    """Anonymous Parallel search and fetch tools, bound to each agent run."""

    timeout: float = 30.0
    name: ClassVar[str] = "parallel_search"

    def __post_init__(self) -> None:
        if not isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Parallel search timeout must be positive and finite")

    def cache_identity(self) -> dict[str, Any]:
        return {"kind": "parallel_search_mcp", "timeout": self.timeout}


def parallel_search(*, timeout: float = 30.0) -> ParallelSearch:
    """Enable keyless web search and page fetching for a DocETL agent.

    The agent receives ``parallel_search`` and ``parallel_fetch`` function tools.
    No connection is opened until a tool is called. Each agent run gets its own
    research session, even when this configuration is reused across rows.
    """
    return ParallelSearch(timeout=timeout)


@dataclass(frozen=True)
class Sandbox:
    """A persistent OpenAI hosted container for agent shell tools."""

    container_id: str

    @classmethod
    def create(
        cls,
        *,
        name: str,
        network: NetworkMode = "disabled",
        allowed_domains: list[str] | None = None,
        memory_limit: MemoryLimit | None = None,
        file_ids: list[str] | None = None,
        expires_after_minutes: int | None = 20,
    ) -> Sandbox:
        """Create a hosted container whose filesystem can be reused by tools."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "docetl.tools.Sandbox requires the OpenAI SDK. Install `openai` "
                "or `openai-agents[litellm]` to create hosted containers."
            ) from exc
        client = OpenAI()
        kwargs: dict[str, Any] = {
            "name": name,
            "network_policy": _build_network_policy(network, allowed_domains),
        }
        if memory_limit:
            kwargs["memory_limit"] = memory_limit
        if file_ids:
            kwargs["file_ids"] = file_ids
        if expires_after_minutes is not None:
            kwargs["expires_after"] = {
                "anchor": "last_active_at",
                "minutes": expires_after_minutes,
            }
        container = client.containers.create(**kwargs)
        return cls(container_id=container.id)

    def bash(self, *, name: str = "bash") -> Any:
        """Create a shell tool bound to this sandbox's persistent container."""
        return bash(name=name, container_id=self.container_id)

    def delete(self) -> None:
        """Delete the hosted container backing this sandbox."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "docetl.tools.Sandbox requires the OpenAI SDK. Install `openai` "
                "or `openai-agents[litellm]` to delete hosted containers."
            ) from exc
        OpenAI().containers.delete(self.container_id)


def bash(
    *,
    name: str = "bash",
    network: NetworkMode = "disabled",
    allowed_domains: list[str] | None = None,
    memory_limit: MemoryLimit | None = None,
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
    environment["network_policy"] = _build_network_policy(network, allowed_domains)
    return ShellTool(name=name, environment=environment)


def _build_network_policy(
    network: NetworkMode, allowed_domains: list[str] | None
) -> dict[str, Any]:
    if network == "allowlist":
        return {"type": "allowlist", "allowed_domains": allowed_domains or []}
    return {"type": "disabled"}
