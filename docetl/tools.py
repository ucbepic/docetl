"""Convenience helpers for common OpenAI Agents SDK tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

NetworkMode = Literal["disabled", "allowlist"]
MemoryLimit = Literal["1g", "4g", "16g", "64g"]


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
