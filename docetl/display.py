"""Display helpers for query plans and execution summaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.markup import escape

from docetl.agents import get_agent_tool_names

if TYPE_CHECKING:
    from docetl.plan.ir import LogicalPlan


def format_query_plan(
    plan: "LogicalPlan",
    default_model: str = "?",
) -> tuple[dict[str, str], str]:
    from docetl.operations.utils.cascade_runner import (
        cascade_oracle_model,
        format_cascade_plan_lines,
    )
    from docetl.plan.ir import JoinNode, ScanNode

    colors = ["cyan", "magenta", "green", "yellow", "blue", "red"]
    step_colors = {
        step.name: colors[i % len(colors)] for i, step in enumerate(plan.steps)
    }

    printed: set[int] = set()

    def _fmt(
        node, step_name: str, indent: int = 0, label: str = "", tee: bool = False
    ) -> str:
        glyph = "├" if tee else "└"
        connector = f"{'  ' * (indent - 1)}[dim]{glyph} [/dim]" if indent else ""

        has_inputs = bool(node.inputs)
        s = "  " * indent
        guide = f"{s}[dim]│[/dim] " if has_inputs else f"{s}  "

        if isinstance(node, ScanNode):
            color = step_colors.get(step_name, "white")
            return f"{connector}[{color}][bold]scan_{node.dataset_name}[/bold][/{color}]  [dim]scan[/dim]{label}"

        if id(node) in printed:
            color = step_colors.get(step_name, "white")
            return f"{connector}[{color}][bold]{node.name}[/bold][/{color}]  [dim](shown above)[/dim]"
        printed.add(id(node))

        color = step_colors.get(step_name, "white")
        config = node.op_config
        lines = [
            f"{connector}[{color}][bold]{node.name}[/bold][/{color}]"
            f"  [cyan]{node.op_type}[/cyan]{label}",
        ]

        if "output" in config and "schema" in config["output"]:
            lines.append(f"{guide}[dim]output[/dim]")
            for field, field_type in config["output"]["schema"].items():
                lines.append(
                    f"{guide}  [bright_white]{field}[/bright_white]"
                    f" [dim]:[/dim] {escape(str(field_type))}"
                )

        agent_tools = get_agent_tool_names(config.get("agent"))
        if agent_tools:
            lines.append(f"{guide}[dim]agent tools[/dim]")
            for tool_name in agent_tools:
                lines.append(
                    f"{guide}  [bright_white]{escape(tool_name)}[/bright_white]"
                )

        if config.get("cascade"):
            oracle_model = cascade_oracle_model(config, default_model)
            lines.extend(
                f"{guide}{line}"
                for line in format_cascade_plan_lines(
                    config["cascade"],
                    op_type=config["type"],
                    oracle_model=oracle_model,
                )
            )

        if isinstance(node, JoinNode) and len(node.inputs) == 2:
            left_step = _step_for(node.inputs[0])
            right_step = _step_for(node.inputs[1])
            lines.append(
                _fmt(
                    node.inputs[0],
                    left_step,
                    indent + 1,
                    " [dim](left)[/dim]",
                    tee=True,
                )
            )
            lines.append(
                _fmt(node.inputs[1], right_step, indent + 1, " [dim](right)[/dim]")
            )
        else:
            for inp in node.inputs:
                inp_step = _step_for(inp)
                lines.append(_fmt(inp, inp_step, indent + 1))

        return "\n".join(lines)

    step_of_cache: dict[int, str] = {}
    for step in plan.steps:
        for n in step.nodes:
            step_of_cache[id(n)] = step.name

    def _step_for(node) -> str:
        return step_of_cache.get(id(node), "?")

    root = plan.root
    if root is None:
        return step_colors, "(empty plan)"

    root_step = _step_for(root)
    return step_colors, _fmt(root, root_step)


def format_execution_summary(
    total_cost: float,
    execution_time: float,
    token_usage: dict,
    intermediate_dir: str | None,
    output_path: str,
    cascade_roles: dict[str, str] | None = None,
) -> str:
    token_lines = ""
    if token_usage:
        token_lines = "\n[bold]Token Usage:[/bold]\n"
        total_prompt = 0
        total_completion = 0
        total_cached = 0
        for model, usage in sorted(token_usage.items()):
            prompt = usage["prompt_tokens"]
            completion = usage["completion_tokens"]
            cached = usage.get("cached_tokens", 0)
            total_prompt += prompt
            total_completion += completion
            total_cached += cached
            role = (cascade_roles or {}).get(model)
            model_label = f"{model} [dim]({role})[/dim]" if role else model
            line = f"  {model_label}: [cyan]{prompt:,}[/cyan] input"
            if cached:
                line += f" ([dim]{cached:,} cached[/dim])"
            line += f", [cyan]{completion:,}[/cyan] output"
            token_lines += line + "\n"
        if len(token_usage) > 1:
            total_line = f"  [bold]Total: [cyan]{total_prompt:,}[/cyan] input"
            if total_cached:
                total_line += f" ([dim]{total_cached:,} cached[/dim])"
            total_line += f", [cyan]{total_completion:,}[/cyan] output[/bold]"
            token_lines += total_line + "\n"

    return (
        f"Cost: [green]${total_cost:.2f}[/green]\n"
        f"Time: {execution_time:.2f}s\n"
        + token_lines
        + (f"Cache: [dim]{intermediate_dir}[/dim]\n" if intermediate_dir else "")
        + f"Output: [dim]{output_path}[/dim]"
    )


def strip_markup(text: str) -> str:
    """Strip Rich markup tags to produce plain text."""
    from rich.text import Text

    return Text.from_markup(text).plain
