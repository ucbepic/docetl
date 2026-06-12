"""Display helpers for query plans and execution summaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.markup import escape

if TYPE_CHECKING:
    from docetl.containers import OpContainer


def format_query_plan(
    root: OpContainer,
    op_container_map: dict[str, OpContainer],
    show_boundaries: bool = False,
    default_model: str = "?",
) -> tuple[dict[str, str], str]:
    from docetl.containers import StepBoundary
    from docetl.operations.utils.cascade_runner import (
        cascade_oracle_model,
        format_cascade_plan_lines,
    )

    step_boundaries = sorted(
        (op for op in op_container_map.values() if isinstance(op, StepBoundary)),
        key=lambda x: x.name,
    )
    colors = ["cyan", "magenta", "green", "yellow", "blue", "red"]
    step_colors = {
        b.step_name: colors[i % len(colors)] for i, b in enumerate(step_boundaries)
    }

    def _fmt(
        op: OpContainer, indent: int = 0, label: str = "", tee: bool = False
    ) -> str:
        glyph = "├" if tee else "└"
        connector = f"{'  ' * (indent - 1)}[dim]{glyph} [/dim]" if indent else ""

        s = "  " * indent
        guide = f"{s}[dim]│[/dim] " if op.children else f"{s}  "

        if isinstance(op, StepBoundary):
            if show_boundaries:
                color = step_colors.get(op.step_name, "white")
                lines = [
                    f"{connector}[{color}][bold]{op.name}[/bold][/{color}]"
                    f"  [dim]step_boundary[/dim]{label}",
                ]
                lines.extend(_fmt(c, indent + 1) for c in op.children)
                return "\n".join(lines)
            elif op.children:
                return _fmt(op.children[0], indent, label, tee)
            return ""

        color = step_colors.get(op.step_name, "white")
        lines = [
            f"{connector}[{color}][bold]{op.name}[/bold][/{color}]"
            f"  [cyan]{op.config['type']}[/cyan]{label}",
        ]

        if "output" in op.config and "schema" in op.config["output"]:
            lines.append(f"{guide}[dim]output[/dim]")
            for field, field_type in op.config["output"]["schema"].items():
                lines.append(
                    f"{guide}  [bright_white]{field}[/bright_white]"
                    f" [dim]:[/dim] {escape(str(field_type))}"
                )

        if op.config.get("cascade"):
            oracle_model = cascade_oracle_model(op.config, default_model)
            lines.extend(
                f"{guide}{line}"
                for line in format_cascade_plan_lines(
                    op.config["cascade"],
                    op_type=op.config["type"],
                    oracle_model=oracle_model,
                )
            )

        if op.children:
            if op.is_equijoin:
                lines.append(
                    _fmt(op.children[0], indent + 1, " [dim](left)[/dim]", tee=True)
                )
                lines.append(_fmt(op.children[1], indent + 1, " [dim](right)[/dim]"))
            else:
                lines.extend(_fmt(c, indent + 1) for c in op.children)

        return "\n".join(lines)

    return step_colors, _fmt(root)


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
