import os
import time
from typing import Any, Optional, Tuple
from rich.console import Console
from io import StringIO
import threading
import queue
from docetl.utils import StageType, get_stage_description

class ThreadSafeConsole(Console):
    def __init__(self, *args, **kwargs):
        self.buffer = StringIO()
        kwargs["file"] = self.buffer
        super().__init__(*args, **kwargs)
        self.input_event = threading.Event()
        self.input_value = None
        self.optimizer_statuses = []
        self.optimizer_rationale = None

    def status(
        self,
        status: "RenderableType",
        *,
        spinner: str = "dots",
        spinner_style: "StyleType" = "status.spinner",
        speed: float = 1.0,
        refresh_per_second: float = 12.5,
    ) -> "Status":
        from rich.status import Status

        status_renderable = Status(
            status,
            console=None,
            spinner=spinner,
            spinner_style=spinner_style,
            speed=speed,
            refresh_per_second=refresh_per_second,
        )
        return status_renderable
    
    def post_optimizer_rationale(self, should_optimize: bool, rationale: str, validator_prompt: str):
        self.optimizer_rationale = (should_optimize, rationale, validator_prompt)

    def post_optimizer_status(self, stage: StageType):
        self.optimizer_statuses.append((stage, time.time()))

    def get_optimizer_progress(self) -> Tuple[str, float]:
        if len(self.optimizer_statuses) == 0:
            return ("Optimization starting...", 0)
        
        if len(self.optimizer_statuses) > 0 and self.optimizer_statuses[-1][0] == StageType.END:
            return (get_stage_description(StageType.END), 1)

        num_stages = len(StageType) - 1
        num_completed = len([s for s in self.optimizer_statuses if s[1]]) - 1
        current_stage = self.optimizer_statuses[-1][0]
        return (get_stage_description(current_stage), num_completed / num_stages)

    def print(self, *args, **kwargs):
        super().print(*args, **kwargs)

    def input(
        self, prompt="", *, markup: bool = True, emoji: bool = True, **kwargs
    ) -> str:
        if prompt:
            self.print(prompt, markup=markup, emoji=emoji, end="")

        # TODO: Handle password

        self.input_event.wait()
        self.input_event.clear()
        return self.input_value

    def post_input(self, value: str):
        if self.input_event.is_set():
            super().print("Warning: Input ignored as we're not waiting for user input.")
            return
        self.input_value = value
        self.input_event.set()


def get_console():
    # Check if we're running with a frontend
    if os.environ.get("USE_FRONTEND") == "true":
        return ThreadSafeConsole(
            force_terminal=True,
            width=80,
            soft_wrap=True,
            highlight=False,
        )
    else:
        class NoOpConsole(Console):
            def post_optimizer_status(self, *args, **kwargs):
                pass

            def post_optimizer_rationale(self, *args, **kwargs):
                pass

        return NoOpConsole()


DOCETL_CONSOLE = get_console()
