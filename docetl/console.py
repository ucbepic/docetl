import os
from typing import Optional
from rich.console import Console
from io import StringIO
import threading
import queue


class ThreadSafeConsole(Console):
    def __init__(self, *args, **kwargs):
        self.buffer = StringIO()
        kwargs["file"] = self.buffer
        super().__init__(*args, **kwargs)
        self.input_event = threading.Event()
        self.input_value = None

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
        return Console()


DOCETL_CONSOLE = get_console()
