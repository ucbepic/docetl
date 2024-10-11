import os
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

    def input(self, prompt: str = "") -> str:
        with StringIO() as buffer:
            self.print(prompt, end="", file=buffer)
        self.input_event.wait()
        self.input_event.clear()
        return self.input_value


def get_console():
    # Check if we're running with a frontend
    if os.environ.get("USE_FRONTEND") == "true":
        return ThreadSafeConsole(force_terminal=True, width=80, soft_wrap=True)
    else:
        return Console()


DOCETL_CONSOLE = get_console()
