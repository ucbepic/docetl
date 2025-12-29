import os
import re
import threading
import time
from io import StringIO

from rich.console import Console, RenderableType
from rich.status import Status
from rich.style import StyleType
from rich.traceback import install

from docetl.utils import StageType, get_stage_description

install(show_locals=False)

# ANSI escape sequences to strip (cursor movement, line clearing, cursor visibility)
ANSI_CURSOR_PATTERNS = re.compile(
    r"\x1b\[\?25[lh]"  # Hide/show cursor
    r"|\x1b\[\d*[ABCD]"  # Cursor up/down/forward/back
    r"|\x1b\[\d*[JK]"  # Clear screen/line (but we handle \x1b[2K specially)
    r"|\x1b\[s|\x1b\[u"  # Save/restore cursor position
)


def process_carriage_returns(text: str) -> str:
    """
    Process terminal control sequences for buffer-captured output.

    Rich's spinner uses ANSI sequences for:
    - `\r` - carriage return (move to start of line)
    - `\x1b[2K` - clear entire line
    - `\x1b[?25l/h` - hide/show cursor
    - `\x1b[1A` - move cursor up

    When captured to a buffer (not a real terminal), we simulate this by:
    1. Handling `\r` as "replace from start of line"
    2. Stripping cursor movement/visibility sequences
    3. Keeping only meaningful content
    """
    # First, strip cursor visibility and movement sequences
    text = ANSI_CURSOR_PATTERNS.sub("", text)

    # Process line by line
    lines = text.split("\n")
    processed_lines = []

    for line in lines:
        # Handle carriage returns - keep only content after the last \r
        if "\r" in line:
            segments = line.split("\r")
            # Keep the last non-empty segment (after stripping ANSI clear codes)
            for segment in reversed(segments):
                # Strip the clear line code if present
                cleaned = segment.replace("\x1b[2K", "").strip()
                if cleaned:
                    # Keep the segment but preserve any color codes
                    processed_lines.append(segment.replace("\x1b[2K", ""))
                    break
            else:
                # All segments were empty, skip this line entirely
                pass
        else:
            # No carriage return, keep the line as-is
            if line.strip() or line == "":  # Keep empty lines between content
                processed_lines.append(line)

    # Remove trailing empty lines
    while processed_lines and not processed_lines[-1].strip():
        processed_lines.pop()

    return "\n".join(processed_lines)


class ThreadSafeConsole(Console):
    def __init__(self, *args, **kwargs):
        self.buffer = StringIO()
        kwargs["file"] = self.buffer
        super().__init__(*args, **kwargs)
        self.input_event = threading.Event()
        self.input_value = None
        self.optimizer_statuses = []
        self.optimizer_rationale = None

    def get_output(self):
        """
        Get the output from the buffer, processing carriage returns.

        Rich's spinner uses carriage returns to overwrite lines in place.
        We process these to only keep the latest content, preventing
        duplicate spinner frames from flooding the output.
        """
        value = self.buffer.getvalue()
        self.buffer.truncate(0)
        self.buffer.seek(0)
        # Process carriage returns to handle spinner overwrites
        return process_carriage_returns(value)

    def status(
        self,
        status: "RenderableType",
        *,
        spinner: str = "dots",
        spinner_style: "StyleType" = "status.spinner",
        speed: float = 1.0,
        refresh_per_second: float = 4,
    ) -> "Status":
        """
        Return a Rich Status with animation.

        The carriage returns from the spinner animation are processed
        in get_output() to prevent duplicate lines.
        """
        status_renderable = Status(
            status,
            console=self,
            spinner=spinner,
            spinner_style=spinner_style,
            speed=speed,
            refresh_per_second=refresh_per_second,
        )
        return status_renderable

    def post_optimizer_rationale(
        self, should_optimize: bool, rationale: str, validator_prompt: str
    ):
        self.optimizer_rationale = (should_optimize, rationale, validator_prompt)

    def post_optimizer_status(self, stage: StageType):
        self.optimizer_statuses.append((stage, time.time()))

    def get_optimizer_progress(self) -> tuple[str, float]:
        if len(self.optimizer_statuses) == 0:
            return ("Optimization starting...", 0)

        if (
            len(self.optimizer_statuses) > 0
            and self.optimizer_statuses[-1][0] == StageType.END
        ):
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
            soft_wrap=True,
            highlight=False,
            log_path=False,
            color_system="truecolor",
            width=120,
            style="bright_white on black",
            record=True,
        )
    else:

        class NoOpConsole(Console):
            def post_optimizer_status(self, *args, **kwargs):
                pass

            def post_optimizer_rationale(self, *args, **kwargs):
                pass

        return NoOpConsole(log_path=False)


# Create the console first
DOCETL_CONSOLE = get_console()
