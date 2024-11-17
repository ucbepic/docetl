import os
import threading
from io import StringIO
from multiprocessing.util import DEFAULT_LOGGING_FORMAT
from typing import override, Optional, Union, Any

from rich.console import Console, JustifyMethod
from rich.style import Style

from docetl.helper.database import DatabaseUtil


class ThreadSafeConsole(Console):
    def __init__(self, *args, **kwargs):
        self.buffer = StringIO()
        kwargs["file"] = self.buffer
        super().__init__(*args, **kwargs)
        self.input_event = threading.Event()
        self.input_value = None
        self.conn: Optional[DatabaseUtil] = None
        self.is_write_to_db = False

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

    def with_db_logging_enabled(self, conn: DatabaseUtil, table_name: str, schema : dict) -> "ThreadSafeConsole":
        self.conn = conn
        self.is_write_to_db = True
        self.schema = schema
        self.table_name = table_name
        return self

    @override(Console.log)
    def log(self,
            *objects: Any,
            sep: str = " ",
            end: str = "\n",
            style: Optional[Union[str, Style]] = None,
            justify: Optional[JustifyMethod] = None,
            emoji: Optional[bool] = None,
            markup: Optional[bool] = None,
            highlight: Optional[bool] = None,
            log_locals: bool = False,
            _stack_offset: int = 1,
            ):
        # call super method
        super().log(*objects, sep=sep, end=end, style=style, justify=justify, emoji=emoji, markup=markup,
                    highlight=highlight, log_locals=log_locals, _stack_offset=_stack_offset)
        if self.is_write_to_db:
            self.conn.log_to_db(log_data=str(*objects), schema=self.schema, table_name=self.table_name)


class DocETLLog(Console):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conn: Optional[DatabaseUtil] = None
        self.is_write_to_db = False

    def with_db_logging_enabled(self, conn: DatabaseUtil, table_name: str, schema: dict) -> "DocETLLog":
        self.conn = conn
        self.is_write_to_db = True
        self.schema = schema
        self.table_name = table_name
        return self

    @override(Console.log)
    def log(
            self,
            *objects: Any,
            sep: str = " ",
            end: str = "\n",
            style: Optional[Union[str, Style]] = None,
            justify: Optional[JustifyMethod] = None,
            emoji: Optional[bool] = None,
            markup: Optional[bool] = None,
            highlight: Optional[bool] = None,
            log_locals: bool = False,
            _stack_offset: int = 1,
    ):
        # call super method
        super().log(*objects, sep=sep, end=end, style=style, justify=justify, emoji=emoji, markup=markup,
                    highlight=highlight, log_locals=log_locals, _stack_offset=_stack_offset)
        if self.is_write_to_db:
            #  this needs to be dictionary of the schema type
            #  user defined schema is causing troubles, strict schema then ?
            DatabaseUtil.DEFAULT_LOG_SCHEMA()
            self.conn.log_to_db(log_data=str(*objects), schema=self.schema, table_name=self.table_name)

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


# override log function to take in a sqlite database object and writes logs to the database


DOCETL_CONSOLE = get_console()
