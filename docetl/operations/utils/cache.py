import functools
import hashlib
import json
import os
import shutil
from typing import Dict, List

from diskcache import Cache
from dotenv import load_dotenv
from frozendict import frozendict
from rich.console import Console

from docetl.console import DOCETL_CONSOLE

load_dotenv()

DOCETL_HOME_DIR = (
    os.environ.get("DOCETL_HOME_DIR", os.path.expanduser("~")) + "/.cache/docetl"
)
CACHE_DIR = os.path.join(DOCETL_HOME_DIR, "general")
LLM_CACHE_DIR = os.path.join(DOCETL_HOME_DIR, "llm")
cache = Cache(LLM_CACHE_DIR)
cache.close()


def freezeargs(func):
    """
    Decorator to convert mutable dictionary arguments into immutable.
    """

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        args = tuple(
            (
                frozendict(arg)
                if isinstance(arg, dict)
                else json.dumps(arg) if isinstance(arg, list) else arg
            )
            for arg in args
        )
        kwargs = {
            k: (
                frozendict(v)
                if isinstance(v, dict)
                else json.dumps(v) if isinstance(v, list) else v
            )
            for k, v in kwargs.items()
        }
        return func(*args, **kwargs)

    return wrapped


def flush_cache(console: Console = DOCETL_CONSOLE):
    """Flush the cache to disk."""
    console.log("[bold green]Flushing cache to disk...[/bold green]")
    cache.close()
    console.log("[bold green]Cache flushed to disk.[/bold green]")


def clear_cache(console: Console = DOCETL_CONSOLE):
    """Clear the LLM cache stored on disk."""
    console.log("[bold yellow]Clearing LLM cache...[/bold yellow]")
    try:
        with cache as c:
            c.clear()
        # Remove all files in the cache directory
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        for filename in os.listdir(CACHE_DIR):
            file_path = os.path.join(CACHE_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                console.log(
                    f"[bold red]Error deleting {file_path}: {str(e)}[/bold red]"
                )
        console.log("[bold green]Cache cleared successfully.[/bold green]")
    except Exception as e:
        console.log(f"[bold red]Error clearing cache: {str(e)}[/bold red]")


def cache_key(
    model: str,
    op_type: str,
    messages: List[Dict[str, str]],
    output_schema: Dict[str, str],
    scratchpad: str = None,
    system_prompt: Dict[str, str] = None,
) -> str:
    """Generate a unique cache key based on function arguments."""
    key_dict = {
        "model": model,
        "op_type": op_type,
        "messages": json.dumps(messages, sort_keys=True),
        "output_schema": json.dumps(output_schema, sort_keys=True),
        "scratchpad": scratchpad,
        "system_prompt": json.dumps(system_prompt, sort_keys=True),
    }
    return hashlib.md5(json.dumps(key_dict, sort_keys=True).encode()).hexdigest()
