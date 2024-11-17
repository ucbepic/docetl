import hashlib
import json
import os
import shutil
from typing import List, Dict, Optional

from rich import Console

from docetl.console import DOCETL_CONSOLE
from docetl.helper.generic import cache, CACHE_DIR


def flush_cache(console: Console = DOCETL_CONSOLE):
    """
    Flush the cache to disk.
    """
    console.log("[bold green]Flushing cache to disk...[/bold green]")
    cache.close()
    console.log("[bold green]Cache flushed to disk.[/bold green]")


def clear_cache(console: Console = DOCETL_CONSOLE):
    """
    Clear the LLM cache stored on disk.

    This function removes all cached items from the disk-based cache,
    effectively clearing the LLM's response history.

    Args:
        console (Console, optional): A Rich console object for logging.
            Defaults to a new Console instance.
    """
    console.log("[bold yellow]Clearing LLM cache...[/bold yellow]")
    try:
        with cache as c:
            c.clear()
        # Remove all files in the cache directory
        cache_dir = CACHE_DIR
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        for filename in os.listdir(cache_dir):
            file_path = os.path.join(cache_dir, filename)
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
    scratchpad: Optional[str] = None,
) -> str:
    """
    Generate a unique cache key based on function arguments.

    This function creates a hash-based key using the input parameters, which can
    be used for caching purposes.

    Args:
        model (str): The model name.
        op_type (str): The operation type.
        messages (List[Dict[str, str]]): The messages to send to the LLM.
        output_schema (Dict[str, str]): The output schema dictionary.
        scratchpad (Optional[str]): The scratchpad to use for the operation.

    Returns:
        str: A unique hash string representing the cache key.
    """
    # Ensure no non-serializable objects are included
    key_dict = {
        "model": model,
        "op_type": op_type,
        "messages": json.dumps(messages, sort_keys=True),
        "output_schema": json.dumps(output_schema, sort_keys=True),
        "scratchpad": scratchpad,
    }
    return hashlib.md5(json.dumps(key_dict, sort_keys=True).encode()).hexdigest()
