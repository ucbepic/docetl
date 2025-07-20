from collections.abc import Iterable
from concurrent.futures import as_completed

from rich.console import Console
from tqdm import tqdm


class RichLoopBar:
    """A progress bar class that integrates with Rich console."""

    def __init__(
        self,
        iterable: Iterable | range | None = None,
        total: int | None = None,
        desc: str | None = None,
        leave: bool = True,
        console: Console | None = None,
    ) -> None:
        if console is None:
            raise ValueError("Console must be provided")
        self.console = console
        self.iterable = iterable
        self.total = self._get_total(iterable, total)
        self.description = desc
        self.leave = leave
        self.tqdm: tqdm | None = None

    def _get_total(
        self, iterable: Iterable | range | None, total: int | None
    ) -> int | None:
        if total is not None:
            return total
        if isinstance(iterable, range):
            return len(iterable)
        try:
            return len(iterable)
        except TypeError:
            return None

    def __iter__(self) -> Iterable:
        self.tqdm = tqdm(
            self.iterable,
            total=self.total,
            desc=self.description,
            file=self.console.file,
        )
        for item in self.tqdm:
            yield item

    def __enter__(self) -> "RichLoopBar":
        self.tqdm = tqdm(
            total=self.total,
            desc=self.description,
            leave=self.leave,
            file=self.console.file,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.tqdm.close()

    def update(self, n: int = 1) -> None:
        if self.tqdm:
            self.tqdm.update(n)


def rich_as_completed(
    futures,
    total: int | None = None,
    desc: str | None = None,
    leave: bool = True,
    console: Console | None = None,
) -> Iterable:
    """Yield completed futures with a Rich progress bar."""
    if console is None:
        raise ValueError("Console must be provided")

    with RichLoopBar(total=total, desc=desc, leave=leave, console=console) as pbar:
        for future in as_completed(futures):
            yield future
            pbar.update()
