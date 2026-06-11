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

    def _active_tracker(self):
        from docetl.progress.tracker import active_tracker

        return active_tracker()

    def __iter__(self) -> Iterable:
        tracker = self._active_tracker()
        if tracker is not None:
            # Interactive TUI run: it draws its own progress, so skip tqdm
            # entirely (tqdm's terminal/lock setup conflicts with Textual).
            tracker.set_phase(self.total)
            self.tqdm = None
            yield from self.iterable
            return
        self.tqdm = tqdm(
            self.iterable,
            total=self.total,
            desc=self.description,
            file=self.console.file,
        )
        for item in self.tqdm:
            yield item

    def __enter__(self) -> "RichLoopBar":
        tracker = self._active_tracker()
        if tracker is not None:
            tracker.set_phase(self.total)
            self.tqdm = None
            return self
        self.tqdm = tqdm(
            total=self.total,
            desc=self.description,
            leave=self.leave,
            file=self.console.file,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.tqdm is not None:
            self.tqdm.close()

    def set_description(self, desc: str) -> None:
        self.description = desc
        if self.tqdm is not None:
            self.tqdm.set_description(desc)

    def update(self, n: int = 1) -> None:
        if self.tqdm is not None:
            self.tqdm.update(n)
        # Feed the interactive progress tracker, if one is active for this run.
        # This is a no-op (and near-zero cost) outside of TUI runs.
        tracker = self._active_tracker()
        if tracker is not None:
            tracker.tick(n)


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
