from concurrent.futures import as_completed
from typing import Optional, Union, Iterable

from tqdm import tqdm


class RichLoopBar:
    """
    A progress bar class that integrates with Rich console.

    This class provides a wrapper around tqdm to create progress bars that work
    with Rich console output.

    Args:
        iterable (Optional[Union[Iterable, range]]): An iterable to track progress.
        total (Optional[int]): The total number of iterations.
        desc (Optional[str]): Description to be displayed alongside the progress bar.
        leave (bool): Whether to leave the progress bar on screen after completion.
        console: The Rich console object to use for output.
    """

    def __init__(
        self,
        iterable: Optional[Union[Iterable, range]] = None,
        total: Optional[int] = None,
        desc: Optional[str] = None,
        leave: bool = True,
        console=None,
    ):
        if console is None:
            raise ValueError("Console must be provided")
        self.console = console
        self.iterable = iterable
        self.total = self._get_total(iterable, total)
        self.description = desc
        self.leave = leave
        self.tqdm = None

    def _get_total(self, iterable, total):
        """
        Determine the total number of iterations for the progress bar.

        Args:
            iterable: The iterable to be processed.
            total: The explicitly specified total, if any.

        Returns:
            int or None: The total number of iterations, or None if it can't be determined.
        """
        if total is not None:
            return total
        if isinstance(iterable, range):
            return len(iterable)
        try:
            return len(iterable)
        except TypeError:
            return None

    def __iter__(self):
        """
        Create and return an iterator with a progress bar.

        Returns:
            Iterator: An iterator that yields items from the wrapped iterable.
        """
        self.tqdm = tqdm(
            self.iterable,
            total=self.total,
            desc=self.description,
            file=self.console.file,
        )
        for item in self.tqdm:
            yield item

    def __enter__(self):
        """
        Enter the context manager, initializing the progress bar.

        Returns:
            RichLoopBar: The RichLoopBar instance.
        """
        self.tqdm = tqdm(
            total=self.total,
            desc=self.description,
            leave=self.leave,
            file=self.console.file,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager, closing the progress bar.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
            exc_val: The instance of the exception that caused the context to be exited.
            exc_tb: A traceback object encoding the stack trace.
        """
        self.tqdm.close()

    def update(self, n=1):
        """
        Update the progress bar.

        Args:
            n (int): The number of iterations to increment the progress bar by.
        """
        if self.tqdm:
            self.tqdm.update(n)


def rich_as_completed(futures, total=None, desc=None, leave=True, console=None):
    """
    Yield completed futures with a Rich progress bar.

    This function wraps concurrent.futures.as_completed with a Rich progress bar.

    Args:
        futures: An iterable of Future objects to monitor.
        total (Optional[int]): The total number of futures.
        desc (Optional[str]): Description for the progress bar.
        leave (bool): Whether to leave the progress bar on screen after completion.
        console: The Rich console object to use for output.

    Yields:
        Future: Completed future objects.

    Raises:
        ValueError: If no console object is provided.
    """
    if console is None:
        raise ValueError("Console must be provided")

    with RichLoopBar(total=total, desc=desc, leave=leave, console=console) as pbar:
        for future in as_completed(futures):
            yield future
            pbar.update()
