"""
Job monitoring for quantum data pipelines.

Provides a real-time progress tracker with visual progress bar,
execution-rate metrics, and ETA estimation for long-running
quantum jobs.
"""

import logging
import sys
import time
from typing import Optional

logger = logging.getLogger(__name__)


class JobMonitor:
    """Real-time progress tracker for quantum data pipelines.

    Displays a visual progress bar with:

    * Completion percentage.
    * Current / total items processed.
    * Execution rate (items per second).
    * Estimated time remaining (ETA).

    Supports use as a context manager for automatic cleanup.

    Args:
        total_items: Total number of items to process.
        description: Human-readable job description.

    Attributes:
        current: Number of items processed so far.
        total: Total items to process.
        start_time: Timestamp (``time.time()``) when the monitor was created.

    Examples:
        >>> monitor = JobMonitor(1000, description="Executing Circuits")
        >>> for i in range(1000):
        ...     # Process circuit …
        ...     monitor.update(1)

        Using as a context manager:

        >>> with JobMonitor(500, description="Batch Processing") as monitor:
        ...     for circuit in circuits:
        ...         result = backend.run(circuit).result()
        ...         monitor.update(1)

    Notes:
        The progress bar updates in-place on a single line using
        ``\\r`` carriage return.  A final newline is printed when
        the job completes or the monitor is closed.
    """

    def __init__(
        self,
        total_items: int,
        description: str = "Processing",
    ) -> None:
        """Initialize Job Monitor.

        Args:
            total_items: Total items to track (must be positive).
            description: Short description shown in the progress bar.

        Raises:
            ValueError: If *total_items* is not positive.
        """
        if total_items <= 0:
            raise ValueError(f"total_items must be positive, got {total_items}")

        self.total = total_items
        self.desc = description
        self.current = 0
        self.start_time = time.time()

        logger.info("%s (%d items)", description, total_items)

    def update(self, n: int = 1) -> None:
        """Advance the progress counter by *n* items.

        Args:
            n: Number of items completed since the last call.  Default: 1.

        Raises:
            ValueError: If *n* is not positive.
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")

        self.current = min(self.current + n, self.total)
        self._print_bar()

    def _print_bar(self) -> None:
        """Render the progress bar to stdout.

        Format::

            |███████████-----| 70.0% [700/1000] (Rate: 10.5 circ/s, ETA: 28s)
        """
        elapsed = time.time() - self.start_time
        percent = self.current / self.total if self.total > 0 else 0

        rate = self.current / elapsed if elapsed > 0 else 0

        if rate > 0:
            remaining_items = self.total - self.current
            eta_seconds = remaining_items / rate
        else:
            eta_seconds = 0

        bar_length = 30
        filled_length = int(bar_length * percent)
        bar = "█" * filled_length + "-" * (bar_length - filled_length)

        status = (
            f"\r{self.desc}: |{bar}| {percent:.1%} "
            f"[{self.current}/{self.total}] "
            f"(Rate: {rate:.1f} circ/s, ETA: {eta_seconds:.0f}s)"
        )

        sys.stdout.write(status)
        sys.stdout.flush()

        if self.current >= self.total:
            sys.stdout.write("\n")
            elapsed_total = time.time() - self.start_time
            logger.info("Complete in %.1fs", elapsed_total)

    def reset(self) -> None:
        """Reset the progress counter to zero.

        Useful for reusing the same monitor across multiple jobs.
        """
        self.current = 0
        self.start_time = time.time()
        logger.debug("Progress reset for '%s'.", self.desc)

    def close(self) -> None:
        """Close the monitor, printing a final newline if incomplete."""
        if self.current < self.total:
            sys.stdout.write("\n")

    def get_elapsed(self) -> float:
        """Get elapsed time in seconds since monitor creation.

        Returns:
            Elapsed seconds.
        """
        return time.time() - self.start_time

    def get_rate(self) -> float:
        """Get the processing rate (items per second).

        Returns:
            Items processed per second (0.0 if no time has elapsed).
        """
        elapsed = self.get_elapsed()
        if elapsed > 0:
            return self.current / elapsed
        return 0.0

    def get_eta(self) -> float:
        """Get the estimated time remaining in seconds.

        Returns:
            Seconds until completion (0.0 if already complete or rate
            is zero).
        """
        rate = self.get_rate()
        if rate > 0:
            return (self.total - self.current) / rate
        return 0.0

    def get_status(self) -> dict:
        """Get the current monitor status as a dictionary.

        Returns:
            Dictionary with keys: ``current``, ``total``, ``percent``,
            ``elapsed_seconds``, ``rate_items_per_second``,
            ``eta_seconds``, ``is_complete``.

        Examples:
            >>> status = monitor.get_status()
            >>> print(f"Progress: {status['percent']:.1%}")
            >>> print(f"ETA: {status['eta_seconds']:.0f}s")
        """
        elapsed = self.get_elapsed()
        rate = self.get_rate()
        eta = self.get_eta()

        return {
            "current": self.current,
            "total": self.total,
            "percent": self.current / self.total if self.total > 0 else 0,
            "elapsed_seconds": elapsed,
            "rate_items_per_second": rate,
            "eta_seconds": eta,
            "is_complete": self.current >= self.total,
        }

    def __enter__(self) -> "JobMonitor":
        """Enter the context manager.

        Returns:
            self
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager, closing the monitor."""
        self.close()

    def __str__(self) -> str:
        """Return a human-readable string representation of the monitor status."""
        status = self.get_status()
        return (
            f"{self.desc}: {status['current']}/{status['total']} "
            f"({status['percent']:.1%}, "
            f"Rate: {status['rate_items_per_second']:.1f} items/s, "
            f"ETA: {status['eta_seconds']:.0f}s)"
        )
