"""
Job Monitor for quantum data pipelines.

Tracks execution progress with real-time status, ETA estimation,
and circuit execution rate metrics.
"""

import time
import sys
from typing import Optional


class JobMonitor:
    """
    Real-time progress tracker for Quantum Data Pipelines.
    
    Displays a visual progress bar with:
    - Completion percentage
    - Current / total items
    - Execution rate (circuits/second)
    - Estimated time remaining (ETA)
    
    Useful for long-running quantum jobs (e.g., 4+ hour runs on real QPUs).
    
    Parameters
    ----------
    total_items : int
        Total number of items to process.
    description : str, optional
        Description of the job. Default: "Processing"
        
    Attributes
    ----------
    current : int
        Current progress (items processed).
    total : int
        Total items to process.
    start_time : float
        Timestamp when monitor was created.
        
    Examples
    --------
    >>> monitor = JobMonitor(1000, description="Executing Circuits")
    >>> for i in range(1000):
    ...     # Process circuit...
    ...     monitor.update(1)
    
    Using context manager:
    
    >>> with JobMonitor(500, description="Batch Processing") as monitor:
    ...     for i, circuit in enumerate(circuits):
    ...         result = backend.run(circuit).result()
    ...         monitor.update(1)
    
    Notes
    -----
    The progress bar updates in-place on a single line.
    Progress information is cleared when job completes.
    """
    
    def __init__(
        self,
        total_items: int,
        description: str = "Processing"
    ):
        """Initialize Job Monitor."""
        if total_items <= 0:
            raise ValueError(f"total_items must be positive, got {total_items}")
        
        self.total = total_items
        self.desc = description
        self.current = 0
        self.start_time = time.time()
        
        print(f"{description} ({total_items} items)")

    def update(self, n: int = 1) -> None:
        """
        Update progress by n items.
        
        Parameters
        ----------
        n : int, optional
            Number of items to advance. Default: 1
            
        Examples
        --------
        >>> monitor.update(10)
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        
        self.current = min(self.current + n, self.total)
        self._print_bar()

    def _print_bar(self) -> None:
        """
        Print progress bar to stdout.
        
        Format: |███████████-----| 70.0% [700/1000] (Rate: 10.5 circ/s, ETA: 28s)
        """
        elapsed = time.time() - self.start_time
        percent = self.current / self.total if self.total > 0 else 0
        
        if elapsed > 0:
            rate = self.current / elapsed
        else:
            rate = 0
        
        if rate > 0:
            remaining_items = self.total - self.current
            eta_seconds = remaining_items / rate
        else:
            eta_seconds = 0
        
        bar_length = 30
        filled_length = int(bar_length * percent)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
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
            print(f"Complete in {elapsed_total:.1f}s")

    def reset(self) -> None:
        """
        Reset progress counter to 0.
        
        Useful for reusing monitor for multiple jobs.
        """
        self.current = 0
        self.start_time = time.time()
        print(f"Progress reset")

    def close(self) -> None:
        """
        Close the monitor (print final newline if incomplete).
        """
        if self.current < self.total:
            sys.stdout.write("\n")

    def get_elapsed(self) -> float:
        """
        Get elapsed time in seconds since monitor creation.
        
        Returns
        -------
        float
            Elapsed seconds.
        """
        return time.time() - self.start_time

    def get_rate(self) -> float:
        """
        Get processing rate (items per second).
        
        Returns
        -------
        float
            Items processed per second.
        """
        elapsed = self.get_elapsed()
        if elapsed > 0:
            return self.current / elapsed
        return 0.0

    def get_eta(self) -> float:
        """
        Get estimated time remaining in seconds.
        
        Returns
        -------
        float
            Seconds until completion (or 0 if complete).
        """
        rate = self.get_rate()
        if rate > 0:
            remaining_items = self.total - self.current
            return remaining_items / rate
        return 0.0

    def get_status(self) -> dict:
        """
        Get current monitor status as a dictionary.
        
        Returns
        -------
        dict
            Status information including progress, rate, ETA.
            
        Examples
        --------
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
            "is_complete": self.current >= self.total
        }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __str__(self) -> str:
        """String representation of monitor status."""
        status = self.get_status()
        return (
            f"{self.desc}: {status['current']}/{status['total']} "
            f"({status['percent']:.1%}, "
            f"Rate: {status['rate_items_per_second']:.1f} items/s, "
            f"ETA: {status['eta_seconds']:.0f}s)"
        )
