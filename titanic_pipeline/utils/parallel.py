"""
Parallel processing utilities for Titanic ML Pipeline.
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ParallelProcessor:
    """Manages parallel execution of tasks using ProcessPoolExecutor."""

    def __init__(self, max_workers: int = None):
        """
        Initialize parallel processor.

        Args:
            max_workers: Maximum number of worker processes. If None, uses CPU count.
        """
        self.max_workers = max_workers
        self.executor = ProcessPoolExecutor(max_workers=max_workers)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(wait=True)

    def submit(self, fn, *args, **kwargs):
        """Submit a task to the executor."""
        return self.executor.submit(fn, *args, **kwargs)

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        """Map a function over iterables."""
        return self.executor.map(fn, *iterables, timeout=timeout, chunksize=chunksize)

    def shutdown(self, wait=True):
        """Shutdown the executor."""
        self.executor.shutdown(wait=wait)

    @property
    def as_completed(self):
        """Return as_completed function for convenience."""
        return as_completed
