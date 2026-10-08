"""B1 Capture: single-slot, newest-wins buffer (SYSTEM_DESIGN.md B1).

A queue converts a throughput problem into permanently growing latency — the
downstream stage falls further and further behind, and the visitor watches the
system respond to what they signed five seconds ago. This buffer holds exactly one
item; `put` always overwrites, `get` always returns (and clears) the freshest one.
Dropping frames is correct behaviour here, not a bug.
"""
from __future__ import annotations

import threading
from typing import Generic, TypeVar

T = TypeVar("T")


class SingleSlotBuffer(Generic[T]):
    """Thread-safe newest-wins slot. Holds zero or one item, never more."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: T | None = None
        self._has_item = False
        self.dropped = 0        # frames overwritten before being read — observability

    def put(self, item: T) -> None:
        with self._lock:
            if self._has_item:
                self.dropped += 1
            self._item = item
            self._has_item = True

    def get(self) -> T | None:
        """Returns and clears the current item, or None if the slot is empty."""
        with self._lock:
            if not self._has_item:
                return None
            item = self._item
            self._item = None
            self._has_item = False
            return item

    def peek_empty(self) -> bool:
        with self._lock:
            return not self._has_item
