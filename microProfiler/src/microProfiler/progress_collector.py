from __future__ import annotations

from typing import Callable, List


class ProgressCollector:
    """Base progress collector."""

    def report(self, step: str, current: int, total: int, message: str) -> None:
        pass

    def step_start(self, step: str, message: str = "") -> None:
        """Convenience: report start of a step (current=0, total=1)."""
        self.report(step, 0, 1, message)

    def step_end(self, step: str, message: str = "") -> None:
        """Convenience: report end of a step (current=1, total=1)."""
        self.report(step, 1, 1, message)


class NullProgressCollector(ProgressCollector):
    """No-op collector — use as default parameter so callers never check for None."""


class SubscribableProgressCollector(ProgressCollector):
    """Pub-sub collector: multiple listeners can attach/detach.

    Used by the GUI: the PipelineWorker attaches a progress-signal emitter,
    and the CLI attaches a print function.

    ``cancel_check`` is an optional ``Callable[[], bool]`` the worker sets so
    long-running black-box calls (e.g. microModel inference) can distinguish
    "aborted because the user cancelled" from "genuinely failed".
    """

    def __init__(self) -> None:
        self._subscribers: List[Callable[[str, int, int, str], None]] = []
        self.cancel_check: Callable[[], bool] | None = None

    def subscribe(self, fn: Callable[[str, int, int, str], None]) -> None:
        self._subscribers.append(fn)

    def report(self, step: str, current: int, total: int, message: str) -> None:
        for fn in self._subscribers:
            fn(step, current, total, message)
