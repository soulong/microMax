"""Single progress mechanism shared by CLI (tqdm) and GUI (collector).

Every preprocessing/profiling step used to create a ``tqdm`` bar *and* call
``progress.report(...)`` in parallel — two systems that had to stay in sync.
``StepProgress`` wraps both behind one object: the tqdm bar renders to the
terminal for the CLI, while the collector feeds the GUI status bar. Both
receive identical ``(step, current, total, msg)`` values.
"""

from __future__ import annotations

from typing import Any, Optional

from microProfiler.progress_collector import ProgressCollector


class StepProgress:
    """Context manager wrapping a tqdm bar + a ProgressCollector.

    Usage::

        with StepProgress("Resize", total, progress, desc="Resizing", unit="img") as sp:
            for i in range(total):
                sp.report(i, "Resizing...")
                ... work ...
    """

    def __init__(
        self,
        step: str,
        total: int,
        progress: ProgressCollector,
        desc: Optional[str] = None,
        unit: str = "it",
        **tqdm_kwargs: Any,
    ) -> None:
        from tqdm import tqdm

        self._step = step
        self._total = total
        self._progress = progress
        self._bar = tqdm(total=total, desc=desc or step, unit=unit, **tqdm_kwargs)

    def report(self, current: int, msg: str = "") -> None:
        """Update the bar to ``current`` and forward to the collector."""
        delta = current - self._bar.n
        if delta:
            self._bar.update(delta)
        self._progress.report(self._step, current, self._total, msg)

    def tick(self, msg: str = "") -> None:
        """Advance the bar by one and forward the new count to the collector."""
        self._bar.update(1)
        self._progress.report(self._step, self._bar.n, self._total, msg)

    def finish(self, msg: str = "complete") -> None:
        """Force the bar to ``total`` and emit a final collector report."""
        if self._bar.n != self._total:
            self._bar.update(self._total - self._bar.n)
        self._progress.report(self._step, self._total, self._total, msg)

    def close(self) -> None:
        self._bar.close()

    def __enter__(self) -> "StepProgress":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self._bar.close()
        return False
