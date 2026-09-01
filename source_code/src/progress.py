"""Small helpers for consistent progress messages in the main scripts."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from time import perf_counter
from typing import Iterator


def _current_time() -> str:
    """Return the local time used as a short console prefix."""
    return datetime.now().strftime("%H:%M:%S")


def log_start(title: str) -> None:
    """Print the start message for a script or pipeline."""
    print(f"\n[{_current_time()}] Start: {title}", flush=True)


def log_end(title: str) -> None:
    """Print the end message for a script or pipeline."""
    print(f"[{_current_time()}] Finished: {title}", flush=True)


@contextmanager
def step(
    title: str,
    *,
    number: int | None = None,
    total: int | None = None,
    detail: str | None = None,
) -> Iterator[None]:
    """Wrap a work block and report start, duration, and errors."""
    step_label = ""
    if number is not None and total is not None:
        step_label = f"Step {number}/{total} - "

    print(f"[{_current_time()}] {step_label}{title}", flush=True)
    if detail:
        print(f"    {detail}", flush=True)

    start = perf_counter()
    try:
        yield
    except Exception as exc:
        duration = perf_counter() - start
        print(f"[{_current_time()}] Error in step: {title} ({duration:.1f}s)", flush=True)
        print(f"    {type(exc).__name__}: {exc}", flush=True)
        raise

    duration = perf_counter() - start
    print(f"[{_current_time()}] Done: {title} ({duration:.1f}s)", flush=True)

