"""What a request cost: simulation runs, wall time and model calls, counted while it runs.

A tracker is active for one request (Session.handle) and is shared by the threads it starts (asyncio.to_thread
copies the context, and the tracker object inside it is the same one). Batch worker processes are counted by the
parent when they are started, so nothing in a worker needs to know about it.
"""
import contextvars
import time
from contextlib import contextmanager
from typing import Iterator


class Cost:
    def __init__(self) -> None:
        self.runs = 0
        self.model_calls = 0
        self.started = time.perf_counter()

    def snapshot(self) -> dict:
        wall_s = round(time.perf_counter() - self.started, 1)
        return {"runs": self.runs, "wall_s": wall_s, "model_calls": self.model_calls,
                "text": describe(self.runs, wall_s, self.model_calls)}


_current: contextvars.ContextVar[Cost | None] = contextvars.ContextVar("simgate_cost", default=None)


@contextmanager
def track() -> Iterator[Cost]:
    """Count everything inside this block; inside another tracked block, keep adding to that one."""
    active = _current.get()
    if active is not None:
        yield active
        return
    cost = Cost()
    token = _current.set(cost)
    try:
        yield cost
    finally:
        _current.reset(token)


def current() -> Cost | None:
    return _current.get()


def add_runs(count: int) -> None:
    if (cost := _current.get()) is not None:
        cost.runs += count


def add_model_call() -> None:
    if (cost := _current.get()) is not None:
        cost.model_calls += 1


def describe(runs: int, wall_s: float, model_calls: int) -> str:
    """"12 runs · 48 s · 3 model calls"."""
    return (f"{runs} run{'' if runs == 1 else 's'} · {wall_s:.0f} s · "
            f"{model_calls} model call{'' if model_calls == 1 else 's'}")
