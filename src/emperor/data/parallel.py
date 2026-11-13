"""Running one item's work on several threads at once (hosted APIs serve requests in parallel).

Results are handled in the calling thread, in completion order, so shard writes and counters
need no extra locking. A `TeacherError` (server gone, budget reached) stops new work; work
already started finishes and is kept.
"""

from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

from emperor.data.teacher import TeacherError


def run_items[T, R](
    items: Iterable[T],
    work: Callable[[T], R],
    on_result: Callable[[T, R], None],
    *,
    workers: int = 1,
    should_stop: Callable[[], bool] = lambda: False,
) -> dict:
    """Returns {"stopped": bool, "error": str} (error only when a TeacherError stopped it)."""
    pending = iter(items)
    outcome: dict = {"stopped": False}
    if workers <= 1:
        for item in pending:
            if should_stop():
                outcome["stopped"] = True
                break
            try:
                result = work(item)
            except TeacherError as error:
                outcome.update(stopped=True, error=str(error))
                break
            on_result(item, result)
        return outcome
    with ThreadPoolExecutor(max_workers=workers) as pool:
        running: dict[Future, T] = {}

        def submit() -> bool:
            if outcome["stopped"]:
                return False
            if should_stop():
                outcome["stopped"] = True
                return False
            item = next(pending, None)
            if item is None:
                return False
            running[pool.submit(work, item)] = item
            return True

        while len(running) < workers and submit():
            pass
        while running:
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                item = running.pop(future)
                try:
                    result = future.result()
                except TeacherError as error:
                    outcome.update(stopped=True, error=str(error))
                    continue
                on_result(item, result)
            while len(running) < workers and submit():
                pass
    return outcome


class Streak:
    """Counts bad results in a row; `tripped` once `limit` came one after another. A run of
    the same HTTP failure means the setup is wrong (a retired model), not the samples."""

    def __init__(self, limit: int = 20):
        self.limit = limit
        self.count = 0

    def add(self, bad: bool) -> None:
        self.count = self.count + 1 if bad else 0

    @property
    def tripped(self) -> bool:
        return self.count >= self.limit

    def error(self, what: str) -> str:
        return f"{self.count} {what} in a row: the setup, not the samples, is failing"
