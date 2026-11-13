import threading
import time

from emperor.data.parallel import run_items
from emperor.data.teacher import TeacherError


def test_items_run_in_parallel_and_results_come_back_in_the_caller_thread():
    seen, threads, caller = [], set(), threading.get_ident()

    def work(n):
        threads.add(threading.get_ident())
        time.sleep(0.02)
        return n * 2

    def handle(n, result):
        assert threading.get_ident() == caller
        seen.append((n, result))

    outcome = run_items(range(8), work, handle, workers=4)
    assert outcome == {"stopped": False}
    assert sorted(seen) == [(n, n * 2) for n in range(8)]
    assert len(threads) > 1


def test_a_teacher_error_stops_new_work_and_keeps_finished_results():
    started = []

    def work(n):
        started.append(n)
        if n == 2:
            raise TeacherError("budget reached")
        return n

    done = []
    outcome = run_items(range(100), work, lambda n, r: done.append(r), workers=2)
    assert outcome["stopped"] and "budget" in outcome["error"]
    assert len(started) < 100 and 2 not in done


def test_one_worker_runs_in_order_and_honours_stop():
    done = []
    outcome = run_items(
        range(5), lambda n: n, lambda n, r: done.append(r), should_stop=lambda: len(done) >= 3
    )
    assert done == [0, 1, 2] and outcome["stopped"]
