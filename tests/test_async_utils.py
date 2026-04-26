import asyncio

from app.async_utils import _close_sync_loop, run_sync


async def _current_loop_id() -> int:
    return id(asyncio.get_running_loop())


def test_run_sync_reuses_same_event_loop_within_process():
    try:
        first = run_sync(_current_loop_id())
        second = run_sync(_current_loop_id())

        assert first == second
    finally:
        _close_sync_loop()


def test_run_sync_recreates_loop_after_close():
    try:
        first = run_sync(_current_loop_id())
        _close_sync_loop()
        second = run_sync(_current_loop_id())

        assert isinstance(first, int)
        assert isinstance(second, int)
    finally:
        _close_sync_loop()
