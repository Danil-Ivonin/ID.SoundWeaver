import atexit
import inspect
import os
from asyncio import AbstractEventLoop, new_event_loop
from typing import TypeVar

T = TypeVar("T")

_sync_loop: AbstractEventLoop | None = None
_sync_loop_pid: int | None = None


async def maybe_await(value: T) -> T:
    if inspect.isawaitable(value):
        return await value
    return value


def _close_sync_loop() -> None:
    global _sync_loop, _sync_loop_pid

    if _sync_loop is not None and not _sync_loop.is_closed():
        _sync_loop.close()
    _sync_loop = None
    _sync_loop_pid = None


def _get_sync_loop() -> AbstractEventLoop:
    global _sync_loop, _sync_loop_pid

    pid = os.getpid()
    if _sync_loop is None or _sync_loop.is_closed() or _sync_loop_pid != pid:
        if _sync_loop is not None and not _sync_loop.is_closed():
            _sync_loop.close()
        _sync_loop = new_event_loop()
        _sync_loop_pid = pid
    return _sync_loop


def run_sync(awaitable):
    return _get_sync_loop().run_until_complete(awaitable)


atexit.register(_close_sync_loop)
