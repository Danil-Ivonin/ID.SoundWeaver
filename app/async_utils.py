import inspect
from typing import TypeVar

T = TypeVar("T")


async def maybe_await(value: T) -> T:
    if inspect.isawaitable(value):
        return await value
    return value
