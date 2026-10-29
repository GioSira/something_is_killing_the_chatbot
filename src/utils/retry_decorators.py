import asyncio
import functools
import random
import time
from typing import Awaitable, Callable, TypeVar, Type, Tuple, ParamSpec


P = ParamSpec("P")
T = TypeVar("T")


def retry_async(
        exception: Tuple[Type[BaseException], ...],
        max_retries: int = 5,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        factor: float = 2.0,
        jitter: float = 0.25
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            delay = base_delay
            for attempt in range(1, max_retries+1):
                try:
                    return await func(*args, **kwargs)
                except exception as error:
                    if attempt == max_retries:
                        raise
                    delay = min(base_delay * factor ** (attempt - 1), max_delay)
                    wait = delay * (1+ random.uniform(-jitter, jitter))
                    await asyncio.sleep(wait)

        return wrapper

    return decorator



def retry_sync(
        exception: Tuple[Type[BaseException], ...],
        max_retries: int = 5,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        factor: float = 2.0,
        jitter: float = 0.25
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            delay = base_delay
            for attempt in range(1, max_retries+1):
                try:
                    return func(*args, **kwargs)
                except exception as error:
                    if attempt == max_retries:
                        raise
                    delay = min(base_delay * factor ** (attempt - 1), max_delay)
                    wait = delay * (1+ random.uniform(-jitter, jitter))
                    time.sleep(wait)

        return wrapper

    return decorator