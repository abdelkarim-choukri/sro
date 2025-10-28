# sro/utils/batching.py
from collections.abc import Iterable, Iterator
from typing import List, TypeVar

T = TypeVar("T")

def batched(xs: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    buf: list[T] = []
    for x in xs:
        buf.append(x)
        if len(buf) == batch_size:
            yield buf
            buf = []
    if buf:  # tail
        yield buf
