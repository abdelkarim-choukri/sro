# sro/utils/batching.py
from typing import Iterable, List, TypeVar, Iterator

T = TypeVar("T")

def batched(xs: Iterable[T], batch_size: int) -> Iterator[List[T]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    buf: List[T] = []
    for x in xs:
        buf.append(x)
        if len(buf) == batch_size:
            yield buf
            buf = []
    if buf:  # tail
        yield buf
