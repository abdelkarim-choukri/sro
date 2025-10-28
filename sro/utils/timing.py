import time
from contextlib import contextmanager


class StageTimer:
    def __init__(self):
        self.t = {}  # stage -> seconds

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.t[name] = self.t.get(name, 0.0) + (time.perf_counter() - t0)

    def get(self, name: str, default: float = 0.0) -> float:
        return float(self.t.get(name, default))
