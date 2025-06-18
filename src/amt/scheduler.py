from typing import Iterator


def linear_schedule(start: float, end: float, steps: int) -> Iterator[float]:
    for i in range(steps):
        yield start + (end - start) * i / (steps - 1)
