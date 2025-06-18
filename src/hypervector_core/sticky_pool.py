import heapq
from typing import List


class StickyPool:
    """Top-K heap of hypervectors keyed by an error metric."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.heap = []

    def add(self, hv, error: float):
        entry = (error, hv)
        if len(self.heap) < self.capacity:
            heapq.heappush(self.heap, entry)
        else:
            heapq.heappushpop(self.heap, entry)

    def sample(self, k: int) -> List:
        return [hv for (_, hv) in heapq.nlargest(k, self.heap)]
