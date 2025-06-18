import heapq


class StickyPool:
    """Keeps top-K hypervectors by error score for replay."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.heap = []

    def add(self, hv, error: float):
        entry = (error, hv)
        if len(self.heap) < self.capacity:
            heapq.heappush(self.heap, entry)
        else:
            heapq.heappushpop(self.heap, entry)

    def sample(self, k: int):
        return [hv for (_, hv) in heapq.nlargest(k, self.heap)]
