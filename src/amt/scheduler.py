"""Simple learning rate scheduler placeholder."""

class DummyScheduler:
    def __init__(self, lr: float):
        self.lr = lr

    def step(self):
        self.lr *= 0.99
        return self.lr
