"""Simple MLP model using numpy."""

import numpy as np


class MLP:
    def __init__(self, input_dim: int, hidden_dim: int):
        self.w1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.w2 = np.random.randn(hidden_dim, 1) * 0.1

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ self.w1)
        return h @ self.w2
