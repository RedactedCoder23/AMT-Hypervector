"""Deterministic hypervector encoder."""

import hashlib
from typing import Sequence

import numpy as np


class HypervectorEncoder:
    def __init__(self, dim: int, alpha: Sequence[float]):
        self.dim = dim
        self.alpha: np.ndarray = np.array(alpha, dtype=float)

    def encode(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode()).digest()
        v = np.frombuffer(h[: self.dim], dtype=np.uint8).astype(np.float32)
        v = (v / 255.0) * 2.0 - 1.0
        v /= np.linalg.norm(v) + 1e-9
        hv = np.sinc(v * self.alpha)
        return np.asarray(hv, dtype=float)
