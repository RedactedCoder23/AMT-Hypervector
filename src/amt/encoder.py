"""Deterministic hypervector encoder."""
import hashlib
import numpy as np

class HypervectorEncoder:
    def __init__(self, dim: int, alpha):
        self.dim = dim
        self.alpha = np.array(alpha, dtype=float)

    def encode(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode()).digest()
        v = np.frombuffer(h[:self.dim], dtype=np.uint8).astype(np.float32)
        v = (v / 255.0) * 2.0 - 1.0
        v /= np.linalg.norm(v) + 1e-9
        return np.sinc(v * self.alpha)
