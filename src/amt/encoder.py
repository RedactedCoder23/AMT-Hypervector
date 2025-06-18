import hashlib
import numpy as np
from typing import Sequence


class HypervectorEncoder:
    def __init__(self, dim: int = 6, alpha: Sequence[float] = None):
        self.dim = dim
        self.alpha = np.array(alpha if alpha is not None else [1.0] * dim)

    def encode(self, text: str) -> np.ndarray:
        # 1) SHA-256 hash → 32 bytes
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # 2) Split into `dim` equal chunks, convert each to float in [–1,1]
        floats = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        chunks = np.array_split(floats, self.dim)
        floats = np.array([c.mean() for c in chunks])
        hv = floats - floats.mean()
        # 3) ℓ₂ normalize
        return hv / np.linalg.norm(hv)
