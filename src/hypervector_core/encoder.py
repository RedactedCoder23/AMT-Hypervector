import hashlib
import numpy as np
import torch


def encode_token(token: str, dims: int = 6, device=None) -> torch.Tensor:
    """
    Deterministic hypervector encoding:
      1) SHA-256 hash \u2192 first `dims` bytes
      2) map [0,255] -> [-1,1], \u21132 normalize
      3) apply separable sinc
    """
    h = hashlib.sha256(token.encode()).digest()
    v = np.frombuffer(h[:dims], dtype=np.uint8).astype(np.float32)
    v = (v / 255.0) * 2.0 - 1.0
    x = torch.tensor(v, dtype=torch.float32, device=device)
    x = x / (x.norm() + 1e-9)
    return torch.sinc(x)
