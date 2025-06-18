import hashlib
import numpy as np
import torch


def encode_token(token: str, dims: int = 6, device=None) -> torch.Tensor:
    """Deterministically encode a token into a hypervector."""
    h = hashlib.sha256(token.encode()).digest()
    v = np.frombuffer(h[:dims], dtype=np.uint8).astype(np.float32)
    v = (v / 255.0) * 2 - 1
    x = torch.from_numpy(v).to(device)
    x = x / (x.norm() + 1e-9)
    return torch.sinc(x)
