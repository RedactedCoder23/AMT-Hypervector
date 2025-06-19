"""
Deterministic text-to-hypervector encoder + cosine helpers.
Uses SHA-256 → RNG seed → N(0,1) floats → L2 normalization.
"""
import hashlib
import numpy as np

ENC_DIM = 10_000  # hypervector dimensionality


def _hash_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, byteorder="big")


def encode(text: str, dim: int = ENC_DIM) -> np.ndarray:
    rng = np.random.default_rng(_hash_seed(text))
    vec = rng.standard_normal(dim)
    return vec / np.linalg.norm(vec)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
