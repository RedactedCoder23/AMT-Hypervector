import numpy as np
from typing import Sequence


def sinc_kernel(x: np.ndarray, y: np.ndarray, alpha: Sequence[float]) -> float:
    """6-D separable sinc kernel: ∏₀⁵ sinc(α[i]*(x[i]–y[i]))"""
    diffs = np.array(alpha, dtype=float) * (x - y)
    return float(np.prod(np.sinc(diffs)))
