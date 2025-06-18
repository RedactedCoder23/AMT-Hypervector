import numpy as np
from typing import List


class ADFMemory:
    def __init__(self, dim: int):
        self.mu_pos = np.zeros(dim)
        self.mu_neg = np.zeros(dim)

    def update(self, hv: np.ndarray, positive: bool):
        target = self.mu_pos if positive else self.mu_neg
        target += hv
        # Renormalize both channels to unit length
        norm_pos = np.linalg.norm(self.mu_pos)
        self.mu_pos /= norm_pos if norm_pos > 0 else 1
        norm_neg = np.linalg.norm(self.mu_neg)
        self.mu_neg /= norm_neg if norm_neg > 0 else 1

    def similarity_table(self, hvs: List[np.ndarray]) -> np.ndarray:
        # Compute [cos(mu_pos, hv), cos(mu_neg, hv)] for each hv
        sims = []
        for hv in hvs:
            p = np.dot(self.mu_pos, hv) / (
                np.linalg.norm(self.mu_pos) * np.linalg.norm(hv) + 1e-12
            )
            n = np.dot(self.mu_neg, hv) / (
                np.linalg.norm(self.mu_neg) * np.linalg.norm(hv) + 1e-12
            )
            sims.append([p, n])
        return np.array(sims)
