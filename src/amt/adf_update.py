"""Simple ADF memory updater."""

import numpy as np


class ADFMemory:
    def __init__(self, dim: int):
        self.mu_plus = np.random.randn(dim)
        self.mu_plus /= np.linalg.norm(self.mu_plus) + 1e-9
        self.mu_minus = self.mu_plus.copy()

    def update(
        self,
        hv: np.ndarray,
        positive: bool = True,
        alpha: float = 0.5,
    ) -> None:
        if positive:
            self.mu_plus += alpha * hv
        else:
            self.mu_minus += alpha * hv
        self.mu_plus /= np.linalg.norm(self.mu_plus) + 1e-9
        self.mu_minus /= np.linalg.norm(self.mu_minus) + 1e-9

    def similarity_table(self, hvs):
        out = []
        for hv in hvs:
            score = float(hv.dot(self.mu_plus) - hv.dot(self.mu_minus))
            out.append(score)
        return out
