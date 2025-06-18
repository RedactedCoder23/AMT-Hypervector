import torch
import torch.nn.functional as F


class ADF:
    """Dual-channel Bayesian hypervector updater."""
    def __init__(self, dims: int, device=None):
        v = torch.randn(dims, device=device)
        self.mu_plus  = F.normalize(v, p=2, dim=-1)
        w = torch.randn(dims, device=device)
        self.mu_minus = F.normalize(w, p=2, dim=-1)

    def update(self, alpha: float = 0.5):
        delta = self.mu_plus - self.mu_minus
        self.mu_plus  = F.normalize(self.mu_plus  + alpha * delta, p=2, dim=-1)
        self.mu_minus = F.normalize(self.mu_minus - alpha * delta, p=2, dim=-1)
