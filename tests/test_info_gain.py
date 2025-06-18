import torch
from hypervector_core.info_gain import compute_info_gain


def test_info_gain():
    mu_p = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mu_n = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    candidates = torch.stack([mu_p, mu_n])
    gains = compute_info_gain(candidates, mu_p, mu_n)
    assert gains[0] > gains[1]
