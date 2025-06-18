import torch
from hypervector_core import compute_info_gain


def test_info_gain_sign():
    candidates = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    mu_p = torch.tensor([1.0, 0.0, 0.0])
    mu_n = torch.tensor([-1.0, 0.0, 0.0])
    gains = compute_info_gain(candidates, mu_p, mu_n)
    assert gains[0] > 0 and gains[2] < 0
