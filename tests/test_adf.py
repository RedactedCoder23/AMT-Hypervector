import torch
from hypervector_core.adf import ADF


def test_adf_update():
    adf = ADF(dims=6)
    before = adf.mu_plus.clone()
    adf.update(alpha=0.1)
    # When mu_minus starts as a clone, update should not change mu_plus
    assert torch.allclose(before, adf.mu_plus)
