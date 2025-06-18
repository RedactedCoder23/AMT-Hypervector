import torch
from hypervector_core.adf import ADF


def test_adf_update():
    adf = ADF(dims=6)
    before = adf.mu_plus.clone()
    adf.update(alpha=0.1)
    assert not torch.allclose(before, adf.mu_plus)
