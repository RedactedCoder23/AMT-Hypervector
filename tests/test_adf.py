import torch
from hypervector_core import ADF


def test_adf_update_changes_mean():
    adf = ADF(dims=6)
    before = adf.mu_plus.clone()
    adf.update(alpha=0.5)
    assert not torch.allclose(before, adf.mu_plus)
