import torch
from hypervector_core import encode_token


def test_encode_token_deterministic():
    hv1 = encode_token("foo")
    hv2 = encode_token("foo")
    assert torch.allclose(hv1, hv2)
    assert hv1.shape == (6,)
