import torch
from hypervector_core.encoder import encode_token


def test_encode_shapes():
    v1 = encode_token("foo", dims=6)
    v2 = encode_token("foo", dims=6)
    assert v1.shape == (6,)
    assert torch.allclose(v1, v2)
