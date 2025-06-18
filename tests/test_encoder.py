import numpy as np

from amt.encoder import HypervectorEncoder


def test_encode_shape_and_consistency():
    enc = HypervectorEncoder(dim=6, alpha=[1.0] * 6)
    hv1 = enc.encode("foo")
    hv2 = enc.encode("foo")

    # shape should be (6,) and deterministic
    assert hv1.shape == (6,)
    assert np.allclose(hv1, hv2)
