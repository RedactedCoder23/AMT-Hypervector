import numpy as np
from amt.encoder import HypervectorEncoder

def test_encode_consistency():
    enc = HypervectorEncoder(dim=6, alpha=[0.2]*6)
    hv1 = enc.encode("foo")
    hv2 = enc.encode("foo")
    assert np.allclose(hv1, hv2)
