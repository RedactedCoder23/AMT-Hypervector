import numpy as np

from amt.kernel import sinc_kernel


def test_sinc_kernel():
    x = np.zeros(6)
    y = np.zeros(6)
    alpha = [1.0] * 6
    assert np.isclose(sinc_kernel(x, y, alpha), 1.0)
