import numpy as np

from amt.kernel import cosine_similarity


def test_cosine_similarity():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    assert np.isclose(cosine_similarity(a, b), 1.0)
