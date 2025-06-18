from amt.kernel import cosine_similarity
import numpy as np

def test_cosine_similarity():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    assert np.isclose(cosine_similarity(a, b), 1.0)
