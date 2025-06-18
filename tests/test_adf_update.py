import numpy as np
from amt.adf_update import ADFMemory

def test_adf_update_clone():
    mem = ADFMemory(dim=3)
    before = mem.mu_plus.copy()
    mem.update(np.array([0.0,0.0,0.0]), positive=True, alpha=0.1)
    assert np.allclose(before, mem.mu_plus)
