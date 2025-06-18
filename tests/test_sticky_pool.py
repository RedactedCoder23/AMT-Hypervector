import torch
from hypervector_core.sticky_pool import StickyPool


def test_sticky_pool():
    pool = StickyPool(capacity=2)
    pool.add(torch.tensor([1]), error=0.1)
    pool.add(torch.tensor([2]), error=0.2)
    pool.add(torch.tensor([3]), error=0.05)
    samples = pool.sample(2)
    # should contain the two highest-error entries: error 0.2 and 0.1
    assert any((s == torch.tensor([2])).all() for s in samples)
    assert any((s == torch.tensor([1])).all() for s in samples)
