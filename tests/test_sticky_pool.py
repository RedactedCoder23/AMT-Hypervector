import torch
from hypervector_core import StickyPool


def test_sticky_pool_topk():
    pool = StickyPool(capacity=3)
    for i in range(5):
        pool.add(torch.tensor([float(i)]), error=float(i))
    sample = pool.sample(3)
    values = [v.item() for v in sample]
    assert values == [4.0, 3.0, 2.0]
