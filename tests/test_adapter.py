import torch
from llm_adapter.adapter import HypervectorAdapter


def test_adapter_forward():
    x = torch.randn(2, 6)
    adapter = HypervectorAdapter(hidden_size=6, r=2)
    out = adapter(x)
    assert out.shape == x.shape
