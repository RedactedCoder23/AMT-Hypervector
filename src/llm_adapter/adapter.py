import torch.nn as nn


class HypervectorAdapter(nn.Module):
    """LoRA-style adapter injecting hypervector signals."""
    def __init__(self, hidden_size: int, r: int = 4):
        super().__init__()
        self.down = nn.Linear(hidden_size, r, bias=False)
        self.up   = nn.Linear(r, hidden_size, bias=False)

    def forward(self, x):
        return x + self.up(self.down(x))
