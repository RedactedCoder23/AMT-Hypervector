try:
    import torch.nn as nn
except Exception:  # pragma: no cover - optional dependency
    nn = None  # type: ignore


if nn is None:

    class _BigMLPPlaceholder:  # pragma: no cover - optional dependency
        def __init__(self, *_: int, **__: int) -> None:
            raise ImportError("torch is required for BigMLP")

    BigMLP = _BigMLPPlaceholder

else:

    class BigMLP(nn.Module):  # type: ignore[no-redef]
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x):
            return self.net(x)
