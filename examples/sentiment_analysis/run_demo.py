import torch
import torch.nn as nn
import torch.nn.functional as F
from hypervector_core.encoder import encode_token
from llm_adapter.adapter import HypervectorAdapter


def encode_text(text: str) -> torch.Tensor:
    tokens = text.lower().split()
    vecs = torch.stack([encode_token(t) for t in tokens])
    return vecs.mean(dim=0)


def main():
    pos = ["i love this", "fantastic film"]
    neg = ["i hate this", "terrible movie"]
    X = torch.stack([encode_text(t) for t in pos + neg])
    y = torch.tensor([1, 1, 0, 0], dtype=torch.float32).unsqueeze(1)

    adapter = HypervectorAdapter(hidden_size=6, r=2)
    clf = nn.Linear(6, 1)
    params = list(adapter.parameters()) + list(clf.parameters())
    opt = torch.optim.SGD(params, lr=0.2)

    for _ in range(200):
        opt.zero_grad()
        out = clf(adapter(X))
        loss = F.binary_cross_entropy_with_logits(out, y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        out = torch.sigmoid(clf(adapter(X)))
        acc = ((out > 0.5).float() == y).float().mean().item()
        print(f"Training accuracy: {acc:.2f}")

    sample = "i love this movie"
    hv = encode_text(sample)
    before = torch.sigmoid(clf(hv)).item()
    after = torch.sigmoid(clf(adapter(hv))).item()
    print(f"'{sample}' -> before {before:.2f}, after {after:.2f}")


if __name__ == "__main__":
    main()
