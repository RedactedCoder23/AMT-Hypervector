"""Example training loop using BHRE components."""

import time
from amt.encoder import HypervectorEncoder
from amt.adf_update import ADFMemory


def main() -> None:
    enc = HypervectorEncoder(dim=6, alpha=[1.0] * 6)
    mem = ADFMemory(dim=6)
    samples = [
        ("move:e2e4", True),
        ("move:d7d5", False),
    ]
    for text, pos in samples:
        hv = enc.encode(text)
        mem.update(hv, positive=pos)
        time.sleep(0.01)


if __name__ == "__main__":
    main()
