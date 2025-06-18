"""BHRE CLI: encode, update, and query similarity with ADFMemory"""

import argparse
import yaml
from amt.encoder import HypervectorEncoder
from amt.adf_update import ADFMemory


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="YAML config path",
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    enc = HypervectorEncoder(dim=cfg["dim"], alpha=cfg["alpha"])
    mem = ADFMemory(dim=cfg["dim"])
    for r in cfg.get("encode_rules", []):
        hv = enc.encode(r["text"])
        mem.update(hv, r["positive"])
    for q in cfg.get("query_rules", []):
        hv = enc.encode(q["text"])
        p, n = mem.similarity_table([hv])[0]
        print(f"{q['text']} → positive:{p:.3f}, negative:{n:.3f}")


if __name__ == "__main__":
    main()
