"""Command line interface for BHRE hypervector experiments."""

import argparse
from .encoder import HypervectorEncoder
from .adf_update import ADFMemory
import yaml  # type: ignore


def main():
    parser = argparse.ArgumentParser(prog="bhre", description=__doc__)
    parser.add_argument(
        "--config",
        "-c",
        help="YAML config path",
        required=True,
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    enc = HypervectorEncoder(dim=cfg["dim"], alpha=cfg["alpha"])
    mem = ADFMemory(dim=cfg["dim"])

    for item in cfg["encode_rules"]:
        hv = enc.encode(item["text"])
        mem.update(hv, positive=item["positive"])

    for query in cfg.get("query_rules", []):
        qhv = enc.encode(query["text"])
        sims = mem.similarity_table([qhv])
        print(f"{query['text']} → {sims}")


if __name__ == "__main__":
    main()
