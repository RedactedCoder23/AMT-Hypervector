"""BHRE command line interface.

This module provides a small wrapper around the main hypervector
components. It loads a YAML configuration file defining ``encode_rules``
and ``query_rules`` and prints similarity results for each query.
"""

import argparse

import yaml  # type: ignore

from .adf_update import ADFMemory
from .encoder import HypervectorEncoder


from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="bhre",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        metavar="PATH",
        help="Path to YAML config with encode and query rules",
        required=True,
    )
    args = parser.parse_args(argv)

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
