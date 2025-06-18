import argparse
from .encoder import HypervectorEncoder
from .adf_update import ADFMemory
import yaml

def main():
    parser = argparse.ArgumentParser(prog="bhre")
    parser.add_argument("--config", "-c", help="YAML config path", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    enc = HypervectorEncoder(dim=cfg["dim"], alpha=cfg["alpha"])
    mem = ADFMemory(dim=cfg["dim"])

    for item in cfg["encode_rules"]:
        hv = enc.encode(item["text"])
        mem.update(hv, positive=item["positive"])

    print(mem.similarity_table([]))

if __name__ == "__main__":
    main()
