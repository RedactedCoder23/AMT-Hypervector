"""Example evaluation script."""

import argparse


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="model.chkpt")
    args = parser.parse_args(argv)
    print(f"Evaluating {args.model}")


if __name__ == "__main__":
    main()
