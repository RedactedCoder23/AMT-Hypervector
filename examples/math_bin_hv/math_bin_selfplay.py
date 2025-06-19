#!/usr/bin/env python3
"""
Toy self-play for:
 • random math expressions
 • random 8-byte hex strings

Labels:
 • math: (a+b==c)
 • binary: checksum(bytes)%7==0

Updates μ_pos/μ_neg via incremental mean and logs each step.
"""
import argparse
import csv
import random

import numpy as np
from tqdm import trange

import hypervector as hv

LOG_FIELDS = ["step", "domain", "sample", "label", "score"]


def random_math_sample():
    a, b = random.randint(0, 19), random.randint(0, 19)
    c = random.randint(0, 38)
    expr = f"{a}+{b}={c}"
    return expr, (a + b) == c


def random_bin_sample():
    data = bytes(random.getrandbits(8) for _ in range(8))
    hex_str = data.hex()
    return hex_str, (sum(data) % 7) == 0


def incremental_mean(current, new, count):
    return current + (new - current) / count


def main():
    p = argparse.ArgumentParser("Math+Binary HV self-play")
    p.add_argument("--cycles", type=int, default=1000)
    p.add_argument("--dim", type=int, default=hv.ENC_DIM)
    p.add_argument("--log", type=str, default="training_log.csv")
    args = p.parse_args()

    mu_pos = np.zeros(args.dim)
    mu_neg = np.zeros(args.dim)
    n_pos = n_neg = 0

    with open(args.log, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writeheader()
        for i in trange(1, args.cycles + 1):
            domain = "math" if random.random() < 0.5 else "bin"
            if domain == "math":
                sample, truth = random_math_sample()
            else:
                sample, truth = random_bin_sample()
            vec = hv.encode(sample, args.dim)
            if truth:
                n_pos += 1
                mu_pos = incremental_mean(mu_pos, vec, n_pos)
                label = 1
            else:
                n_neg += 1
                mu_neg = incremental_mean(mu_neg, vec, n_neg)
                label = -1
            score = hv.cosine(vec, mu_pos) - hv.cosine(vec, mu_neg)
            w.writerow(
                {
                    "step": i,
                    "domain": domain,
                    "sample": sample,
                    "label": label,
                    "score": score,
                }
            )

    print(f"Finished {args.cycles} cycles — log in {args.log}")


if __name__ == "__main__":
    main()
