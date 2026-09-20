"""Report cyclic weight banking for square dense layers."""

from __future__ import annotations

import argparse

from predcircuit.weight_banking import analyze


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--weight-bits", type=int, default=14)
    parser.add_argument("--parallelism", type=int, nargs="+", default=[8, 16, 32, 64])
    args = parser.parse_args()

    print(
        "P,forward_conflicts,transpose_conflicts,max_bank_weights,"
        "RAMB36,capacity_efficiency"
    )
    for p in args.parallelism:
        result = analyze(args.width, p, args.weight_bits)
        print(
            f"{p},{result.forward_conflicts},{result.transpose_conflicts},"
            f"{result.max_bank_occupancy},{result.total_ramb36},"
            f"{result.capacity_efficiency:.6f}"
        )


if __name__ == "__main__":
    main()
