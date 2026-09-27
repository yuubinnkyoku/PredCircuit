"""Report cyclic weight banking for square dense layers."""

from __future__ import annotations

import argparse

from predcircuit.weight_banking import analyze


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=31)
    parser.add_argument("--weight-bits", type=int, default=16)
    parser.add_argument("--parallelism", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    args = parser.parse_args()

    print(
        "P,layers,forward_conflicts,transpose_conflicts,max_bank_weights,"
        "capacity_packed_RAMB36,layer_parallel_RAMB36,capacity_efficiency"
    )
    for p in args.parallelism:
        packed = analyze(args.width, p, args.weight_bits, layers=args.layers)
        one_layer = analyze(args.width, p, args.weight_bits, layers=1)
        # Capacity packing shares the same P physical banks across layers.  That is
        # valid for time-multiplexed layers, but it cannot feed all layers in the
        # same cycle.  Full layer parallelism needs an independent bank set per
        # simultaneously active layer.
        layer_parallel_ramb36 = args.layers * one_layer.total_ramb36
        print(
            f"{p},{args.layers},{packed.forward_conflicts},{packed.transpose_conflicts},"
            f"{packed.max_bank_occupancy},{packed.total_ramb36},"
            f"{layer_parallel_ramb36},{packed.capacity_efficiency:.6f}"
        )


if __name__ == "__main__":
    main()
