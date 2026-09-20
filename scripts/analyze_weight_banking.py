"""Exhaustively analyze cyclic weight banking for square dense layers.

The mapping bank(r, c) = (r + c) mod P is intended to serve both W h
(row-oriented) and W^T c (column-oriented) from one physical weight copy.
This script verifies every aligned P-wide access group and estimates the
RAMB36 fragmentation caused by splitting the matrix into P independent banks.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

RAMB36_BITS = 36_864


@dataclass(frozen=True)
class BankingResult:
    width: int
    parallelism: int
    weight_bits: int
    forward_conflicts: int
    transpose_conflicts: int
    max_bank_occupancy: int
    ramb36_per_bank: int
    total_ramb36: int
    capacity_efficiency: float


def bank(row: int, col: int, parallelism: int) -> int:
    return (row + col) % parallelism


def _conflicts(indices: list[int]) -> int:
    return len(indices) - len(set(indices))


def analyze(width: int, parallelism: int, weight_bits: int = 14) -> BankingResult:
    if width <= 0 or parallelism <= 0 or weight_bits <= 0:
        raise ValueError("width, parallelism, and weight_bits must be positive")
    if width % parallelism != 0:
        raise ValueError("this aligned schedule requires parallelism to divide width")

    forward_conflicts = 0
    transpose_conflicts = 0
    occupancy = [0] * parallelism

    for row in range(width):
        for col in range(width):
            occupancy[bank(row, col, parallelism)] += 1

    for row in range(width):
        for col0 in range(0, width, parallelism):
            banks = [bank(row, col0 + lane, parallelism) for lane in range(parallelism)]
            forward_conflicts += _conflicts(banks)

    for col in range(width):
        for row0 in range(0, width, parallelism):
            banks = [bank(row0 + lane, col, parallelism) for lane in range(parallelism)]
            transpose_conflicts += _conflicts(banks)

    max_bank_occupancy = max(occupancy)
    bits_per_bank = max_bank_occupancy * weight_bits
    ramb36_per_bank = math.ceil(bits_per_bank / RAMB36_BITS)
    total_ramb36 = parallelism * ramb36_per_bank
    useful_bits = width * width * weight_bits
    capacity_efficiency = useful_bits / (total_ramb36 * RAMB36_BITS)

    return BankingResult(
        width=width,
        parallelism=parallelism,
        weight_bits=weight_bits,
        forward_conflicts=forward_conflicts,
        transpose_conflicts=transpose_conflicts,
        max_bank_occupancy=max_bank_occupancy,
        ramb36_per_bank=ramb36_per_bank,
        total_ramb36=total_ramb36,
        capacity_efficiency=capacity_efficiency,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--weight-bits", type=int, default=14)
    parser.add_argument("--parallelism", type=int, nargs="+", default=[8, 16, 32, 64])
    args = parser.parse_args()

    print("P,forward_conflicts,transpose_conflicts,max_bank_weights,RAMB36,capacity_efficiency")
    for p in args.parallelism:
        result = analyze(args.width, p, args.weight_bits)
        print(
            f"{p},{result.forward_conflicts},{result.transpose_conflicts},"
            f"{result.max_bank_occupancy},{result.total_ramb36},"
            f"{result.capacity_efficiency:.6f}"
        )


if __name__ == "__main__":
    main()
