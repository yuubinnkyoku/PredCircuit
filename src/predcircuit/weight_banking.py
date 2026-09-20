"""Cyclic weight-banking analysis for square dense layers."""

from __future__ import annotations

import math
from dataclasses import dataclass

RAMB36_BITS = 36_864


@dataclass(frozen=True)
class BankingResult:
    width: int
    parallelism: int
    weight_bits: int
    layers: int
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


def analyze(
    width: int, parallelism: int, weight_bits: int = 14, layers: int = 1
) -> BankingResult:
    if width <= 0 or parallelism <= 0 or weight_bits <= 0 or layers <= 0:
        raise ValueError("width, parallelism, weight_bits, and layers must be positive")
    if width % parallelism != 0:
        raise ValueError("this aligned schedule requires parallelism to divide width")

    occupancy = [0] * parallelism
    for row in range(width):
        for col in range(width):
            occupancy[bank(row, col, parallelism)] += layers

    forward_conflicts = 0
    for row in range(width):
        for col0 in range(0, width, parallelism):
            banks = [bank(row, col0 + lane, parallelism) for lane in range(parallelism)]
            forward_conflicts += _conflicts(banks)

    transpose_conflicts = 0
    for col in range(width):
        for row0 in range(0, width, parallelism):
            banks = [bank(row0 + lane, col, parallelism) for lane in range(parallelism)]
            transpose_conflicts += _conflicts(banks)

    max_bank_occupancy = max(occupancy)
    bits_per_bank = max_bank_occupancy * weight_bits
    ramb36_per_bank = math.ceil(bits_per_bank / RAMB36_BITS)
    total_ramb36 = parallelism * ramb36_per_bank
    useful_bits = layers * width * width * weight_bits
    capacity_efficiency = useful_bits / (total_ramb36 * RAMB36_BITS)

    return BankingResult(
        width=width,
        parallelism=parallelism,
        weight_bits=weight_bits,
        layers=layers,
        forward_conflicts=forward_conflicts,
        transpose_conflicts=transpose_conflicts,
        max_bank_occupancy=max_bank_occupancy,
        ramb36_per_bank=ramb36_per_bank,
        total_ramb36=total_ramb36,
        capacity_efficiency=capacity_efficiency,
    )
