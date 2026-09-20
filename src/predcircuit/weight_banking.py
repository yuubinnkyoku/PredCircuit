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
    ramb36_word_width: int
    ramb36_word_depth: int
    ramb36_per_bank: int
    total_ramb36: int
    capacity_efficiency: float


def bank(row: int, col: int, parallelism: int) -> int:
    return (row + col) % parallelism


def bank_address(layer: int, row: int, col: int, width: int, parallelism: int) -> tuple[int, int]:
    """Map one logical weight to a bank and depth-packed bank-local address.

    For parallelism dividing width, every row contributes ``width / parallelism``
    entries to every bank. Layer-major, row-major packing therefore needs no
    lookup table: the bank is cyclic and the local address is an affine function
    plus ``col // parallelism``.
    """
    if width <= 0 or parallelism <= 0 or width % parallelism != 0:
        raise ValueError("width must be positive and divisible by parallelism")
    if layer < 0 or not (0 <= row < width) or not (0 <= col < width):
        raise ValueError("layer, row, and col are outside the logical matrix")

    entries_per_row_bank = width // parallelism
    entries_per_layer_bank = width * entries_per_row_bank
    address = layer * entries_per_layer_bank + row * entries_per_row_bank + col // parallelism
    return bank(row, col, parallelism), address


def _ramb36_tdp_shape(word_bits: int) -> tuple[int, int]:
    """Return the smallest legal RAMB36E1 TDP word width and its depth.

    7-series RAMB36E1 true-dual-port configurations are 32768x1, 16384x2,
    8192x4, 4096x9, 2048x18, or 1024x36.  PredCircuit needs writable weights,
    so the 512x72 simple-dual-port-only mode is deliberately excluded.
    """
    if word_bits <= 0:
        raise ValueError("word_bits must be positive")
    for physical_width, depth in ((1, 32768), (2, 16384), (4, 8192), (9, 4096), (18, 2048), (36, 1024)):
        if word_bits <= physical_width:
            return physical_width, depth
    raise ValueError("one RAMB36E1 TDP word cannot exceed 36 bits")


def _conflicts(indices: list[int]) -> int:
    return len(indices) - len(set(indices))


def analyze(width: int, parallelism: int, weight_bits: int = 14, layers: int = 1) -> BankingResult:
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
    ramb36_word_width, ramb36_word_depth = _ramb36_tdp_shape(weight_bits)
    ramb36_per_bank = math.ceil(max_bank_occupancy / ramb36_word_depth)
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
        ramb36_word_width=ramb36_word_width,
        ramb36_word_depth=ramb36_word_depth,
        ramb36_per_bank=ramb36_per_bank,
        total_ramb36=total_ramb36,
        capacity_efficiency=capacity_efficiency,
    )
