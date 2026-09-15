"""Probe fixed-point sensitivity of the PC-ALM dual integrator.

This isolates the hardware-critical recurrence lambda <- lambda + alpha*r from
network effects.  A decaying residual sequence models late relaxation, where
small corrections are most likely to disappear under quantization.
"""

from __future__ import annotations

import argparse
import json
import math


def quantize(x: float, *, total_bits: int, frac_bits: int) -> tuple[float, bool]:
    scale = 1 << frac_bits
    lo = -(1 << (total_bits - 1))
    hi = (1 << (total_bits - 1)) - 1
    q = round(x * scale)
    saturated = q < lo or q > hi
    q = min(max(q, lo), hi)
    return q / scale, saturated


def simulate(
    *,
    total_bits: int,
    frac_bits: int,
    alpha: float,
    r0: float,
    decay: float,
    steps: int,
) -> dict[str, float | int]:
    lam_ref = 0.0
    lam_q = 0.0
    saturations = 0
    lost_updates = 0
    first_lost = 0
    for step in range(1, steps + 1):
        residual = r0 * decay ** (step - 1)
        lam_ref += alpha * residual
        candidate = lam_q + alpha * residual
        new_lam, saturated = quantize(candidate, total_bits=total_bits, frac_bits=frac_bits)
        saturations += int(saturated)
        if new_lam == lam_q and residual != 0.0:
            lost_updates += 1
            if first_lost == 0:
                first_lost = step
        lam_q = new_lam
    abs_error = abs(lam_q - lam_ref)
    return {
        "total_bits": total_bits,
        "frac_bits": frac_bits,
        "integer_bits_including_sign": total_bits - frac_bits,
        "lsb": 2.0**-frac_bits,
        "reference_lambda": lam_ref,
        "quantized_lambda": lam_q,
        "absolute_error": abs_error,
        "relative_error": abs_error / max(abs(lam_ref), 1e-30),
        "saturations": saturations,
        "lost_updates": lost_updates,
        "first_lost_update_step": first_lost,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--r0", type=float, default=0.25)
    parser.add_argument("--decay", type=float, default=0.97)
    parser.add_argument("--steps", type=int, default=256)
    args = parser.parse_args()
    formats = [(16, 12), (16, 10), (12, 8), (12, 6), (8, 4), (8, 2)]
    rows = [
        simulate(
            total_bits=bits,
            frac_bits=frac,
            alpha=args.alpha,
            r0=args.r0,
            decay=args.decay,
            steps=args.steps,
        )
        for bits, frac in formats
    ]
    assert all(math.isfinite(float(row["relative_error"])) for row in rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
