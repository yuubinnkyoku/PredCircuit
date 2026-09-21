from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from diagnose_pcalm_update_precision import run_precision
from diagnose_pcalm_weight_precision import WeightQuantizedResidualMLP


WEIGHT_STEP = {"fixed9_i3": 1.0 / 32.0}


def stochastic_update_metrics(
    grads: list[torch.Tensor], *, lr: float, step: float, rng_bits: int
) -> tuple[float, float, float]:
    """Analytic metrics for stateless stochastic rounding from a weight grid point.

    For an ideal stochastic update, one LSB is emitted with probability
    p=min(|lr*g|/step, 1).  A k-bit uniform comparator can only represent
    probabilities in multiples of 2^-k; we conservatively use floor(p*2^k)/2^k.

    Returns expected fraction of weights moving by >=1 LSB, relative L1 bias in
    expected update magnitude caused by finite RNG resolution, and the fraction
    of nonzero ideal probabilities lost entirely by the finite RNG threshold.
    """
    scale = float(1 << rng_bits)
    ps: list[torch.Tensor] = []
    for grad in grads:
        ps.append(torch.clamp((lr * grad.abs()) / step, max=1.0).reshape(-1))
    p = torch.cat(ps)
    p_eff = torch.floor(p * scale) / scale
    expected_move_rate = float(p_eff.mean())
    denom = float(p.sum())
    relative_bias = float((p - p_eff).sum()) / denom if denom else 0.0
    nonzero = p > 0
    lost = nonzero & (p_eff == 0)
    lost_fraction = float(lost.sum()) / float(nonzero.sum()) if bool(nonzero.any()) else 0.0
    return expected_move_rate, relative_bias, lost_fraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    depth = 32
    width = 8
    generator = torch.Generator().manual_seed(args.seed + 10000 + depth)
    x = torch.randn(4, 8, generator=generator)
    y = torch.randn(4, 4, generator=generator)
    model = WeightQuantizedResidualMLP(
        depth=depth,
        width=width,
        input_dim=8,
        output_dim=4,
        activation="relu",
        seed=args.seed + depth,
        weight_precision="fixed9_i3",
    )
    grads, metrics = run_precision(
        model,
        x,
        y,
        state_lr=0.25,
        rho=1.0,
        alpha=0.925,
        dual_leak=0.01,
        budget=112,
        update_precision="fixed12_i2",
    )

    rows: list[dict[str, object]] = []
    for lr in (0.001, 0.01, 0.1):
        for rng_bits in (2, 4, 6, 8, 12):
            move, bias, lost = stochastic_update_metrics(
                grads,
                lr=lr,
                step=WEIGHT_STEP["fixed9_i3"],
                rng_bits=rng_bits,
            )
            rows.append(
                {
                    "seed": args.seed,
                    "learning_rate": lr,
                    "rng_bits": rng_bits,
                    "expected_move_rate": move,
                    "expected_update_relative_l1_bias": bias,
                    "nonzero_probability_lost_fraction": lost,
                    "finite": bool(metrics["finite"]),
                }
            )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
