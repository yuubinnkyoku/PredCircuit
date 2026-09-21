from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from diagnose_pcalm_dual_precision import quantize_dual
from diagnose_pcalm_update_precision import run_precision
from diagnose_pcalm_weight_precision import WeightQuantizedResidualMLP


def stored_forward(model: WeightQuantizedResidualMLP, x: torch.Tensor) -> torch.Tensor:
    z = x
    for i in range(len(model.weights)):
        z = model.block_pred(i, z)
    return z


def apply_update(
    model: WeightQuantizedResidualMLP,
    grads: list[torch.Tensor],
    *,
    lr: float,
    stochastic: bool,
    rng_bits: int,
    generator: torch.Generator,
) -> float:
    moved = total = 0
    step = 1.0 / 32.0
    with torch.no_grad():
        for w, g in zip(model.weights, grads, strict=True):
            before, _, _, _ = quantize_dual(w, model.weight_precision)
            target = before - lr * g
            if model.weight_precision == "fp32":
                after = target
            elif stochastic:
                q = target / step
                lo = torch.floor(q)
                frac = q - lo
                levels = 1 << rng_bits
                p = torch.floor(frac * levels) / levels
                u = torch.randint(levels, p.shape, generator=generator).to(p.dtype) / levels
                after = (lo + (u < p)) * step
                after = torch.clamp(after, -8.0, 8.0 - step)
            else:
                after, _, _, _ = quantize_dual(target, model.weight_precision)
            moved += int((after != before).sum())
            total += before.numel()
            w.copy_(after)
    return moved / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()
    depth, width = 32, 8
    data_gen = torch.Generator().manual_seed(args.seed + 20000)
    update_gen = torch.Generator().manual_seed(args.seed + 30000)
    batches = [
        (torch.randn(4, 8, generator=data_gen), torch.randn(4, 4, generator=data_gen))
        for _ in range(args.steps)
    ]
    eval_x = torch.randn(16, 8, generator=data_gen)
    eval_y = torch.randn(16, 4, generator=data_gen)
    configs = [
        ("fp32_lr0p01", "fp32", 0.01, False),
        ("fp32_lr0p1", "fp32", 0.1, False),
        ("q9_det_lr0p1", "fixed9_i3", 0.1, False),
        ("q9_sr12_lr0p1", "fixed9_i3", 0.1, True),
    ]
    rows: list[dict[str, object]] = []
    for name, precision, lr, stochastic in configs:
        model = WeightQuantizedResidualMLP(
            depth=depth,
            width=width,
            input_dim=8,
            output_dim=4,
            activation="relu",
            seed=args.seed + depth,
            weight_precision=precision,
        )
        finite = True
        for step_ix, (x, y) in enumerate(batches):
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
            finite = finite and bool(metrics["finite"])
            move = apply_update(
                model,
                grads,
                lr=lr,
                stochastic=stochastic,
                rng_bits=12,
                generator=update_gen,
            )
            with torch.no_grad():
                loss = float(torch.mean((stored_forward(model, eval_x) - eval_y) ** 2))
            finite = finite and torch.isfinite(torch.tensor(loss)).item()
            rows.append(
                {
                    "seed": args.seed,
                    "config": name,
                    "step": step_ix + 1,
                    "eval_mse": loss,
                    "weight_move_rate": move,
                    "finite": finite,
                }
            )
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.groupby("config").tail(1).to_string(index=False))


if __name__ == "__main__":
    main()
