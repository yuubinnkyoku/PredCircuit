from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from diagnose_pcalm_width64_update_lattice_alignment import run_alignment
from predcircuit.pcalm import ResidualMLP, Schedule, bp_loss, method_grad

STATE_LR = 15.0 / 64.0
ALPHA = 59.0 / 64.0
DUAL_LEAK = 5.0 / 256.0
BUDGETS = (64, 80, 96)
DUAL_STAT_KEYS = (
    "dual_saturation_rate",
    "dual_prequant_max_abs",
    "dual_quantized_max_abs",
    "dual_zero_fraction",
)


def clone(base: ResidualMLP) -> ResidualMLP:
    model = ResidualMLP(
        depth=32,
        width=64,
        input_dim=8,
        output_dim=4,
        activation="relu",
        seed=0,
    )
    model.load_state_dict(base.state_dict())
    return model


def apply(model: ResidualMLP, grads: list[torch.Tensor], lr: float) -> None:
    with torch.no_grad():
        for weight, grad in zip(model.weights, grads, strict=True):
            weight.add_(grad, alpha=-lr)


def loss(model: ResidualMLP, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return float(bp_loss(model, x, y))


def dual_stats(stats: dict[str, float | bool]) -> dict[str, float]:
    return {key: float(stats[key]) for key in DUAL_STAT_KEYS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--weight-lr", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    generator = torch.Generator().manual_seed(args.seed + 90000)
    teacher = torch.randn(8, 4, generator=generator) / 8**0.5
    train_x = torch.randn(64, 8, generator=generator)
    eval_x = torch.randn(256, 8, generator=generator)
    train_y = torch.tanh(train_x @ teacher)
    eval_y = torch.tanh(eval_x @ teacher)

    base = ResidualMLP(
        depth=32,
        width=64,
        input_dim=8,
        output_dim=4,
        activation="relu",
        seed=args.seed + 64032,
    )
    names = [
        "bp",
        "spc_t80",
        "pcalm_official_t64",
        *[f"pcalm_fixed_t{budget}" for budget in BUDGETS],
    ]
    models = {name: clone(base) for name in names}
    latest_dual_stats: dict[str, dict[str, float]] = {}
    rows: list[dict[str, object]] = []

    for update in range(args.updates + 1):
        for name, model in models.items():
            row: dict[str, object] = {
                "seed": args.seed,
                "method": name,
                "update": update,
                "eval_loss": loss(model, eval_x, eval_y),
            }
            row.update(latest_dual_stats.get(name, {}))
            rows.append(row)

        if update == args.updates:
            break

        idx = torch.arange(update * 4, update * 4 + 4) % 64
        x, y = train_x[idx], train_y[idx]

        bp_grads = method_grad(
            models["bp"],
            x,
            y,
            Schedule("bp", budget=0),
            state_lr=STATE_LR,
            rho=1.0,
        )
        apply(models["bp"], bp_grads, args.weight_lr)

        spc_grads = method_grad(
            models["spc_t80"],
            x,
            y,
            Schedule("pc", budget=80),
            state_lr=STATE_LR,
            rho=1.0,
        )
        if not all(torch.isfinite(grad).all() for grad in spc_grads):
            raise RuntimeError("sPC non-finite")
        apply(models["spc_t80"], spc_grads, args.weight_lr)

        official_grads, stats = run_alignment(
            models["pcalm_official_t64"],
            x,
            y,
            update_precision="fixed14_i1",
            state_precision="fixed16_i3",
            dual_precision="fixed12_i1",
            budget=64,
            state_lr=STATE_LR,
            dual_leak=0.0,
            alpha=1.0,
        )
        if not bool(stats["finite"]) or not all(
            torch.isfinite(grad).all() for grad in official_grads
        ):
            raise RuntimeError("pcalm_official_t64 non-finite")
        latest_dual_stats["pcalm_official_t64"] = dual_stats(stats)
        apply(models["pcalm_official_t64"], official_grads, args.weight_lr)

        for budget in BUDGETS:
            name = f"pcalm_fixed_t{budget}"
            grads, stats = run_alignment(
                models[name],
                x,
                y,
                update_precision="fixed14_i1",
                state_precision="fixed16_i3",
                dual_precision="fixed12_i1",
                budget=budget,
                state_lr=STATE_LR,
                dual_leak=DUAL_LEAK,
                alpha=ALPHA,
            )
            if not bool(stats["finite"]) or not all(
                torch.isfinite(grad).all() for grad in grads
            ):
                raise RuntimeError(f"{name} non-finite")
            latest_dual_stats[name] = dual_stats(stats)
            apply(models[name], grads, args.weight_lr)

    frame = pd.DataFrame(rows)
    initial = frame[frame["update"] == 0].set_index("method")["eval_loss"]
    frame["loss_ratio_to_initial"] = frame["eval_loss"] / frame["method"].map(initial)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame[frame["update"] == args.updates].to_string(index=False))


if __name__ == "__main__":
    main()
