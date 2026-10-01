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


def clone(base: ResidualMLP, width: int) -> ResidualMLP:
    model = ResidualMLP(
        depth=32,
        width=width,
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


def gradient_geometry(grads: list[torch.Tensor], bp_grads: list[torch.Tensor]) -> dict[str, float]:
    def cosine_or_nan(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
        a = a.reshape(-1)
        b = b.reshape(-1)
        a_norm = float(torch.linalg.vector_norm(a))
        b_norm = float(torch.linalg.vector_norm(b))
        if a_norm == 0.0 or b_norm == 0.0:
            return float("nan"), a_norm, b_norm
        cosine = float(torch.dot(a, b) / (a_norm * b_norm))
        return cosine, a_norm, b_norm

    flat = torch.cat([grad.reshape(-1) for grad in grads])
    flat_bp = torch.cat([grad.reshape(-1) for grad in bp_grads])
    cosine, grad_norm, bp_grad_norm = cosine_or_nan(flat, flat_bp)
    result = {
        "bp_grad_cosine": cosine,
        "grad_norm": grad_norm,
        "bp_grad_norm": bp_grad_norm,
    }
    for layer, (grad, bp_grad) in enumerate(zip(grads, bp_grads, strict=True)):
        cosine, grad_norm, bp_grad_norm = cosine_or_nan(grad, bp_grad)
        result[f"bp_grad_cosine_l{layer:02d}"] = cosine
        result[f"grad_norm_l{layer:02d}"] = grad_norm
        result[f"bp_grad_norm_l{layer:02d}"] = bp_grad_norm
    return result


def local_bp_grads(model: ResidualMLP, x: torch.Tensor, y: torch.Tensor) -> list[torch.Tensor]:
    return method_grad(
        model,
        x,
        y,
        Schedule("bp", budget=0),
        state_lr=STATE_LR,
        rho=1.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--weight-lr", type=float, default=1.0)
    parser.add_argument("--methods", nargs="+", default=None)
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
        width=args.width,
        input_dim=8,
        output_dim=4,
        activation="relu",
        seed=args.seed + 64032,
    )
    names = [
        "bp",
        "spc_t80",
        "pcalm_fp32_official_t64",
        "pcalm_fixed_official_t64",
        *[f"pcalm_fixed_t{budget}" for budget in BUDGETS],
    ]
    if args.methods is not None:
        unknown = sorted(set(args.methods) - set(names))
        if unknown:
            parser.error(f"unknown methods: {', '.join(unknown)}")
        names = args.methods
    models = {name: clone(base, args.width) for name in names}
    latest_dual_stats: dict[str, dict[str, float]] = {}
    latest_grad_stats: dict[str, dict[str, float]] = {}
    rows: list[dict[str, object]] = []

    for update in range(args.updates + 1):
        for name, model in models.items():
            row: dict[str, object] = {
                "seed": args.seed,
                "method": name,
                "update": update,
                "model_update": update,
                "diagnostic_update": update - 1 if update > 0 else pd.NA,
                "eval_loss": loss(model, eval_x, eval_y),
            }
            row.update(latest_dual_stats.get(name, {}))
            row.update(latest_grad_stats.get(name, {}))
            rows.append(row)

        if update == args.updates:
            break

        idx = torch.arange(update * 4, update * 4 + 4) % 64
        x, y = train_x[idx], train_y[idx]

        if "bp" in models:
            bp_grads = method_grad(
                models["bp"],
            x,
            y,
            Schedule("bp", budget=0),
            state_lr=STATE_LR,
            rho=1.0,
        )
            latest_grad_stats["bp"] = gradient_geometry(bp_grads, bp_grads)
            apply(models["bp"], bp_grads, args.weight_lr)

        if "spc_t80" in models:
            spc_bp_grads = local_bp_grads(models["spc_t80"], x, y)
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
            latest_grad_stats["spc_t80"] = gradient_geometry(spc_grads, spc_bp_grads)
            apply(models["spc_t80"], spc_grads, args.weight_lr)

        for name, update_precision, state_precision, dual_precision in (
            ("pcalm_fp32_official_t64", "fp32", "fp32", "fp32"),
            (
                "pcalm_fixed_official_t64",
                "fixed14_i1",
                "fixed16_i3",
                "fixed12_i1",
            ),
        ):
            if name not in models:
                continue
            official_bp_grads = local_bp_grads(models[name], x, y)
            official_grads, stats = run_alignment(
                models[name],
                x,
                y,
                update_precision=update_precision,
                state_precision=state_precision,
                dual_precision=dual_precision,
                budget=64,
                state_lr=STATE_LR,
                dual_leak=0.0,
                alpha=1.0,
            )
            if not bool(stats["finite"]) or not all(
                torch.isfinite(grad).all() for grad in official_grads
            ):
                raise RuntimeError(f"{name} non-finite")
            latest_dual_stats[name] = dual_stats(stats)
            latest_grad_stats[name] = gradient_geometry(official_grads, official_bp_grads)
            apply(models[name], official_grads, args.weight_lr)

        for budget in BUDGETS:
            name = f"pcalm_fixed_t{budget}"
            if name not in models:
                continue
            fixed_bp_grads = local_bp_grads(models[name], x, y)
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
            if not bool(stats["finite"]) or not all(torch.isfinite(grad).all() for grad in grads):
                raise RuntimeError(f"{name} non-finite")
            latest_dual_stats[name] = dual_stats(stats)
            latest_grad_stats[name] = gradient_geometry(grads, fixed_bp_grads)
            apply(models[name], grads, args.weight_lr)

    frame = pd.DataFrame(rows)
    initial = frame[frame["update"] == 0].set_index("method")["eval_loss"]
    frame["loss_ratio_to_initial"] = frame["eval_loss"] / frame["method"].map(initial)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame[frame["update"] == args.updates].to_string(index=False))


if __name__ == "__main__":
    main()
