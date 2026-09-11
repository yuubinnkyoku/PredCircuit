from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_strict_curvature import exact_cycle_loss
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit
from run_flyvis_type_pair_consensus import type_pair_indices
from run_flyvis_type_pair_linear_mix import linear_mix_direction
from run_flyvis_type_pair_norm_matched_control import match_norm
from run_flyvis_type_pair_shuffle_control import shuffled_groups_like

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

FROZEN_EPOCH = 75
SAMPLE_EPOCHS = (75, 80, 85, 90, 95)


def run_seed(
    *,
    seed: int,
    shuffle_seed: int,
    learning_rate: float,
    max_update: float,
) -> list[dict[str, float | int | bool | str]]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    biological_groups = type_pair_indices(circuit)
    shuffled_groups = shuffled_groups_like(
        biological_groups,
        edge_count=model.weight.numel(),
        shuffle_seed=shuffle_seed,
    )

    for epoch in range(FROZEN_EPOCH):
        edge, bias = credit(model, circuit, seed=seed, epoch=epoch)
        apply_local_credit(
            model,
            edge,
            bias,
            learning_rate=learning_rate,
            weight_decay=0.0,
            max_update=max_update,
        )

    sums: dict[str, dict[str, float]] = {"type": {}, "shuffle": {}}
    finite = bool(torch.isfinite(model.weight).all()) and bool(
        torch.isfinite(model.bias).all()
    )

    for epoch in SAMPLE_EPOCHS:
        raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)

        weight = model.weight.detach().clone().requires_grad_(True)
        bias = model.bias.detach().clone().requires_grad_(True)
        loss = exact_cycle_loss(circuit, weight, bias, seed=seed, epoch=epoch)
        grad_edge, grad_bias = torch.autograd.grad(loss, (weight, bias), create_graph=True)
        baseline_loss = float(loss.detach())

        for grouping, groups in (
            ("type", biological_groups),
            ("shuffle", shuffled_groups),
        ):
            mixed = linear_mix_direction(
                raw_edge,
                groups,
                coarse_gain=4.0,
                residual_gain=1.0,
            )
            matched = match_norm(mixed, raw_edge)

            first_derivative = torch.dot(grad_edge, matched) + torch.dot(
                grad_bias, raw_bias
            )
            hvp_edge, hvp_bias = torch.autograd.grad(
                first_derivative,
                (weight, bias),
                retain_graph=True,
            )
            curvature = torch.dot(matched, hvp_edge) + torch.dot(raw_bias, hvp_bias)

            updated = copy.deepcopy(model)
            before_weight = updated.weight.detach().clone()
            before_bias = updated.bias.detach().clone()
            apply_local_credit(
                updated,
                matched,
                raw_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )
            after_loss = float(
                exact_cycle_loss(
                    circuit,
                    updated.weight,
                    updated.bias,
                    seed=seed,
                    epoch=epoch,
                ).detach()
            )
            actual_step_edge = updated.weight.detach() - before_weight
            actual_step_bias = updated.bias.detach() - before_bias
            step_norm_sq = torch.dot(actual_step_edge, actual_step_edge) + torch.dot(
                actual_step_bias, actual_step_bias
            )
            applied_first = torch.dot(grad_edge.detach(), actual_step_edge) + torch.dot(
                grad_bias.detach(), actual_step_bias
            )
            applied_derivative = torch.dot(grad_edge, actual_step_edge) + torch.dot(
                grad_bias, actual_step_bias
            )
            applied_hvp_edge, applied_hvp_bias = torch.autograd.grad(
                applied_derivative,
                (weight, bias),
            )
            applied_curvature = torch.dot(actual_step_edge, applied_hvp_edge) + torch.dot(
                actual_step_bias, applied_hvp_bias
            )
            second_order_prediction = applied_first + 0.5 * applied_curvature
            clip_fraction = float(
                torch.mean((torch.abs(learning_rate * matched) > max_update).float())
            )

            values = {
                "baseline_ce": baseline_loss,
                "post_update_ce": after_loss,
                "actual_ce_delta": after_loss - baseline_loss,
                "direction_first_derivative": float(first_derivative.detach()),
                "direction_curvature": float(curvature.detach()),
                "applied_first_order_delta": float(applied_first),
                "applied_second_order_term": float(0.5 * applied_curvature.detach()),
                "applied_second_order_prediction": float(second_order_prediction.detach()),
                "applied_step_norm": float(torch.sqrt(step_norm_sq)),
                "edge_clip_fraction": clip_fraction,
            }
            target = sums[grouping]
            for key, value in values.items():
                target[key] = target.get(key, 0.0) + value
            finite = finite and all(math.isfinite(value) for value in values.values())

    rows: list[dict[str, float | int | bool | str]] = []
    count = float(len(SAMPLE_EPOCHS))
    for grouping in ("type", "shuffle"):
        rows.append(
            {
                "seed": seed,
                "grouping": grouping,
                "sample_count": len(SAMPLE_EPOCHS),
                **{key: value / count for key, value in sums[grouping].items()},
                "finite": finite,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare biological type-pair and matched shuffled 4m+r directions using exact "
            "one-update CE and first/second-order loss geometry at a frozen local state"
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shuffle-seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.DataFrame(
        run_seed(
            seed=args.seed,
            shuffle_seed=args.shuffle_seed,
            learning_rate=args.learning_rate,
            max_update=args.max_update,
        )
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
