from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_branch_nudge_alignment import cycle_branched_credit
from diagnose_flyvis_strict_curvature import exact_cycle_loss
from diagnose_flyvis_temporal_ce_credit_alignment import cycle_ce_oracles, geometry
from diagnose_flyvis_type_pair_credit_curvature import type_pair_indices
from run_flyvis_strict_residual_controls import match_projection_and_norm
from run_flyvis_temporal_coherence_gate import apply_local_credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def train_to_checkpoint(
    model: PredictiveCodingGraph,
    circuit,
    *,
    seed: int,
    checkpoint: int,
    learning_rate: float,
    max_update: float,
) -> None:
    for epoch in range(checkpoint):
        edge_direction, bias_direction = cycle_branched_credit(
            model,
            circuit,
            seed=seed,
            epoch=epoch,
            frames=7,
            width=0.5,
            beta=0.03,
            frame_steps=2,
            nudge_steps=2,
            step_size=0.015,
            terminal_only=False,
        )
        apply_local_credit(
            model,
            edge_direction,
            bias_direction,
            learning_rate=learning_rate,
            weight_decay=0.0,
            max_update=max_update,
        )


def filtered_edge_residual(
    residual_edge: torch.Tensor,
    oracle_edge: torch.Tensor,
    groups: dict[tuple[str, str], torch.Tensor],
    selected: set[tuple[str, str]],
) -> torch.Tensor:
    result = residual_edge.clone()
    for pair in selected:
        indices = groups[pair]
        target = residual_edge[indices]
        candidate = torch.ones_like(target) * target.mean()
        result[indices] = match_projection_and_norm(
            candidate,
            oracle_edge[indices],
            target,
        )
    return result


def clipped_update(
    direction: torch.Tensor,
    *,
    learning_rate: float,
    max_update: float,
) -> torch.Tensor:
    update = learning_rate * direction
    if max_update > 0.0:
        update = update.clamp(-max_update, max_update)
    return update


def run_diagnostic(
    *,
    seed: int,
    checkpoint: int,
    probe_epoch: int,
    train_learning_rate: float,
    probe_learning_rate: float,
    max_update: float,
) -> list[dict[str, float | int | str | bool]]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    train_to_checkpoint(
        model,
        circuit,
        seed=seed,
        checkpoint=checkpoint,
        learning_rate=train_learning_rate,
        max_update=max_update,
    )

    local_edge, local_bias = cycle_branched_credit(
        model,
        circuit,
        seed=seed,
        epoch=probe_epoch,
        frames=7,
        width=0.5,
        beta=0.03,
        frame_steps=2,
        nudge_steps=2,
        step_size=0.015,
        terminal_only=False,
    )
    oracle_edge, oracle_bias = cycle_ce_oracles(
        model,
        circuit,
        seed=seed,
        epoch=probe_epoch,
        frames=7,
        width=0.5,
        frame_steps=2,
        step_size=0.015,
    )["final_ce_oracle"]
    local_geometry = geometry(local_edge, local_bias, oracle_edge, oracle_bias)

    oracle_norm_sq = torch.dot(oracle_edge, oracle_edge) + torch.dot(
        oracle_bias, oracle_bias
    )
    alpha = (
        torch.dot(local_edge, oracle_edge) + torch.dot(local_bias, oracle_bias)
    ) / oracle_norm_sq.clamp_min(1e-30)
    parallel_edge = alpha * oracle_edge
    parallel_bias = alpha * oracle_bias
    residual_edge = local_edge - parallel_edge
    residual_bias = local_bias - parallel_bias

    weight = model.weight.detach().clone().requires_grad_(True)
    bias = model.bias.detach().clone().requires_grad_(True)
    exact_before = exact_cycle_loss(
        circuit,
        weight,
        bias,
        seed=seed,
        epoch=probe_epoch,
    )
    grad_edge, grad_bias = torch.autograd.grad(
        exact_before,
        (weight, bias),
        create_graph=True,
    )
    residual_derivative = torch.dot(grad_edge, residual_edge) + torch.dot(
        grad_bias, residual_bias
    )
    hvp_edge, hvp_bias = torch.autograd.grad(
        residual_derivative,
        (weight, bias),
    )
    total_residual_curvature = torch.dot(residual_edge, hvp_edge) + torch.dot(
        residual_bias, hvp_bias
    )
    total_cross_curvature = 2.0 * (
        torch.dot(parallel_edge, hvp_edge) + torch.dot(parallel_bias, hvp_bias)
    )

    local_edge_update = clipped_update(
        local_edge,
        learning_rate=probe_learning_rate,
        max_update=max_update,
    )
    local_bias_update = clipped_update(
        local_bias,
        learning_rate=probe_learning_rate,
        max_update=max_update,
    )
    with torch.no_grad():
        local_post_loss = exact_cycle_loss(
            circuit,
            model.weight + local_edge_update,
            model.bias + local_bias_update,
            seed=seed,
            epoch=probe_epoch,
        )
    local_update_geometry = geometry(
        local_edge_update,
        local_bias_update,
        oracle_edge,
        oracle_bias,
    )

    groups = type_pair_indices(circuit)
    rows: list[dict[str, float | int | str | bool]] = []
    selections: list[tuple[str, set[tuple[str, str]]]] = [
        (f"{source_type}->{target_type}", {(source_type, target_type)})
        for source_type, target_type in groups
    ]
    selections.append(("__all_type_pairs__", set(groups)))

    for label, selected in selections:
        filtered_residual_edge = filtered_edge_residual(
            residual_edge,
            oracle_edge,
            groups,
            selected,
        )
        direction_edge = parallel_edge + filtered_residual_edge
        direction_bias = parallel_bias + residual_bias
        filtered_geometry = geometry(
            direction_edge,
            direction_bias,
            oracle_edge,
            oracle_bias,
        )
        edge_update = clipped_update(
            direction_edge,
            learning_rate=probe_learning_rate,
            max_update=max_update,
        )
        bias_update = clipped_update(
            direction_bias,
            learning_rate=probe_learning_rate,
            max_update=max_update,
        )
        update_geometry = geometry(
            edge_update,
            bias_update,
            oracle_edge,
            oracle_bias,
        )
        with torch.no_grad():
            post_loss = exact_cycle_loss(
                circuit,
                model.weight + edge_update,
                model.bias + bias_update,
                seed=seed,
                epoch=probe_epoch,
            )

        if label == "__all_type_pairs__":
            indices = torch.arange(local_edge.numel())
            source_type = "__all__"
            target_type = "__all__"
        else:
            pair = next(iter(selected))
            indices = groups[pair]
            source_type, target_type = pair

        raw_change = torch.linalg.vector_norm(direction_edge - local_edge)
        local_norm = torch.linalg.vector_norm(local_edge)
        rows.append(
            {
                "seed": seed,
                "checkpoint": checkpoint,
                "probe_epoch": probe_epoch,
                "train_learning_rate": train_learning_rate,
                "probe_learning_rate": probe_learning_rate,
                "source_type": source_type,
                "target_type": target_type,
                "edge_count": int(indices.numel()),
                "exact_loss_before": float(exact_before.detach()),
                "local_post_loss": float(local_post_loss),
                "filtered_post_loss": float(post_loss),
                "filtered_loss_gain_vs_local": float(local_post_loss - post_loss),
                "residual_curvature_contribution": float(
                    torch.dot(residual_edge[indices], hvp_edge[indices])
                ),
                "cross_curvature_contribution": float(
                    2.0 * torch.dot(parallel_edge[indices], hvp_edge[indices])
                ),
                "total_residual_curvature": float(total_residual_curvature),
                "total_cross_curvature": float(total_cross_curvature),
                "residual_norm": float(torch.linalg.vector_norm(residual_edge[indices])),
                "relative_raw_direction_change": float(
                    raw_change / local_norm.clamp_min(1e-30)
                ),
                "raw_cosine_error": abs(
                    filtered_geometry["cosine"] - local_geometry["cosine"]
                ),
                "raw_norm_ratio_error": abs(
                    filtered_geometry["norm_ratio"] - local_geometry["norm_ratio"]
                ),
                "raw_projection_error": abs(
                    filtered_geometry["projection_coefficient"]
                    - local_geometry["projection_coefficient"]
                ),
                "applied_update_cosine": update_geometry["cosine"],
                "local_applied_update_cosine": local_update_geometry["cosine"],
                "applied_update_cosine_change": (
                    update_geometry["cosine"] - local_update_geometry["cosine"]
                ),
                "edge_clip_fraction_local": float(
                    (probe_learning_rate * local_edge).abs().gt(max_update).float().mean()
                )
                if max_update > 0.0
                else 0.0,
                "edge_clip_fraction_filtered": float(
                    (probe_learning_rate * direction_edge).abs().gt(max_update).float().mean()
                )
                if max_update > 0.0
                else 0.0,
                "finite": math.isfinite(float(post_loss)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure actual one-step exact-CE effects of strict residual mean filtering "
            "for each FlyVis cell-type pair"
        )
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=int, default=100)
    parser.add_argument("--probe-epoch", type=int, default=0)
    parser.add_argument("--train-learning-rate", type=float, default=160.0)
    parser.add_argument("--probe-learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_type_pair_intervention_loss.csv"),
    )
    args = parser.parse_args()

    frame = pd.DataFrame(
        run_diagnostic(
            seed=args.seed,
            checkpoint=args.checkpoint,
            probe_epoch=args.probe_epoch,
            train_learning_rate=args.train_learning_rate,
            probe_learning_rate=args.probe_learning_rate,
            max_update=args.max_update,
        )
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    pairs = frame[frame["source_type"] != "__all__"].copy()
    spearman = pairs[
        ["residual_curvature_contribution", "filtered_loss_gain_vs_local"]
    ].corr(method="spearman").iloc[0, 1]
    print(f"Spearman(old burden, actual one-step gain): {spearman:.6f}")
    print("\nTop beneficial pair interventions:")
    print(
        pairs.sort_values("filtered_loss_gain_vs_local", ascending=False)[
            [
                "source_type",
                "target_type",
                "edge_count",
                "filtered_loss_gain_vs_local",
                "residual_curvature_contribution",
                "relative_raw_direction_change",
            ]
        ].head(20).to_string(index=False)
    )
    print("\nMost harmful pair interventions:")
    print(
        pairs.sort_values("filtered_loss_gain_vs_local", ascending=True)[
            [
                "source_type",
                "target_type",
                "edge_count",
                "filtered_loss_gain_vs_local",
                "residual_curvature_contribution",
                "relative_raw_direction_change",
            ]
        ].head(20).to_string(index=False)
    )
    print("\nAll-pair intervention:")
    print(
        frame[frame["source_type"] == "__all__"][
            [
                "filtered_loss_gain_vs_local",
                "relative_raw_direction_change",
                "raw_cosine_error",
                "raw_norm_ratio_error",
                "raw_projection_error",
                "applied_update_cosine_change",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
