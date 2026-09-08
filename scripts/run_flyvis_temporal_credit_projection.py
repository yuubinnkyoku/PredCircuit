from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from run_flyvis_retinotopic_contrastive import evaluate
from run_flyvis_temporal_oracle_interpolation import cycle_directions

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import RetinotopicFlyVisCircuit, graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def flatten_credit(edge: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.cat((edge.flatten(), bias.flatten()))


def split_credit(
    vector: torch.Tensor,
    edge_template: torch.Tensor,
    bias_template: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_size = edge_template.numel()
    edge = vector[:edge_size].reshape_as(edge_template)
    bias = vector[edge_size:].reshape_as(bias_template)
    return edge, bias


def projection_decomposition(
    local_edge: torch.Tensor,
    local_bias: torch.Tensor,
    oracle_edge: torch.Tensor,
    oracle_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    local = flatten_credit(local_edge, local_bias)
    oracle = flatten_credit(oracle_edge, oracle_bias)
    oracle_norm_sq = torch.dot(oracle, oracle)
    coefficient = torch.dot(local, oracle) / oracle_norm_sq.clamp_min(1e-30)
    parallel = coefficient * oracle
    residual = local - parallel
    cosine = torch.dot(local, oracle) / (
        torch.linalg.vector_norm(local) * torch.linalg.vector_norm(oracle)
    ).clamp_min(1e-30)
    return local, parallel, residual, float(coefficient), float(cosine)


def choose_credit(
    *,
    mode: str,
    local: torch.Tensor,
    parallel: torch.Tensor,
    residual: torch.Tensor,
    oracle: torch.Tensor,
    residual_scale: float,
) -> torch.Tensor:
    if mode == "local":
        return local
    if mode == "projected":
        return parallel
    if mode == "projected_equal_norm":
        scale = torch.linalg.vector_norm(local) / torch.linalg.vector_norm(parallel).clamp_min(1e-30)
        return parallel * scale
    if mode == "residual":
        return residual
    if mode == "attenuated_residual":
        return parallel + residual_scale * residual
    if mode == "oracle":
        return oracle
    raise ValueError(f"unknown mode: {mode}")


@torch.no_grad()
def apply_credit(
    model: PredictiveCodingGraph,
    direction: torch.Tensor,
    *,
    learning_rate: float,
    weight_decay: float,
    max_update: float,
) -> tuple[float, float]:
    edge_direction, bias_direction = split_credit(direction, model.weight, model.bias)
    edge_update = learning_rate * (edge_direction - weight_decay * model.weight)
    bias_update = learning_rate * bias_direction
    if max_update > 0.0:
        edge_update.clamp_(-max_update, max_update)
        bias_update.clamp_(-max_update, max_update)
    model.weight.add_(edge_update)
    model.bias.add_(bias_update)
    return float(edge_update.abs().mean()), float(edge_update.abs().max())


def run_one(
    circuit: RetinotopicFlyVisCircuit,
    *,
    seed: int,
    mode: str,
    residual_scale: float,
    epochs: int,
    frames: int,
    width: float,
    frame_steps: int,
    step_size: float,
    beta: float,
    learning_rate: float,
    weight_decay: float,
    max_update: float,
    test_repeats: int,
) -> dict[str, float | int | str | bool]:
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    mse_before, accuracy_before, margin_before = evaluate(
        model,
        circuit,
        repeats=test_repeats,
        frames=frames,
        width=width,
        frame_steps=frame_steps,
        step_size=step_size,
        jitter_seed=900_000 + seed,
    )

    mean_update_sum = 0.0
    max_abs_update = 0.0
    cosine_sum = 0.0
    projection_coefficient_sum = 0.0
    residual_fraction_sum = 0.0
    parallel_fraction_sum = 0.0

    for epoch in range(epochs):
        local_edge, local_bias, oracle_edge, oracle_bias = cycle_directions(
            model,
            circuit,
            seed=seed,
            epoch=epoch,
            frames=frames,
            width=width,
            beta=beta,
            frame_steps=frame_steps,
            step_size=step_size,
        )
        local, parallel, residual, coefficient, cosine = projection_decomposition(
            local_edge,
            local_bias,
            oracle_edge,
            oracle_bias,
        )
        oracle = flatten_credit(oracle_edge, oracle_bias)
        direction = choose_credit(
            mode=mode,
            local=local,
            parallel=parallel,
            residual=residual,
            oracle=oracle,
            residual_scale=residual_scale,
        )
        mean_update, step_max = apply_credit(
            model,
            direction,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_update=max_update,
        )
        local_norm = torch.linalg.vector_norm(local).clamp_min(1e-30)
        mean_update_sum += mean_update
        max_abs_update = max(max_abs_update, step_max)
        cosine_sum += cosine
        projection_coefficient_sum += coefficient
        residual_fraction_sum += float(torch.linalg.vector_norm(residual) / local_norm)
        parallel_fraction_sum += float(torch.linalg.vector_norm(parallel) / local_norm)

    mse_after, accuracy_after, margin_after = evaluate(
        model,
        circuit,
        repeats=test_repeats,
        frames=frames,
        width=width,
        frame_steps=frame_steps,
        step_size=step_size,
        jitter_seed=900_000 + seed,
    )
    count = max(epochs, 1)
    return {
        "learning_rule": "credit_projection_diagnostic",
        "topology": "biological",
        "seed": seed,
        "mode": mode,
        "residual_scale": residual_scale,
        "epochs": epochs,
        "beta": beta,
        "learning_rate": learning_rate,
        "mse_before": mse_before,
        "mse_after": mse_after,
        "mse_improvement": mse_before - mse_after,
        "accuracy_before": accuracy_before,
        "accuracy_after": accuracy_after,
        "margin_before": margin_before,
        "margin_after": margin_after,
        "mean_local_oracle_cosine": cosine_sum / count,
        "mean_projection_coefficient": projection_coefficient_sum / count,
        "mean_parallel_norm_fraction": parallel_fraction_sum / count,
        "mean_residual_norm_fraction": residual_fraction_sum / count,
        "mean_abs_update": mean_update_sum / count,
        "max_abs_update": max_abs_update,
        "mean_abs_weight": float(model.weight.abs().mean()),
        "nodes": circuit.graph.num_nodes,
        "edges": circuit.graph.num_edges,
        "finite": bool(torch.isfinite(model.weight).all()) and math.isfinite(mse_after),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose cycle-summed local PC credit into components parallel and "
            "orthogonal to the exact descent direction"
        )
    )
    parser.add_argument("--extent", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--mode",
        choices=(
            "local",
            "projected",
            "projected_equal_norm",
            "residual",
            "attenuated_residual",
            "oracle",
        ),
        required=True,
    )
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument("--learning-rate", type=float, default=10.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_temporal_credit_projection.csv"),
    )
    args = parser.parse_args()
    if args.residual_scale < 0.0:
        raise ValueError("--residual-scale must be non-negative")

    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=args.extent)
    rows = [
        run_one(
            circuit,
            seed=seed,
            mode=args.mode,
            residual_scale=args.residual_scale,
            epochs=args.epochs,
            frames=args.frames,
            width=args.bar_width,
            frame_steps=args.frame_steps,
            step_size=args.step_size,
            beta=args.beta,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_update=args.max_update,
            test_repeats=args.test_repeats,
        )
        for seed in range(args.seeds)
    ]
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = frame[["mse_after", "accuracy_after", "margin_after"]].agg(
        ["mean", "median", "std"]
    )
    print(
        f"Credit projection: mode={args.mode}, residual_scale={args.residual_scale:g}, "
        f"extent={args.extent}, beta={args.beta:g}, lr={args.learning_rate:g}"
    )
    print(summary.to_string())
    print("\nPer-seed results:")
    print(
        frame[
            [
                "seed",
                "mode",
                "residual_scale",
                "mse_after",
                "accuracy_after",
                "margin_after",
                "mean_local_oracle_cosine",
                "mean_parallel_norm_fraction",
                "mean_residual_norm_fraction",
                "mean_abs_update",
                "finite",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
