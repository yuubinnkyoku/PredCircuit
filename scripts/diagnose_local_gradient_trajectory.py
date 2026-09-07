from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from run_flyvis_exact_pc_gradient import differentiable_sequence
from run_flyvis_gradient_alignment import vector_metrics
from run_flyvis_retinotopic_contrastive import (
    DIRECTIONS,
    evaluate,
    output_nodes,
    render_motion_batch,
    targets_for,
)
from run_flyvis_retinotopic_temporal_adam import temporal_local_direction

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import (
    RetinotopicFlyVisCircuit,
    graph_from_flyvis_retinotopy,
    type_pair_preserving_rewire,
)
from predcircuit.model import PredictiveCodingGraph


def oracle_descent_directions(
    model: PredictiveCodingGraph,
    circuit: RetinotopicFlyVisCircuit,
    stimulus: torch.Tensor,
    targets: torch.Tensor,
    *,
    frame_steps: int,
    step_size: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    weight = model.weight.detach().clone().requires_grad_(True)
    bias = model.bias.detach().clone().requires_grad_(True)
    state = differentiable_sequence(
        circuit,
        stimulus,
        weight,
        bias,
        frame_steps=frame_steps,
        step_size=step_size,
    )
    readout = state[:, output_nodes(circuit)]
    loss = (readout - targets).square().mean()
    edge_grad, bias_grad = torch.autograd.grad(loss, (weight, bias))
    return -edge_grad.detach(), -bias_grad.detach(), float(loss.detach())


def extended_metrics(candidate: torch.Tensor, oracle: torch.Tensor) -> dict[str, float]:
    cosine, norm_ratio, sign_agreement = vector_metrics(candidate, oracle)
    candidate_flat = candidate.float().flatten()
    oracle_flat = oracle.float().flatten()
    candidate_sq = float(torch.dot(candidate_flat, candidate_flat))
    oracle_sq = float(torch.dot(oracle_flat, oracle_flat))
    dot = float(torch.dot(candidate_flat, oracle_flat))
    scale_to_oracle = dot / candidate_sq if candidate_sq > 0.0 else float("nan")
    oracle_projection = dot / oracle_sq if oracle_sq > 0.0 else float("nan")
    return {
        "cosine": cosine,
        "norm_ratio": norm_ratio,
        "sign_agreement": sign_agreement,
        "candidate_norm": math.sqrt(candidate_sq),
        "oracle_norm": math.sqrt(oracle_sq),
        "scale_to_oracle": scale_to_oracle,
        "oracle_projection": oracle_projection,
    }


def metric_row(
    *,
    topology: str,
    seed: int,
    epoch: int,
    component: str,
    local: torch.Tensor,
    oracle: torch.Tensor,
    supervised_loss: float,
    mse: float,
    accuracy: float,
    margin: float,
    output_shift: float,
    mean_abs_weight: float,
) -> dict[str, float | int | str]:
    metrics = extended_metrics(local, oracle)
    return {
        "topology": topology,
        "seed": seed,
        "epoch": epoch,
        "component": component,
        "supervised_loss": supervised_loss,
        "eval_mse": mse,
        "eval_accuracy": accuracy,
        "eval_margin": margin,
        "output_shift": output_shift,
        "mean_abs_weight": mean_abs_weight,
        **metrics,
    }


def diagnose(
    model: PredictiveCodingGraph,
    circuit: RetinotopicFlyVisCircuit,
    *,
    topology: str,
    seed: int,
    epoch: int,
    stimulus: torch.Tensor,
    targets: torch.Tensor,
    beta: float,
    frame_steps: int,
    step_size: float,
    frames: int,
    width: float,
    test_repeats: int,
) -> list[dict[str, float | int | str]]:
    oracle_edge, oracle_bias, supervised_loss = oracle_descent_directions(
        model,
        circuit,
        stimulus,
        targets,
        frame_steps=frame_steps,
        step_size=step_size,
    )
    local_edge, local_bias, output_shift = temporal_local_direction(
        model,
        circuit,
        stimulus,
        targets,
        beta=beta,
        frame_steps=frame_steps,
        step_size=step_size,
    )
    mse, accuracy, margin = evaluate(
        model,
        circuit,
        repeats=test_repeats,
        frames=frames,
        width=width,
        frame_steps=frame_steps,
        step_size=step_size,
        jitter_seed=900_000 + seed,
    )
    combined_local = torch.cat((local_edge.flatten(), local_bias.flatten()))
    combined_oracle = torch.cat((oracle_edge.flatten(), oracle_bias.flatten()))
    common = {
        "topology": topology,
        "seed": seed,
        "epoch": epoch,
        "supervised_loss": supervised_loss,
        "mse": mse,
        "accuracy": accuracy,
        "margin": margin,
        "output_shift": output_shift,
        "mean_abs_weight": float(model.weight.abs().mean()),
    }
    return [
        metric_row(
            component="edge",
            local=local_edge,
            oracle=oracle_edge,
            **common,
        ),
        metric_row(
            component="bias",
            local=local_bias,
            oracle=oracle_bias,
            **common,
        ),
        metric_row(
            component="combined",
            local=combined_local,
            oracle=combined_oracle,
            **common,
        ),
    ]


def run_one(
    circuit: RetinotopicFlyVisCircuit,
    *,
    topology: str,
    seed: int,
    epochs: int,
    checkpoints: set[int],
    frames: int,
    width: float,
    frame_steps: int,
    step_size: float,
    beta: float,
    learning_rate: float,
    train_repeats: int,
    test_repeats: int,
) -> list[dict[str, float | int | str]]:
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    rows: list[dict[str, float | int | str]] = []

    for epoch in range(epochs + 1):
        directions = list(DIRECTIONS) * train_repeats
        stimulus = render_motion_batch(
            circuit,
            directions,
            frames=frames,
            width=width,
            jitter_seed=100_000 * seed + epoch,
        )
        targets, _ = targets_for(directions)

        if epoch in checkpoints:
            rows.extend(
                diagnose(
                    model,
                    circuit,
                    topology=topology,
                    seed=seed,
                    epoch=epoch,
                    stimulus=stimulus,
                    targets=targets,
                    beta=beta,
                    frame_steps=frame_steps,
                    step_size=step_size,
                    frames=frames,
                    width=width,
                    test_repeats=test_repeats,
                )
            )
        if epoch == epochs:
            break

        with torch.no_grad():
            edge_direction, bias_direction, _ = temporal_local_direction(
                model,
                circuit,
                stimulus,
                targets,
                beta=beta,
                frame_steps=frame_steps,
                step_size=step_size,
            )
            model.weight.add_(edge_direction, alpha=learning_rate)
            model.bias.add_(bias_direction, alpha=learning_rate)

    return rows


def parse_checkpoints(text: str, epochs: int) -> set[int]:
    checkpoints = {int(value) for value in text.split(",") if value.strip()}
    checkpoints.add(0)
    checkpoints.add(epochs)
    invalid = sorted(value for value in checkpoints if value < 0 or value > epochs)
    if invalid:
        raise ValueError(f"checkpoints outside [0, {epochs}]: {invalid}")
    return checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track local-vs-exact PC gradient alignment during local training"
    )
    parser.add_argument("--extent", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--checkpoints", default="0,1,5,10,25,50,100")
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--train-repeats", type=int, default=4)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_local_gradient_trajectory.csv"),
    )
    args = parser.parse_args()
    checkpoints = parse_checkpoints(args.checkpoints, args.epochs)

    base = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=args.extent)
    rows: list[dict[str, float | int | str]] = []
    for seed in range(args.seeds):
        rewired = type_pair_preserving_rewire(
            base,
            swaps=max(base.graph.num_edges * 2, 1),
            seed=10_000 + seed,
        )
        for topology, circuit in (("biological", base), ("type_pair_rewire", rewired)):
            rows.extend(
                run_one(
                    circuit,
                    topology=topology,
                    seed=seed,
                    epochs=args.epochs,
                    checkpoints=checkpoints,
                    frames=args.frames,
                    width=args.bar_width,
                    frame_steps=args.frame_steps,
                    step_size=args.step_size,
                    beta=args.beta,
                    learning_rate=args.learning_rate,
                    train_repeats=args.train_repeats,
                    test_repeats=args.test_repeats,
                )
            )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    combined = frame[frame["component"] == "combined"]
    summary = (
        combined.groupby(["topology", "epoch"])[
            ["cosine", "norm_ratio", "eval_mse", "eval_accuracy", "eval_margin"]
        ]
        .mean()
        .sort_index()
    )
    print(
        f"Local-gradient trajectory: lr={args.learning_rate:g}, beta={args.beta:g}, "
        f"checkpoints={sorted(checkpoints)}"
    )
    print(summary.to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
