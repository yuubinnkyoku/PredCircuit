from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
import torch
from diagnose_local_gradient_trajectory import extended_metrics, oracle_descent_directions
from run_flyvis_retinotopic_contrastive import (
    DIRECTIONS,
    evaluate,
    render_motion_batch,
    targets_for,
)
from run_flyvis_retinotopic_temporal_adam import local_adam_step, temporal_local_direction

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import (
    RetinotopicFlyVisCircuit,
    graph_from_flyvis_retinotopy,
    type_pair_preserving_rewire,
)
from predcircuit.model import PredictiveCodingGraph


@torch.no_grad()
def adam_delta_preview(
    direction: torch.Tensor,
    first_moment: torch.Tensor,
    second_moment: torch.Tensor,
    *,
    step: int,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
) -> torch.Tensor:
    first = beta1 * first_moment + (1.0 - beta1) * direction
    second = beta2 * second_moment + (1.0 - beta2) * direction.square()
    first_hat = first / (1.0 - beta1**step)
    second_hat = second / (1.0 - beta2**step)
    return learning_rate * first_hat / (second_hat.sqrt() + epsilon)


def diagnose_checkpoint(
    model: PredictiveCodingGraph,
    circuit: RetinotopicFlyVisCircuit,
    *,
    topology: str,
    optimizer: str,
    seed: int,
    epoch: int,
    update_count: int,
    edge_m: torch.Tensor,
    edge_v: torch.Tensor,
    learning_rate: float,
    beta1: float,
    beta2: float,
    adam_epsilon: float,
    frames: int,
    width: float,
    frame_steps: int,
    step_size: float,
    beta: float,
    test_repeats: int,
) -> list[dict[str, float | int | str]]:
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
    rows: list[dict[str, float | int | str]] = []
    for direction in DIRECTIONS:
        stimulus = render_motion_batch(
            circuit,
            [direction],
            frames=frames,
            width=width,
            jitter_seed=800_000 + 10_000 * seed + int(direction),
        )
        targets, _ = targets_for([direction])
        oracle_edge, _, supervised_loss = oracle_descent_directions(
            model,
            circuit,
            stimulus,
            targets,
            frame_steps=frame_steps,
            step_size=step_size,
        )
        local_edge, _, output_shift = temporal_local_direction(
            model,
            circuit,
            stimulus,
            targets,
            beta=beta,
            frame_steps=frame_steps,
            step_size=step_size,
        )
        if optimizer == "adam":
            effective_edge = adam_delta_preview(
                local_edge,
                edge_m,
                edge_v,
                step=update_count + 1,
                learning_rate=learning_rate,
                beta1=beta1,
                beta2=beta2,
                epsilon=adam_epsilon,
            )
        else:
            effective_edge = learning_rate * local_edge

        raw = extended_metrics(local_edge, oracle_edge)
        effective = extended_metrics(effective_edge, oracle_edge)
        rows.append(
            {
                "topology": topology,
                "optimizer": optimizer,
                "seed": seed,
                "epoch": epoch,
                "direction": direction,
                "updates_so_far": update_count,
                "eval_mse": mse,
                "eval_accuracy": accuracy,
                "eval_margin": margin,
                "supervised_loss": supervised_loss,
                "output_shift": output_shift,
                "raw_cosine": raw["cosine"],
                "raw_norm_ratio": raw["norm_ratio"],
                "raw_scale_to_oracle": raw["scale_to_oracle"],
                "effective_cosine": effective["cosine"],
                "effective_norm_ratio": effective["norm_ratio"],
                "effective_scale_to_oracle": effective["scale_to_oracle"],
                "mean_abs_local_edge": float(local_edge.abs().mean()),
                "mean_abs_effective_edge": float(effective_edge.abs().mean()),
                "mean_abs_oracle_edge": float(oracle_edge.abs().mean()),
                "mean_abs_weight": float(model.weight.abs().mean()),
            }
        )
    return rows


def run_one(
    circuit: RetinotopicFlyVisCircuit,
    *,
    topology: str,
    optimizer: str,
    seed: int,
    epochs: int,
    checkpoints: set[int],
    frames: int,
    width: float,
    frame_steps: int,
    step_size: float,
    beta: float,
    learning_rate: float,
    beta1: float,
    beta2: float,
    adam_epsilon: float,
    test_repeats: int,
) -> list[dict[str, float | int | str]]:
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    edge_m = torch.zeros_like(model.weight)
    edge_v = torch.zeros_like(model.weight)
    bias_m = torch.zeros_like(model.bias)
    bias_v = torch.zeros_like(model.bias)
    update_count = 0
    rows: list[dict[str, float | int | str]] = []

    for epoch in range(epochs + 1):
        if epoch in checkpoints:
            rows.extend(
                diagnose_checkpoint(
                    model,
                    circuit,
                    topology=topology,
                    optimizer=optimizer,
                    seed=seed,
                    epoch=epoch,
                    update_count=update_count,
                    edge_m=edge_m,
                    edge_v=edge_v,
                    learning_rate=learning_rate,
                    beta1=beta1,
                    beta2=beta2,
                    adam_epsilon=adam_epsilon,
                    frames=frames,
                    width=width,
                    frame_steps=frame_steps,
                    step_size=step_size,
                    beta=beta,
                    test_repeats=test_repeats,
                )
            )
        if epoch == epochs:
            break

        directions = list(DIRECTIONS)
        random.Random(900_000 * seed + epoch).shuffle(directions)
        for sample_index, direction in enumerate(directions):
            stimulus = render_motion_batch(
                circuit,
                [direction],
                frames=frames,
                width=width,
                jitter_seed=100_000 * seed + 100 * epoch + sample_index,
            )
            targets, _ = targets_for([direction])
            edge_direction, bias_direction, _ = temporal_local_direction(
                model,
                circuit,
                stimulus,
                targets,
                beta=beta,
                frame_steps=frame_steps,
                step_size=step_size,
            )
            update_count += 1
            if optimizer == "adam":
                local_adam_step(
                    model.weight,
                    edge_direction,
                    edge_m,
                    edge_v,
                    step=update_count,
                    learning_rate=learning_rate,
                    beta1=beta1,
                    beta2=beta2,
                    epsilon=adam_epsilon,
                )
                local_adam_step(
                    model.bias,
                    bias_direction,
                    bias_m,
                    bias_v,
                    step=update_count,
                    learning_rate=learning_rate,
                    beta1=beta1,
                    beta2=beta2,
                    epsilon=adam_epsilon,
                )
            else:
                with torch.no_grad():
                    model.weight.add_(edge_direction, alpha=learning_rate)
                    model.bias.add_(bias_direction, alpha=learning_rate)
    return rows


def parse_checkpoints(text: str, epochs: int) -> set[int]:
    checkpoints = {int(value) for value in text.split(",") if value.strip()}
    checkpoints.update((0, epochs))
    invalid = sorted(value for value in checkpoints if value < 0 or value > epochs)
    if invalid:
        raise ValueError(f"checkpoints outside [0, {epochs}]: {invalid}")
    return checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track per-example local and effective update alignment during online PC training"
    )
    parser.add_argument("--extent", type=int, default=1)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--checkpoints", default="0,1,5,10,25,50,100")
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--test-repeats", type=int, default=4)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_online_gradient_scaling.csv"),
    )
    args = parser.parse_args()
    checkpoints = parse_checkpoints(args.checkpoints, args.epochs)

    biological = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=args.extent)
    rows: list[dict[str, float | int | str]] = []
    for seed in range(args.seeds):
        rewired = type_pair_preserving_rewire(
            biological,
            swaps=max(biological.graph.num_edges * 2, 1),
            seed=10_000 + seed,
        )
        for topology, circuit in (
            ("biological", biological),
            ("type_pair_rewire", rewired),
        ):
            rows.extend(
                run_one(
                    circuit,
                    topology=topology,
                    optimizer=args.optimizer,
                    seed=seed,
                    epochs=args.epochs,
                    checkpoints=checkpoints,
                    frames=args.frames,
                    width=args.bar_width,
                    frame_steps=args.frame_steps,
                    step_size=args.step_size,
                    beta=args.beta,
                    learning_rate=args.learning_rate,
                    beta1=args.beta1,
                    beta2=args.beta2,
                    adam_epsilon=args.adam_epsilon,
                    test_repeats=args.test_repeats,
                )
            )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = (
        frame.groupby(["topology", "epoch"])[
            [
                "raw_cosine",
                "effective_cosine",
                "raw_norm_ratio",
                "effective_norm_ratio",
                "mean_abs_local_edge",
                "mean_abs_effective_edge",
                "mean_abs_oracle_edge",
                "eval_mse",
                "eval_accuracy",
            ]
        ]
        .mean()
        .sort_index()
    )
    print(
        f"Online gradient scaling: extent={args.extent}, optimizer={args.optimizer}, "
        f"lr={args.learning_rate:g}, checkpoints={sorted(checkpoints)}"
    )
    print(summary.to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
