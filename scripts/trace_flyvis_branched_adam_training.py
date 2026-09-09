from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from diagnose_flyvis_branch_nudge_alignment import cycle_branched_credit
from diagnose_flyvis_temporal_ce_credit_alignment import cycle_ce_oracles, geometry
from run_flyvis_retinotopic_contrastive import (
    DIRECTIONS,
    infer_sequence,
    output_nodes,
    render_motion_batch,
    targets_for,
)
from run_flyvis_retinotopic_temporal_adam import local_adam_step

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def adam_delta(
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


@torch.no_grad()
def direction_metrics(
    model: PredictiveCodingGraph,
    circuit,
    *,
    seed: int,
    repeats: int,
    frames: int,
    width: float,
    frame_steps: int,
    step_size: float,
) -> dict[str, float]:
    directions = list(DIRECTIONS) * repeats
    stimulus = render_motion_batch(
        circuit,
        directions,
        frames=frames,
        width=width,
        jitter_seed=900_000 + seed,
    )
    _, classes = targets_for(directions)
    state = infer_sequence(
        model,
        circuit,
        stimulus,
        frame_steps=frame_steps,
        step_size=step_size,
    )
    readout = state[:, output_nodes(circuit)]
    predicted = readout.argmax(dim=1)
    correct = readout.gather(1, classes[:, None]).squeeze(1)
    masked = readout.clone()
    masked[torch.arange(len(classes)), classes] = -torch.inf
    margins = correct - masked.max(dim=1).values
    losses = F.cross_entropy(readout, classes, reduction="none")

    result: dict[str, float] = {}
    for class_index, direction in enumerate(DIRECTIONS):
        mask = classes == class_index
        key = str(direction).lower().replace(" ", "_")
        result[f"accuracy_{key}"] = float((predicted[mask] == classes[mask]).float().mean())
        result[f"margin_{key}"] = float(margins[mask].mean())
        result[f"cross_entropy_{key}"] = float(losses[mask].mean())
    return result


def credit_metrics(
    model: PredictiveCodingGraph,
    circuit,
    *,
    seed: int,
    frames: int,
    width: float,
    beta: float,
    frame_steps: int,
    nudge_steps: int,
    step_size: float,
    edge_m: torch.Tensor,
    edge_v: torch.Tensor,
    bias_m: torch.Tensor,
    bias_v: torch.Tensor,
    adam_step: int,
    learning_rate: float,
    beta1: float,
    beta2: float,
    adam_epsilon: float,
) -> dict[str, float]:
    edge, bias = cycle_branched_credit(
        model,
        circuit,
        seed=seed,
        epoch=0,
        frames=frames,
        width=width,
        beta=beta,
        frame_steps=frame_steps,
        nudge_steps=nudge_steps,
        step_size=step_size,
        terminal_only=False,
    )
    oracles = cycle_ce_oracles(
        model,
        circuit,
        seed=seed,
        epoch=0,
        frames=frames,
        width=width,
        frame_steps=frame_steps,
        step_size=step_size,
    )
    final_edge, final_bias = oracles["final_ce_oracle"]
    temporal_edge, temporal_bias = oracles["temporal_ce_oracle"]
    raw_final = geometry(edge, bias, final_edge, final_bias)
    raw_temporal = geometry(edge, bias, temporal_edge, temporal_bias)

    step = max(adam_step, 1)
    edge_delta = adam_delta(
        edge,
        edge_m,
        edge_v,
        step=step,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=adam_epsilon,
    )
    bias_delta = adam_delta(
        bias,
        bias_m,
        bias_v,
        step=step,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=adam_epsilon,
    )
    adam_final = geometry(edge_delta, bias_delta, final_edge, final_bias)
    adam_temporal = geometry(edge_delta, bias_delta, temporal_edge, temporal_bias)
    return {
        "raw_final_cosine": raw_final["cosine"],
        "raw_final_residual_fraction": raw_final["residual_fraction"],
        "raw_final_norm_ratio": raw_final["norm_ratio"],
        "raw_temporal_cosine": raw_temporal["cosine"],
        "adam_final_cosine": adam_final["cosine"],
        "adam_final_residual_fraction": adam_final["residual_fraction"],
        "adam_final_norm_ratio": adam_final["norm_ratio"],
        "adam_temporal_cosine": adam_temporal["cosine"],
    }


def parse_checkpoints(value: str) -> list[int]:
    checkpoints = sorted({int(item) for item in value.split(",")})
    if not checkpoints or checkpoints[0] < 0:
        raise ValueError("checkpoints must be non-negative integers")
    return checkpoints


def run_one(
    *,
    extent: int,
    seed: int,
    checkpoints: list[int],
    frames: int,
    width: float,
    frame_steps: int,
    nudge_steps: int,
    step_size: float,
    beta: float,
    learning_rate: float,
    beta1: float,
    beta2: float,
    adam_epsilon: float,
    test_repeats: int,
) -> list[dict[str, float | int | bool]]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=extent)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    edge_m = torch.zeros_like(model.weight)
    edge_v = torch.zeros_like(model.weight)
    bias_m = torch.zeros_like(model.bias)
    bias_v = torch.zeros_like(model.bias)
    rows: list[dict[str, float | int | bool]] = []
    previous_epoch = 0
    update_sum = 0.0
    update_count = 0

    for checkpoint in checkpoints:
        for epoch in range(previous_epoch, checkpoint):
            edge, bias = cycle_branched_credit(
                model,
                circuit,
                seed=seed,
                epoch=epoch,
                frames=frames,
                width=width,
                beta=beta,
                frame_steps=frame_steps,
                nudge_steps=nudge_steps,
                step_size=step_size,
                terminal_only=False,
            )
            edge_delta = local_adam_step(
                model.weight,
                edge,
                edge_m,
                edge_v,
                step=epoch + 1,
                learning_rate=learning_rate,
                beta1=beta1,
                beta2=beta2,
                epsilon=adam_epsilon,
            )
            local_adam_step(
                model.bias,
                bias,
                bias_m,
                bias_v,
                step=epoch + 1,
                learning_rate=learning_rate,
                beta1=beta1,
                beta2=beta2,
                epsilon=adam_epsilon,
            )
            update_sum += float(edge_delta.abs().mean())
            update_count += 1

        task = direction_metrics(
            model,
            circuit,
            seed=seed,
            repeats=test_repeats,
            frames=frames,
            width=width,
            frame_steps=frame_steps,
            step_size=step_size,
        )
        credit = credit_metrics(
            model,
            circuit,
            seed=seed,
            frames=frames,
            width=width,
            beta=beta,
            frame_steps=frame_steps,
            nudge_steps=nudge_steps,
            step_size=step_size,
            edge_m=edge_m,
            edge_v=edge_v,
            bias_m=bias_m,
            bias_v=bias_v,
            adam_step=checkpoint + 1,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            adam_epsilon=adam_epsilon,
        )
        rows.append(
            {
                "seed": seed,
                "epoch": checkpoint,
                "learning_rate": learning_rate,
                "weight_norm": float(torch.linalg.vector_norm(model.weight)),
                "bias_norm": float(torch.linalg.vector_norm(model.bias)),
                "mean_abs_update_since_previous": (
                    update_sum / update_count if update_count else 0.0
                ),
                **task,
                **credit,
                "finite": bool(torch.isfinite(model.weight).all()),
            }
        )
        previous_epoch = checkpoint
        update_sum = 0.0
        update_count = 0
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace task and exact-gradient geometry during branched local Adam learning"
    )
    parser.add_argument("--extent", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--checkpoints", type=str, default="0,10,25,50,100,200")
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--nudge-steps", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_branched_adam_training_trace.csv"),
    )
    args = parser.parse_args()
    checkpoints = parse_checkpoints(args.checkpoints)

    rows: list[dict[str, float | int | bool]] = []
    for seed in range(args.seeds):
        rows.extend(
            run_one(
                extent=args.extent,
                seed=seed,
                checkpoints=checkpoints,
                frames=args.frames,
                width=args.bar_width,
                frame_steps=args.frame_steps,
                nudge_steps=args.nudge_steps,
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
    summary = frame.groupby("epoch")[[
        "raw_final_cosine",
        "adam_final_cosine",
        "raw_temporal_cosine",
        "adam_temporal_cosine",
        "weight_norm",
        "mean_abs_update_since_previous",
    ]].agg(["mean", "median", "std"])
    print(
        f"Branched Adam trace: lr={args.learning_rate:g}, seeds={args.seeds}, "
        f"checkpoints={checkpoints}"
    )
    print(summary.to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
