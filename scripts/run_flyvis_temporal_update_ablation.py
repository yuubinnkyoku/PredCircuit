from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import pandas as pd
import torch
from run_flyvis_retinotopic_contrastive import (
    DIRECTIONS,
    evaluate,
    output_nodes,
    render_motion_batch,
    targets_for,
)
from run_flyvis_retinotopic_temporal_contrastive import trajectory_statistics

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import (
    RetinotopicFlyVisCircuit,
    graph_from_flyvis_retinotopy,
    type_pair_preserving_rewire,
)
from predcircuit.model import PredictiveCodingGraph


@torch.no_grad()
def ablated_step(
    model: PredictiveCodingGraph,
    circuit: RetinotopicFlyVisCircuit,
    stimulus: torch.Tensor,
    targets: torch.Tensor,
    *,
    beta: float,
    frame_steps: int,
    step_size: float,
    learning_rate: float,
    update_edges: bool,
    update_bias: bool,
) -> tuple[float, float]:
    free_state, free_edge, free_bias = trajectory_statistics(
        model,
        circuit,
        stimulus,
        targets=None,
        beta=beta,
        frame_steps=frame_steps,
        step_size=step_size,
    )
    nudged_state, nudged_edge, nudged_bias = trajectory_statistics(
        model,
        circuit,
        stimulus,
        targets=targets,
        beta=beta,
        frame_steps=frame_steps,
        step_size=step_size,
    )

    mean_update = 0.0
    if update_edges:
        edge_delta = learning_rate * (nudged_edge - free_edge) / beta
        edge_delta -= learning_rate * 1e-5 * model.weight
        edge_delta = edge_delta.clamp(-0.05, 0.05)
        model.weight.add_(edge_delta)
        mean_update = float(edge_delta.abs().mean())
    if update_bias:
        model.bias.add_((learning_rate / beta) * (nudged_bias - free_bias))

    output_shift = float(
        (nudged_state[:, output_nodes(circuit)] - free_state[:, output_nodes(circuit)]).abs().mean()
    )
    return mean_update, output_shift


def run_one(
    circuit: RetinotopicFlyVisCircuit,
    *,
    topology: str,
    mode: str,
    seed: int,
    epochs: int,
    frames: int,
    width: float,
    frame_steps: int,
    step_size: float,
    beta: float,
    learning_rate: float,
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
    update_edges = mode in {"full", "edges_only"}
    update_bias = mode in {"full", "bias_only"}
    mean_update_sum = 0.0
    output_shift_sum = 0.0
    updates = 0

    for epoch in range(epochs):
        directions = list(DIRECTIONS)
        random.Random(700_000 * seed + epoch).shuffle(directions)
        for sample_index, direction in enumerate(directions):
            stimulus = render_motion_batch(
                circuit,
                [direction],
                frames=frames,
                width=width,
                jitter_seed=100_000 * seed + 100 * epoch + sample_index,
            )
            targets, _ = targets_for([direction])
            mean_update, output_shift = ablated_step(
                model,
                circuit,
                stimulus,
                targets,
                beta=beta,
                frame_steps=frame_steps,
                step_size=step_size,
                learning_rate=learning_rate,
                update_edges=update_edges,
                update_bias=update_bias,
            )
            mean_update_sum += mean_update
            output_shift_sum += output_shift
            updates += 1

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
    return {
        "topology": topology,
        "mode": mode,
        "seed": seed,
        "learning_rate": learning_rate,
        "mse_before": mse_before,
        "mse_after": mse_after,
        "mse_improvement": mse_before - mse_after,
        "accuracy_before": accuracy_before,
        "accuracy_after": accuracy_after,
        "margin_before": margin_before,
        "margin_after": margin_after,
        "mean_abs_edge_update": mean_update_sum / max(updates, 1),
        "mean_output_nudge": output_shift_sum / max(updates, 1),
        "mean_abs_weight": float(model.weight.abs().mean()),
        "mean_abs_bias": float(model.bias.abs().mean()),
        "finite": bool(torch.isfinite(model.weight).all())
        and bool(torch.isfinite(model.bias).all())
        and math.isfinite(mse_after),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablate edge and bias plasticity in online temporal local PC"
    )
    parser.add_argument("--extent", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=10.0)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_temporal_update_ablation.csv"),
    )
    args = parser.parse_args()

    biological = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=args.extent)
    rows: list[dict[str, float | int | str | bool]] = []
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
            for mode in ("full", "edges_only", "bias_only"):
                rows.append(
                    run_one(
                        circuit,
                        topology=topology,
                        mode=mode,
                        seed=seed,
                        epochs=args.epochs,
                        frames=args.frames,
                        width=args.bar_width,
                        frame_steps=args.frame_steps,
                        step_size=args.step_size,
                        beta=args.beta,
                        learning_rate=args.learning_rate,
                        test_repeats=args.test_repeats,
                    )
                )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = (
        frame.groupby(["topology", "mode"])[
            ["mse_after", "mse_improvement", "accuracy_after", "margin_after"]
        ]
        .agg(["mean", "median", "std"])
        .sort_index()
    )
    print(f"Temporal local PC update ablation: lr={args.learning_rate:g}, seeds={args.seeds}")
    print(summary.to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
