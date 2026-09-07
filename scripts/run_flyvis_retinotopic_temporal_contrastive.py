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

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import (
    RetinotopicFlyVisCircuit,
    graph_from_flyvis_retinotopy,
    type_pair_preserving_rewire,
)
from predcircuit.model import PredictiveCodingGraph


@torch.no_grad()
def trajectory_statistics(
    model: PredictiveCodingGraph,
    circuit: RetinotopicFlyVisCircuit,
    stimulus: torch.Tensor,
    *,
    targets: torch.Tensor | None,
    beta: float,
    frame_steps: int,
    step_size: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = torch.zeros(stimulus.shape[0], circuit.graph.num_nodes, dtype=torch.float32)
    clamp_mask = torch.zeros(circuit.graph.num_nodes, dtype=torch.bool)
    clamp_mask[list(circuit.input_nodes)] = True
    edge_stats = torch.zeros(circuit.graph.num_edges, dtype=torch.float32)
    bias_stats = torch.zeros(circuit.graph.num_nodes, dtype=torch.float32)
    outs = output_nodes(circuit)

    for frame in stimulus.unbind(dim=1):
        clamp_values = torch.zeros_like(state)
        clamp_values[:, list(circuit.input_nodes)] = frame
        if targets is None:
            state, _ = model.infer(
                state,
                clamp_mask=clamp_mask,
                clamp_values=clamp_values,
                steps=frame_steps,
                step_size=step_size,
            )
        else:
            state = model.infer_nudged(
                state,
                clamp_mask=clamp_mask,
                clamp_values=clamp_values,
                nudged_nodes=outs,
                nudged_values=targets,
                beta=beta,
                steps=frame_steps,
                step_size=step_size,
            )
        edge_stats += model.local_edge_statistics(state)
        bias_stats += model.errors(state).mean(dim=0)

    scale = 1.0 / stimulus.shape[1]
    return state, edge_stats * scale, bias_stats * scale


@torch.no_grad()
def temporal_contrastive_step(
    model: PredictiveCodingGraph,
    circuit: RetinotopicFlyVisCircuit,
    stimulus: torch.Tensor,
    targets: torch.Tensor,
    *,
    beta: float,
    frame_steps: int,
    step_size: float,
    weight_lr: float,
) -> tuple[float, float, float]:
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

    delta = weight_lr * (nudged_edge - free_edge) / beta
    delta -= weight_lr * 1e-5 * model.weight
    delta = delta.clamp(-0.05, 0.05)
    model.weight.add_(delta.to(model.weight.device))
    model.bias.add_((weight_lr / beta) * (nudged_bias - free_bias).to(model.bias.device))
    output_shift = float(
        (nudged_state[:, output_nodes(circuit)] - free_state[:, output_nodes(circuit)]).abs().mean()
    )
    return float(delta.abs().mean()), float(delta.abs().max()), output_shift


def run_one(
    circuit: RetinotopicFlyVisCircuit,
    *,
    topology: str,
    seed: int,
    epochs: int,
    frames: int,
    width: float,
    frame_steps: int,
    step_size: float,
    beta: float,
    weight_lr: float,
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
    mean_nudge_sum = 0.0
    max_update = 0.0
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
            mean_update, step_max, output_shift = temporal_contrastive_step(
                model,
                circuit,
                stimulus,
                targets,
                beta=beta,
                frame_steps=frame_steps,
                step_size=step_size,
                weight_lr=weight_lr,
            )
            mean_update_sum += mean_update
            mean_nudge_sum += output_shift
            max_update = max(max_update, step_max)
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
        "learning_rule": "predictive_coding_temporal_contrastive",
        "topology": topology,
        "seed": seed,
        "epochs": epochs,
        "beta": beta,
        "weight_lr": weight_lr,
        "frame_steps": frame_steps,
        "mse_before": mse_before,
        "mse_after": mse_after,
        "mse_improvement": mse_before - mse_after,
        "accuracy_before": accuracy_before,
        "accuracy_after": accuracy_after,
        "margin_before": margin_before,
        "margin_after": margin_after,
        "mean_abs_update": mean_update_sum / max(updates, 1),
        "max_abs_update": max_update,
        "mean_output_nudge": mean_nudge_sum / max(updates, 1),
        "mean_abs_weight": float(model.weight.abs().mean()),
        "nodes": circuit.graph.num_nodes,
        "edges": circuit.graph.num_edges,
        "finite": bool(torch.isfinite(model.weight).all()) and math.isfinite(mse_after),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-pass temporal contrastive local learning on FlyVis motion"
    )
    parser.add_argument("--extent", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--weight-lr", type=float, default=3e-3)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_retinotopic_temporal_contrastive.csv"),
    )
    args = parser.parse_args()

    base = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=args.extent)
    rows: list[dict[str, float | int | str | bool]] = []
    for seed in range(args.seeds):
        rewired = type_pair_preserving_rewire(
            base,
            swaps=max(base.graph.num_edges * 2, 1),
            seed=10_000 + seed,
        )
        for topology, circuit in (("biological", base), ("type_pair_rewire", rewired)):
            rows.append(
                run_one(
                    circuit,
                    topology=topology,
                    seed=seed,
                    epochs=args.epochs,
                    frames=args.frames,
                    width=args.bar_width,
                    frame_steps=args.frame_steps,
                    step_size=args.step_size,
                    beta=args.beta,
                    weight_lr=args.weight_lr,
                    test_repeats=args.test_repeats,
                )
            )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = (
        frame.groupby("topology")[
            ["mse_after", "accuracy_after", "margin_after", "mean_abs_update"]
        ]
        .agg(["mean", "median", "std"])
        .sort_index()
    )
    print(
        f"Temporal contrastive PC: extent={args.extent}, {base.graph.num_nodes} nodes, "
        f"{base.graph.num_edges} edges, beta={args.beta:g}, lr={args.weight_lr:g}, "
        f"frame_steps={args.frame_steps}"
    )
    print(summary.to_string())
    print("\nPer-seed results:")
    print(
        frame[
            [
                "topology",
                "seed",
                "mse_before",
                "mse_after",
                "accuracy_before",
                "accuracy_after",
                "margin_after",
                "mean_abs_update",
                "mean_output_nudge",
                "finite",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
