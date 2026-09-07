from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import pandas as pd
import torch

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import (
    RetinotopicFlyVisCircuit,
    graph_from_flyvis_retinotopy,
    type_pair_preserving_rewire,
)
from predcircuit.model import PredictiveCodingGraph
from run_flyvis_retinotopic_contrastive import (
    DIRECTIONS,
    evaluate,
    infer_sequence,
    output_nodes,
    render_motion_batch,
    targets_for,
)


@torch.no_grad()
def matched_contrastive_local_step(
    model: PredictiveCodingGraph,
    circuit: RetinotopicFlyVisCircuit,
    stimulus: torch.Tensor,
    targets: torch.Tensor,
    *,
    frame_steps: int,
    common_steps: int,
    branch_steps: int,
    step_size: float,
    beta: float,
    weight_lr: float,
) -> tuple[float, float, float, float]:
    temporal_state = infer_sequence(
        model,
        circuit,
        stimulus,
        frame_steps=frame_steps,
        step_size=step_size,
    )
    clamp_mask = torch.zeros(circuit.graph.num_nodes, dtype=torch.bool)
    clamp_mask[list(circuit.input_nodes)] = True
    clamp_values = torch.zeros_like(temporal_state)
    clamp_values[:, list(circuit.input_nodes)] = stimulus[:, -1]

    branch_start = temporal_state
    if common_steps:
        branch_start, _ = model.infer(
            branch_start,
            clamp_mask=clamp_mask,
            clamp_values=clamp_values,
            steps=common_steps,
            step_size=step_size,
        )

    free_state, _ = model.infer(
        branch_start,
        clamp_mask=clamp_mask,
        clamp_values=clamp_values,
        steps=branch_steps,
        step_size=step_size,
    )
    nudged_state = model.infer_nudged(
        branch_start,
        clamp_mask=clamp_mask,
        clamp_values=clamp_values,
        nudged_nodes=output_nodes(circuit),
        nudged_values=targets,
        beta=beta,
        steps=branch_steps,
        step_size=step_size,
    )
    delta = model.contrastive_weight_step(
        free_state,
        nudged_state,
        beta=beta,
        learning_rate=weight_lr,
        weight_decay=1e-5,
        clip=0.05,
    )
    output_shift = float(
        (nudged_state[:, output_nodes(circuit)] - free_state[:, output_nodes(circuit)]).abs().mean()
    )
    return (
        float(model.energy(free_state)),
        float(delta.abs().mean()),
        float(delta.abs().max()),
        output_shift,
    )


def run_one(
    circuit: RetinotopicFlyVisCircuit,
    *,
    topology: str,
    seed: int,
    epochs: int,
    frames: int,
    width: float,
    frame_steps: int,
    common_steps: int,
    branch_steps: int,
    step_size: float,
    beta: float,
    weight_lr: float,
    train_batch_size: int,
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

    update_sum = 0.0
    output_shift_sum = 0.0
    max_abs_update = 0.0
    update_count = 0
    final_free_energy = float("nan")
    for epoch in range(epochs):
        directions = list(DIRECTIONS)
        random.Random(500_000 * seed + epoch).shuffle(directions)
        for batch_index, start in enumerate(range(0, len(directions), train_batch_size)):
            batch_directions = directions[start : start + train_batch_size]
            stimulus = render_motion_batch(
                circuit,
                batch_directions,
                frames=frames,
                width=width,
                jitter_seed=100_000 * seed + 100 * epoch + batch_index,
            )
            targets, _ = targets_for(batch_directions)
            final_free_energy, mean_update, step_max, output_shift = matched_contrastive_local_step(
                model,
                circuit,
                stimulus,
                targets,
                frame_steps=frame_steps,
                common_steps=common_steps,
                branch_steps=branch_steps,
                step_size=step_size,
                beta=beta,
                weight_lr=weight_lr,
            )
            update_sum += mean_update
            output_shift_sum += output_shift
            max_abs_update = max(max_abs_update, step_max)
            update_count += 1

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
        "learning_rule": "predictive_coding_matched_contrastive",
        "topology": topology,
        "seed": seed,
        "epochs": epochs,
        "beta": beta,
        "weight_lr": weight_lr,
        "train_batch_size": train_batch_size,
        "common_steps": common_steps,
        "branch_steps": branch_steps,
        "mse_before": mse_before,
        "mse_after": mse_after,
        "mse_improvement": mse_before - mse_after,
        "accuracy_before": accuracy_before,
        "accuracy_after": accuracy_after,
        "margin_before": margin_before,
        "margin_after": margin_after,
        "final_free_energy": final_free_energy,
        "mean_abs_update": update_sum / max(update_count, 1),
        "max_abs_update": max_abs_update,
        "mean_output_nudge": output_shift_sum / max(update_count, 1),
        "mean_abs_weight": float(model.weight.abs().mean()),
        "nodes": circuit.graph.num_nodes,
        "edges": circuit.graph.num_edges,
        "finite": bool(torch.isfinite(model.weight).all()) and math.isfinite(mse_after),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Matched free/nudged finite-time contrastive PC on retinotopic motion"
    )
    parser.add_argument("--extent", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--common-steps", type=int, default=0)
    parser.add_argument("--branch-steps", type=int, default=32)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--weight-lr", type=float, default=3e-3)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_retinotopic_matched_contrastive.csv"),
    )
    args = parser.parse_args()

    if args.train_batch_size <= 0 or args.train_batch_size > len(DIRECTIONS):
        raise ValueError("train_batch_size must be between 1 and 4")
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
                    common_steps=args.common_steps,
                    branch_steps=args.branch_steps,
                    step_size=args.step_size,
                    beta=args.beta,
                    weight_lr=args.weight_lr,
                    train_batch_size=args.train_batch_size,
                    test_repeats=args.test_repeats,
                )
            )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = (
        frame.groupby("topology")[["mse_after", "accuracy_after", "margin_after", "mean_abs_update"]]
        .agg(["mean", "median", "std"])
        .sort_index()
    )
    print(
        f"Matched contrastive PC: extent={args.extent}, {base.graph.num_nodes} nodes, "
        f"{base.graph.num_edges} edges, beta={args.beta:g}, lr={args.weight_lr:g}, "
        f"common={args.common_steps}, branch={args.branch_steps}"
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
