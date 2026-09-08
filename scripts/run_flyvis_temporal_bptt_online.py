from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import pandas as pd
import torch
from run_flyvis_exact_pc_gradient import differentiable_sequence, sync_model
from run_flyvis_retinotopic_contrastive import DIRECTIONS, output_nodes, render_motion_batch
from run_flyvis_temporal_target_alignment import (
    evaluate_mapping,
    shifted_classes,
    targets_for_mapping,
)

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import (
    RetinotopicFlyVisCircuit,
    graph_from_flyvis_retinotopy,
    type_pair_preserving_rewire,
)
from predcircuit.model import PredictiveCodingGraph


def run_one(
    circuit: RetinotopicFlyVisCircuit,
    *,
    topology: str,
    target_shift: int,
    seed: int,
    epochs: int,
    frames: int,
    width: float,
    frame_steps: int,
    step_size: float,
    learning_rate: float,
    test_repeats: int,
) -> dict[str, float | int | str | bool]:
    class_map = shifted_classes(target_shift)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    weight = torch.nn.Parameter(model.weight.detach().clone())
    bias = torch.nn.Parameter(model.bias.detach().clone())
    optimizer = torch.optim.Adam([weight, bias], lr=learning_rate)

    mse_before, accuracy_before, margin_before = evaluate_mapping(
        model,
        circuit,
        class_map=class_map,
        repeats=test_repeats,
        frames=frames,
        width=width,
        frame_steps=frame_steps,
        step_size=step_size,
        jitter_seed=900_000 + seed,
    )

    max_grad_norm = 0.0
    final_train_loss = float("nan")
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
            targets, _ = targets_for_mapping([direction], class_map)
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
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([weight, bias], 10.0)
            max_grad_norm = max(max_grad_norm, float(grad_norm))
            optimizer.step()
            final_train_loss = float(loss.detach())

    sync_model(model, weight, bias)
    mse_after, accuracy_after, margin_after = evaluate_mapping(
        model,
        circuit,
        class_map=class_map,
        repeats=test_repeats,
        frames=frames,
        width=width,
        frame_steps=frame_steps,
        step_size=step_size,
        jitter_seed=900_000 + seed,
    )
    return {
        "learning_rule": "bptt_online_exact_unrolled_pc",
        "optimizer": "adam",
        "topology": topology,
        "target_shift": target_shift,
        "target_map": ",".join(str(value) for value in class_map),
        "seed": seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "mse_before": mse_before,
        "mse_after": mse_after,
        "mse_improvement": mse_before - mse_after,
        "accuracy_before": accuracy_before,
        "accuracy_after": accuracy_after,
        "accuracy_improvement": accuracy_after - accuracy_before,
        "margin_before": margin_before,
        "margin_after": margin_after,
        "final_train_loss": final_train_loss,
        "max_grad_norm": max_grad_norm,
        "mean_abs_weight": float(weight.detach().abs().mean()),
        "nodes": circuit.graph.num_nodes,
        "edges": circuit.graph.num_edges,
        "finite": bool(torch.isfinite(weight).all())
        and bool(torch.isfinite(bias).all())
        and math.isfinite(mse_after),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Online BPTT control through the same FlyVis predictive-coding dynamics and "
            "sample schedule as local PC"
        )
    )
    parser.add_argument("--target-shift", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--extent", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_temporal_bptt_online.csv"),
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
            rows.append(
                run_one(
                    circuit,
                    topology=topology,
                    target_shift=args.target_shift,
                    seed=seed,
                    epochs=args.epochs,
                    frames=args.frames,
                    width=args.bar_width,
                    frame_steps=args.frame_steps,
                    step_size=args.step_size,
                    learning_rate=args.learning_rate,
                    test_repeats=args.test_repeats,
                )
            )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = (
        frame.groupby("topology")[["mse_after", "mse_improvement", "accuracy_after", "margin_after"]]
        .agg(["mean", "median", "std"])
        .sort_index()
    )
    print(
        f"Online BPTT target alignment: shift={args.target_shift}, "
        f"map={shifted_classes(args.target_shift)}, lr={args.learning_rate:g}"
    )
    print(summary.to_string())
    print("\nPer-seed results:")
    print(
        frame[
            [
                "topology",
                "target_shift",
                "seed",
                "mse_before",
                "mse_after",
                "accuracy_before",
                "accuracy_after",
                "margin_after",
                "max_grad_norm",
                "finite",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
