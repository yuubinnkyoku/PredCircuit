from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_branch_nudge_alignment import cycle_branched_credit
from run_flyvis_branched_classification_adam import run_circuit
from run_flyvis_classification_nudge_topology import circuit_for
from run_flyvis_credit_residual_scaling import evaluate_metrics
from run_flyvis_retinotopic_temporal_adam import local_adam_step
from trace_flyvis_branched_adam_training import credit_metrics, parse_checkpoints

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def trace_one(
    circuit,
    *,
    topology: str,
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
) -> list[dict[str, float | int | bool | str]]:
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    edge_m = torch.zeros_like(model.weight)
    edge_v = torch.zeros_like(model.weight)
    bias_m = torch.zeros_like(model.bias)
    bias_v = torch.zeros_like(model.bias)
    rows: list[dict[str, float | int | bool | str]] = []
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

        task = evaluate_metrics(
            model,
            circuit,
            repeats=test_repeats,
            frames=frames,
            width=width,
            frame_steps=frame_steps,
            step_size=step_size,
            jitter_seed=900_000 + seed,
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
                "topology": topology,
                "seed": seed,
                "epoch": checkpoint,
                "learning_rate": learning_rate,
                "accuracy": task["accuracy"],
                "cross_entropy": task["cross_entropy"],
                "margin": task["margin"],
                "mse": task["mse"],
                "weight_norm": float(torch.linalg.vector_norm(model.weight)),
                "bias_norm": float(torch.linalg.vector_norm(model.bias)),
                "mean_abs_update_since_previous": (
                    update_sum / update_count if update_count else 0.0
                ),
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
        description="Trace branched local Adam geometry across FlyVis topology nulls"
    )
    parser.add_argument(
        "--topology",
        choices=("biological", "type_pair_rewire", "pair_rotation"),
        required=True,
    )
    parser.add_argument("--extent", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--checkpoints", type=str, default="0,10,25,50,100")
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
        default=Path("results/generated/flyvis_branched_adam_topology_trace.csv"),
    )
    args = parser.parse_args()
    checkpoints = parse_checkpoints(args.checkpoints)

    spec = load_flyvis_spec()
    base = graph_from_flyvis_retinotopy(spec, extent=args.extent)
    rows: list[dict[str, float | int | bool | str]] = []
    for seed in range(args.seeds):
        circuit = circuit_for(
            args.topology,
            base,
            spec,
            extent=args.extent,
            seed=seed,
        )
        rows.extend(
            trace_one(
                circuit,
                topology=args.topology,
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
    metrics = [
        "accuracy",
        "cross_entropy",
        "margin",
        "raw_final_cosine",
        "adam_final_cosine",
        "raw_final_residual_fraction",
        "raw_final_norm_ratio",
        "adam_final_norm_ratio",
        "weight_norm",
    ]
    print(frame.groupby("epoch")[metrics].agg(["mean", "median", "std"]).to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
