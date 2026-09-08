from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from run_flyvis_exact_pc_gradient import differentiable_sequence
from run_flyvis_retinotopic_contrastive import (
    DIRECTIONS,
    output_nodes,
    render_motion_batch,
    targets_for,
)
from run_flyvis_temporal_classification_nudge import cycle_classification_credit
from run_flyvis_temporal_credit_projection import flatten_credit, projection_decomposition
from run_flyvis_temporal_oracle_interpolation import cycle_directions

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import RetinotopicFlyVisCircuit, graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def ce_oracle_descent(
    model: PredictiveCodingGraph,
    circuit: RetinotopicFlyVisCircuit,
    stimulus: torch.Tensor,
    classes: torch.Tensor,
    *,
    frame_steps: int,
    step_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    loss = F.cross_entropy(readout, classes)
    edge_grad, bias_grad = torch.autograd.grad(loss, (weight, bias))
    return -edge_grad.detach(), -bias_grad.detach()


def cycle_ce_oracle(
    model: PredictiveCodingGraph,
    circuit: RetinotopicFlyVisCircuit,
    *,
    seed: int,
    epoch: int,
    frames: int,
    width: float,
    frame_steps: int,
    step_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_sum = torch.zeros_like(model.weight)
    bias_sum = torch.zeros_like(model.bias)
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
        _, classes = targets_for([direction])
        edge, bias = ce_oracle_descent(
            model,
            circuit,
            stimulus,
            classes,
            frame_steps=frame_steps,
            step_size=step_size,
        )
        edge_sum += edge
        bias_sum += bias
    return edge_sum, bias_sum


def geometry(
    edge: torch.Tensor,
    bias: torch.Tensor,
    oracle_edge: torch.Tensor,
    oracle_bias: torch.Tensor,
) -> tuple[float, float, float]:
    local, parallel, residual, _, cosine = projection_decomposition(
        edge,
        bias,
        oracle_edge,
        oracle_bias,
    )
    local_norm = torch.linalg.vector_norm(local).clamp_min(1e-30)
    residual_fraction = float(torch.linalg.vector_norm(residual) / local_norm)
    parallel_fraction = float(torch.linalg.vector_norm(parallel) / local_norm)
    return cosine, residual_fraction, parallel_fraction


def measure(
    circuit: RetinotopicFlyVisCircuit,
    *,
    seed: int,
    repeat: int,
    frames: int,
    width: float,
    beta: float,
    frame_steps: int,
    step_size: float,
) -> list[dict[str, float | int | str]]:
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    mse_local_edge, mse_local_bias, mse_oracle_edge, mse_oracle_bias = cycle_directions(
        model,
        circuit,
        seed=seed,
        epoch=repeat,
        frames=frames,
        width=width,
        beta=beta,
        frame_steps=frame_steps,
        step_size=step_size,
    )
    ce_local_edge, ce_local_bias, _ = cycle_classification_credit(
        model,
        circuit,
        seed=seed,
        epoch=repeat,
        frames=frames,
        width=width,
        beta=beta,
        frame_steps=frame_steps,
        step_size=step_size,
    )
    ce_oracle_edge, ce_oracle_bias = cycle_ce_oracle(
        model,
        circuit,
        seed=seed,
        epoch=repeat,
        frames=frames,
        width=width,
        frame_steps=frame_steps,
        step_size=step_size,
    )

    rows: list[dict[str, float | int | str]] = []
    for name, edge, bias, oracle_edge, oracle_bias in (
        ("mse_local_vs_ce_oracle", mse_local_edge, mse_local_bias, ce_oracle_edge, ce_oracle_bias),
        ("ce_local_vs_ce_oracle", ce_local_edge, ce_local_bias, ce_oracle_edge, ce_oracle_bias),
        ("mse_local_vs_mse_oracle", mse_local_edge, mse_local_bias, mse_oracle_edge, mse_oracle_bias),
        ("ce_local_vs_mse_oracle", ce_local_edge, ce_local_bias, mse_oracle_edge, mse_oracle_bias),
        ("mse_oracle_vs_ce_oracle", mse_oracle_edge, mse_oracle_bias, ce_oracle_edge, ce_oracle_bias),
    ):
        cosine, residual_fraction, parallel_fraction = geometry(
            edge,
            bias,
            oracle_edge,
            oracle_bias,
        )
        rows.append(
            {
                "comparison": name,
                "seed": seed,
                "repeat": repeat,
                "cosine": cosine,
                "residual_fraction": residual_fraction,
                "parallel_fraction": parallel_fraction,
                "candidate_norm": float(torch.linalg.vector_norm(flatten_credit(edge, bias))),
                "oracle_norm": float(
                    torch.linalg.vector_norm(flatten_credit(oracle_edge, oracle_bias))
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare MSE-nudged and cross-entropy-nudged local predictive-coding credit "
            "against exact MSE and classification BPTT directions"
        )
    )
    parser.add_argument("--extent", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_classification_credit_alignment.csv"),
    )
    args = parser.parse_args()

    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=args.extent)
    rows: list[dict[str, float | int | str]] = []
    for seed in range(args.seeds):
        for repeat in range(args.repeats):
            rows.extend(
                measure(
                    circuit,
                    seed=seed,
                    repeat=repeat,
                    frames=args.frames,
                    width=args.bar_width,
                    beta=args.beta,
                    frame_steps=args.frame_steps,
                    step_size=args.step_size,
                )
            )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = frame.groupby("comparison")[["cosine", "residual_fraction"]].agg(
        ["mean", "median", "std"]
    )
    print(
        f"Classification credit alignment: extent={args.extent}, seeds={args.seeds}, "
        f"repeats={args.repeats}, {circuit.graph.num_nodes} nodes, {circuit.graph.num_edges} edges"
    )
    print(summary.to_string())

    per_seed = frame.groupby(["comparison", "seed"])["cosine"].mean().unstack("comparison")
    delta = per_seed["ce_local_vs_ce_oracle"] - per_seed["mse_local_vs_ce_oracle"]
    print("\nCE-local minus MSE-local cosine against CE oracle:")
    print(f"mean={delta.mean():.6f}, median={delta.median():.6f}")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
