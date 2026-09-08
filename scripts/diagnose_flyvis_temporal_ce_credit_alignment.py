from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from run_flyvis_gradient_alignment import _differentiable_infer
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


def differentiable_frame_states(
    circuit: RetinotopicFlyVisCircuit,
    stimulus: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    frame_steps: int,
    step_size: float,
) -> list[torch.Tensor]:
    state = torch.zeros(stimulus.shape[0], circuit.graph.num_nodes, dtype=torch.float32)
    clamp_mask = torch.zeros(circuit.graph.num_nodes, dtype=torch.bool)
    clamp_mask[list(circuit.input_nodes)] = True
    clamp_values = torch.zeros_like(state)
    state = _differentiable_infer(
        circuit,
        state,
        weight,
        bias,
        clamp_mask=clamp_mask,
        clamp_values=clamp_values,
        steps=max(frame_steps * 2, 1),
        step_size=step_size,
    )

    states: list[torch.Tensor] = []
    for frame in stimulus.unbind(dim=1):
        clamp_values = torch.zeros_like(state)
        clamp_values[:, list(circuit.input_nodes)] = frame
        state = _differentiable_infer(
            circuit,
            state,
            weight,
            bias,
            clamp_mask=clamp_mask,
            clamp_values=clamp_values,
            steps=frame_steps,
            step_size=step_size,
        )
        states.append(state)
    return states


def temporal_ce_oracle_descent(
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
    states = differentiable_frame_states(
        circuit,
        stimulus,
        weight,
        bias,
        frame_steps=frame_steps,
        step_size=step_size,
    )
    outs = output_nodes(circuit)
    losses = [F.cross_entropy(state[:, outs], classes) for state in states]
    loss = torch.stack(losses).mean()
    edge_grad, bias_grad = torch.autograd.grad(loss, (weight, bias))
    return -edge_grad.detach(), -bias_grad.detach()


def cycle_temporal_ce_oracle(
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
        edge, bias = temporal_ce_oracle_descent(
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
    candidate_edge: torch.Tensor,
    candidate_bias: torch.Tensor,
    oracle_edge: torch.Tensor,
    oracle_bias: torch.Tensor,
) -> dict[str, float]:
    candidate, parallel, residual, _, cosine = projection_decomposition(
        candidate_edge,
        candidate_bias,
        oracle_edge,
        oracle_bias,
    )
    candidate_norm = torch.linalg.vector_norm(candidate).clamp_min(1e-30)
    return {
        "cosine": cosine,
        "residual_fraction": float(torch.linalg.vector_norm(residual) / candidate_norm),
        "parallel_fraction": float(torch.linalg.vector_norm(parallel) / candidate_norm),
        "candidate_norm": float(candidate_norm),
        "oracle_norm": float(
            torch.linalg.vector_norm(flatten_credit(oracle_edge, oracle_bias))
        ),
    }


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
    mse_local_edge, mse_local_bias, _, _ = cycle_directions(
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
    oracle_edge, oracle_bias = cycle_temporal_ce_oracle(
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
    for comparison, edge, bias in (
        ("mse_local_vs_temporal_ce_oracle", mse_local_edge, mse_local_bias),
        ("ce_local_vs_temporal_ce_oracle", ce_local_edge, ce_local_bias),
    ):
        rows.append(
            {
                "comparison": comparison,
                "seed": seed,
                "repeat": repeat,
                **geometry(edge, bias, oracle_edge, oracle_bias),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare MSE and classification local PC credit against an exact CE objective "
            "averaged over the same temporal frames"
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
        default=Path("results/generated/flyvis_temporal_ce_credit_alignment.csv"),
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
        f"Temporal CE alignment: extent={args.extent}, seeds={args.seeds}, "
        f"repeats={args.repeats}, {circuit.graph.num_nodes} nodes, {circuit.graph.num_edges} edges"
    )
    print(summary.to_string())

    per_seed = frame.groupby(["comparison", "seed"])["cosine"].mean().unstack("comparison")
    delta = (
        per_seed["ce_local_vs_temporal_ce_oracle"]
        - per_seed["mse_local_vs_temporal_ce_oracle"]
    )
    print("\nCE-local minus MSE-local cosine against temporal CE oracle:")
    print(f"mean={delta.mean():.6f}, median={delta.median():.6f}")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
