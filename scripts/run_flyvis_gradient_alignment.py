from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from run_flyvis_retinotopic_contrastive import (
    DIRECTIONS,
    infer_sequence,
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


def _predict(
    circuit: RetinotopicFlyVisCircuit,
    state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    src, dst = circuit.graph.edge_index
    contribution = torch.tanh(state[:, src]) * weight
    pred = torch.zeros_like(state)
    pred = pred.index_add(1, dst, contribution)
    return pred + bias


def _differentiable_infer(
    circuit: RetinotopicFlyVisCircuit,
    initial: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    clamp_mask: torch.Tensor,
    clamp_values: torch.Tensor,
    steps: int,
    step_size: float,
) -> torch.Tensor:
    state = torch.where(clamp_mask[None, :], clamp_values, initial)
    src, dst = circuit.graph.edge_index
    for _ in range(steps):
        pred = _predict(circuit, state, weight, bias)
        eps = state - pred
        downstream = torch.zeros_like(state).index_add(1, src, eps[:, dst] * weight)
        activation_prime = 1.0 - torch.tanh(state).square()
        grad = eps - activation_prime * downstream
        grad = torch.where(clamp_mask[None, :], torch.zeros_like(grad), grad)
        state = state - step_size * grad
        state = torch.where(clamp_mask[None, :], clamp_values, state)
    return state


def oracle_descent_direction(
    model: PredictiveCodingGraph,
    circuit: RetinotopicFlyVisCircuit,
    stimulus: torch.Tensor,
    targets: torch.Tensor,
    *,
    frame_steps: int,
    step_size: float,
) -> tuple[torch.Tensor, float]:
    weight = model.weight.detach().clone().requires_grad_(True)
    bias = model.bias.detach().clone()
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
    readout = state[:, output_nodes(circuit)]
    loss = (readout - targets).square().mean()
    (grad,) = torch.autograd.grad(loss, weight)
    return -grad.detach(), float(loss.detach())


@torch.no_grad()
def temporal_contrastive_direction(
    model: PredictiveCodingGraph,
    circuit: RetinotopicFlyVisCircuit,
    stimulus: torch.Tensor,
    targets: torch.Tensor,
    *,
    beta: float,
    frame_steps: int,
    step_size: float,
) -> torch.Tensor:
    _, free_edge, _ = trajectory_statistics(
        model,
        circuit,
        stimulus,
        targets=None,
        beta=beta,
        frame_steps=frame_steps,
        step_size=step_size,
    )
    _, nudged_edge, _ = trajectory_statistics(
        model,
        circuit,
        stimulus,
        targets=targets,
        beta=beta,
        frame_steps=frame_steps,
        step_size=step_size,
    )
    return (nudged_edge - free_edge) / beta


@torch.no_grad()
def matched_endpoint_direction(
    model: PredictiveCodingGraph,
    circuit: RetinotopicFlyVisCircuit,
    stimulus: torch.Tensor,
    targets: torch.Tensor,
    *,
    beta: float,
    frame_steps: int,
    branch_steps: int,
    step_size: float,
) -> torch.Tensor:
    branch_start = infer_sequence(
        model,
        circuit,
        stimulus,
        frame_steps=frame_steps,
        step_size=step_size,
    )
    clamp_mask = torch.zeros(circuit.graph.num_nodes, dtype=torch.bool)
    clamp_mask[list(circuit.input_nodes)] = True
    clamp_values = torch.zeros_like(branch_start)
    clamp_values[:, list(circuit.input_nodes)] = stimulus[:, -1]
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
    return (model.local_edge_statistics(nudged_state) - model.local_edge_statistics(free_state)) / beta


@torch.no_grad()
def clamped_one_phase_direction(
    model: PredictiveCodingGraph,
    circuit: RetinotopicFlyVisCircuit,
    stimulus: torch.Tensor,
    targets: torch.Tensor,
    *,
    frame_steps: int,
    target_steps: int,
    step_size: float,
) -> torch.Tensor:
    state = infer_sequence(
        model,
        circuit,
        stimulus,
        frame_steps=frame_steps,
        step_size=step_size,
    )
    clamp_mask = torch.zeros(circuit.graph.num_nodes, dtype=torch.bool)
    clamp_mask[list(circuit.input_nodes)] = True
    clamp_mask[output_nodes(circuit)] = True
    clamp_values = torch.zeros_like(state)
    clamp_values[:, list(circuit.input_nodes)] = stimulus[:, -1]
    clamp_values[:, output_nodes(circuit)] = targets
    state, _ = model.infer(
        state,
        clamp_mask=clamp_mask,
        clamp_values=clamp_values,
        steps=target_steps,
        step_size=step_size,
    )
    return model.local_edge_statistics(state)


def vector_metrics(candidate: torch.Tensor, oracle: torch.Tensor) -> tuple[float, float, float]:
    candidate = candidate.float().flatten()
    oracle = oracle.float().flatten()
    candidate_norm = float(torch.linalg.vector_norm(candidate))
    oracle_norm = float(torch.linalg.vector_norm(oracle))
    denom = candidate_norm * oracle_norm
    cosine = float(torch.dot(candidate, oracle) / denom) if denom > 0.0 else float("nan")
    norm_ratio = candidate_norm / oracle_norm if oracle_norm > 0.0 else float("nan")
    sign_agreement = float((torch.sign(candidate) == torch.sign(oracle)).float().mean())
    return cosine, norm_ratio, sign_agreement


def run_one(
    circuit: RetinotopicFlyVisCircuit,
    *,
    topology: str,
    seed: int,
    direction: float,
    frames: int,
    width: float,
    frame_steps: int,
    branch_steps: int,
    step_size: float,
    beta: float,
) -> list[dict[str, float | int | str]]:
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    stimulus = render_motion_batch(
        circuit,
        [direction],
        frames=frames,
        width=width,
        jitter_seed=800_000 + 10_000 * seed + int(direction),
    )
    targets, _ = targets_for([direction])
    oracle, supervised_loss = oracle_descent_direction(
        model,
        circuit,
        stimulus,
        targets,
        frame_steps=frame_steps,
        step_size=step_size,
    )
    candidates = {
        "one_phase_clamped": clamped_one_phase_direction(
            model,
            circuit,
            stimulus,
            targets,
            frame_steps=frame_steps,
            target_steps=branch_steps,
            step_size=step_size,
        ),
        "matched_endpoint": matched_endpoint_direction(
            model,
            circuit,
            stimulus,
            targets,
            beta=beta,
            frame_steps=frame_steps,
            branch_steps=branch_steps,
            step_size=step_size,
        ),
        "temporal_contrastive": temporal_contrastive_direction(
            model,
            circuit,
            stimulus,
            targets,
            beta=beta,
            frame_steps=frame_steps,
            step_size=step_size,
        ),
    }
    rows: list[dict[str, float | int | str]] = []
    for rule, candidate in candidates.items():
        cosine, norm_ratio, sign_agreement = vector_metrics(candidate, oracle)
        rows.append(
            {
                "topology": topology,
                "seed": seed,
                "direction": direction,
                "rule": rule,
                "supervised_loss": supervised_loss,
                "cosine_to_oracle": cosine,
                "candidate_to_oracle_norm": norm_ratio,
                "sign_agreement": sign_agreement,
                "candidate_norm": float(torch.linalg.vector_norm(candidate)),
                "oracle_norm": float(torch.linalg.vector_norm(oracle)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare local PC update directions with exact gradients")
    parser.add_argument("--extent", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--branch-steps", type=int, default=32)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument(
        "--out", type=Path, default=Path("results/generated/flyvis_gradient_alignment.csv")
    )
    args = parser.parse_args()

    base = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=args.extent)
    rows: list[dict[str, float | int | str]] = []
    for seed in range(args.seeds):
        rewired = type_pair_preserving_rewire(
            base,
            swaps=max(base.graph.num_edges * 2, 1),
            seed=10_000 + seed,
        )
        for topology, circuit in (("biological", base), ("type_pair_rewire", rewired)):
            for direction in DIRECTIONS:
                rows.extend(
                    run_one(
                        circuit,
                        topology=topology,
                        seed=seed,
                        direction=direction,
                        frames=args.frames,
                        width=args.bar_width,
                        frame_steps=args.frame_steps,
                        branch_steps=args.branch_steps,
                        step_size=args.step_size,
                        beta=args.beta,
                    )
                )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = (
        frame.groupby(["topology", "rule"])[
            ["cosine_to_oracle", "candidate_to_oracle_norm", "sign_agreement"]
        ]
        .agg(["mean", "median", "std"])
        .sort_index()
    )
    print(summary.to_string())
    print("\nPer-instance cosine similarity:")
    print(
        frame.pivot_table(
            index=["topology", "seed", "direction"],
            columns="rule",
            values="cosine_to_oracle",
        ).to_string()
    )
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
