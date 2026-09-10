from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_branch_nudge_alignment import cycle_branched_credit
from run_flyvis_retinotopic_contrastive import DIRECTIONS, infer_sequence, render_motion_batch
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_consensus import consensus_direction, type_pair_indices

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def differentiable_internal_gradient(
    circuit,
    state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    src, dst = circuit.graph.edge_index
    contribution = torch.tanh(state[:, src]) * weight
    prediction = torch.zeros_like(state).index_add(1, dst, contribution) + bias
    error = state - prediction
    downstream = torch.zeros_like(state).index_add(1, src, error[:, dst] * weight)
    activation_prime = 1.0 - torch.tanh(state).square()
    return error - activation_prime * downstream


def one_inference_step(
    circuit,
    state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    clamp_mask: torch.Tensor,
    clamp_values: torch.Tensor,
    step_size: float,
) -> torch.Tensor:
    clamped = torch.where(clamp_mask[None, :], clamp_values, state)
    gradient = differentiable_internal_gradient(circuit, clamped, weight, bias)
    gradient = torch.where(clamp_mask[None, :], torch.zeros_like(gradient), gradient)
    result = clamped - step_size * gradient
    return torch.where(clamp_mask[None, :], clamp_values, result)


def free_norm(value: torch.Tensor, free_mask: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(value[:, free_mask])


def top_jacobian_singular(
    circuit,
    state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    clamp_mask: torch.Tensor,
    clamp_values: torch.Tensor,
    step_size: float,
    iterations: int,
    seed: int,
) -> tuple[float, torch.Tensor]:
    free_mask = ~clamp_mask
    generator = torch.Generator().manual_seed(seed)
    vector = torch.randn(state.shape, generator=generator, dtype=state.dtype)
    vector[:, clamp_mask] = 0.0
    vector /= free_norm(vector, free_mask).clamp_min(1e-30)

    def step_fn(candidate: torch.Tensor) -> torch.Tensor:
        return one_inference_step(
            circuit,
            candidate,
            weight,
            bias,
            clamp_mask=clamp_mask,
            clamp_values=clamp_values,
            step_size=step_size,
        )

    sigma = float("nan")
    base = state.detach()
    for _ in range(iterations):
        base_req = base.clone().requires_grad_(True)
        output, jv = torch.autograd.functional.jvp(
            step_fn,
            base_req,
            vector,
            create_graph=True,
            strict=False,
        )
        jv = jv.clone()
        jv[:, clamp_mask] = 0.0
        sigma = float(free_norm(jv, free_mask))
        jt_j_v = torch.autograd.grad(
            torch.sum(output * jv.detach()),
            base_req,
        )[0]
        jt_j_v[:, clamp_mask] = 0.0
        norm = free_norm(jt_j_v, free_mask)
        if float(norm) <= 1e-30:
            break
        vector = (jt_j_v / norm).detach()

    base_req = base.clone().requires_grad_(True)
    _, jv = torch.autograd.functional.jvp(
        step_fn,
        base_req,
        vector,
        create_graph=False,
        strict=False,
    )
    jv[:, clamp_mask] = 0.0
    sigma = float(free_norm(jv, free_mask))
    return sigma, vector.detach()


@torch.no_grad()
def perturbation_amplification(
    model: PredictiveCodingGraph,
    state: torch.Tensor,
    direction: torch.Tensor,
    *,
    clamp_mask: torch.Tensor,
    clamp_values: torch.Tensor,
    step_size: float,
    epsilon: float,
    horizons: tuple[int, ...],
) -> dict[int, float]:
    free_mask = ~clamp_mask
    perturbation = epsilon * direction
    base = state.clone()
    perturbed = state + perturbation
    base[:, clamp_mask] = clamp_values[:, clamp_mask]
    perturbed[:, clamp_mask] = clamp_values[:, clamp_mask]
    initial_distance = free_norm(perturbed - base, free_mask).clamp_min(1e-30)
    results: dict[int, float] = {}
    maximum = max(horizons)
    for step in range(1, maximum + 1):
        base, _ = model.infer(
            base,
            clamp_mask=clamp_mask,
            clamp_values=clamp_values,
            steps=1,
            step_size=step_size,
        )
        perturbed, _ = model.infer(
            perturbed,
            clamp_mask=clamp_mask,
            clamp_values=clamp_values,
            steps=1,
            step_size=step_size,
        )
        if step in horizons:
            results[step] = float(
                free_norm(perturbed - base, free_mask) / initial_distance
            )
    return results


@torch.no_grad()
def state_statistics(
    model: PredictiveCodingGraph,
    state: torch.Tensor,
    *,
    free_mask: torch.Tensor,
) -> dict[str, float]:
    prediction = model.predict_nodes(state)
    error = state - prediction
    gradient = model._internal_gradient(state)
    activation_prime = model.activation_prime(state)
    free_state = state[:, free_mask]
    free_prediction = prediction[:, free_mask]
    free_error = error[:, free_mask]
    free_gradient = gradient[:, free_mask]
    free_prime = activation_prime[:, free_mask]
    return {
        "state_rms": float(torch.sqrt(free_state.square().mean())),
        "state_mean_abs": float(free_state.abs().mean()),
        "state_max_abs": float(free_state.abs().max()),
        "prediction_rms": float(torch.sqrt(free_prediction.square().mean())),
        "error_rms": float(torch.sqrt(free_error.square().mean())),
        "internal_gradient_rms": float(torch.sqrt(free_gradient.square().mean())),
        "activation_prime_mean": float(free_prime.mean()),
        "activation_prime_p10": float(torch.quantile(free_prime.flatten(), 0.10)),
        "activation_prime_p01": float(torch.quantile(free_prime.flatten(), 0.01)),
        "activation_prime_lt_0_5": float((free_prime < 0.5).float().mean()),
        "activation_prime_lt_0_2": float((free_prime < 0.2).float().mean()),
        "energy": float(model.energy(state)),
    }


def diagnostic_state(
    model: PredictiveCodingGraph,
    circuit,
    *,
    seed: int,
    frame_steps: int,
    step_size: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    stimulus = render_motion_batch(
        circuit,
        list(DIRECTIONS),
        frames=7,
        width=0.5,
        jitter_seed=1_700_000 + seed,
    )
    state = infer_sequence(
        model,
        circuit,
        stimulus,
        frame_steps=frame_steps,
        step_size=step_size,
    )
    clamp_mask = torch.zeros(circuit.graph.num_nodes, dtype=torch.bool)
    clamp_mask[list(circuit.input_nodes)] = True
    clamp_values = torch.zeros_like(state)
    clamp_values[:, list(circuit.input_nodes)] = stimulus[:, -1]
    return state, clamp_mask, clamp_values


def checkpoint_row(
    model: PredictiveCodingGraph,
    circuit,
    *,
    seed: int,
    trajectory_rule: str,
    checkpoint: int,
    frame_steps: int,
    step_size: float,
    power_iterations: int,
) -> dict[str, float | int | str | bool]:
    state, clamp_mask, clamp_values = diagnostic_state(
        model,
        circuit,
        seed=seed,
        frame_steps=frame_steps,
        step_size=step_size,
    )
    free_mask = ~clamp_mask
    weight = model.weight.detach()
    bias = model.bias.detach()
    sigma, singular_direction = top_jacobian_singular(
        circuit,
        state,
        weight,
        bias,
        clamp_mask=clamp_mask,
        clamp_values=clamp_values,
        step_size=step_size,
        iterations=power_iterations,
        seed=8_000_000 + 1000 * seed + checkpoint,
    )
    amplification = perturbation_amplification(
        model,
        state,
        singular_direction,
        clamp_mask=clamp_mask,
        clamp_values=clamp_values,
        step_size=step_size,
        epsilon=1e-3,
        horizons=(1, 2, 4, 8),
    )
    stats = state_statistics(model, state, free_mask=free_mask)
    return {
        "seed": seed,
        "trajectory_rule": trajectory_rule,
        "checkpoint": checkpoint,
        "jacobian_sigma_max": sigma,
        "perturb_amp_1": amplification[1],
        "perturb_amp_2": amplification[2],
        "perturb_amp_4": amplification[4],
        "perturb_amp_8": amplification[8],
        "weight_norm": float(torch.linalg.vector_norm(model.weight)),
        "bias_norm": float(torch.linalg.vector_norm(model.bias)),
        **stats,
        "finite": math.isfinite(sigma)
        and all(math.isfinite(value) for value in amplification.values()),
    }


def run_trajectory(
    *,
    seed: int,
    trajectory_rule: str,
    gamma: float,
    checkpoints: list[int],
    learning_rate: float,
    max_update: float,
    frame_steps: int,
    step_size: float,
    power_iterations: int,
) -> list[dict[str, float | int | str | bool]]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    groups = type_pair_indices(circuit)
    rows: list[dict[str, float | int | str | bool]] = []
    previous = 0
    for checkpoint in checkpoints:
        for epoch in range(previous, checkpoint):
            edge_direction, bias_direction = cycle_branched_credit(
                model,
                circuit,
                seed=seed,
                epoch=epoch,
                frames=7,
                width=0.5,
                beta=0.03,
                frame_steps=frame_steps,
                nudge_steps=2,
                step_size=step_size,
                terminal_only=False,
            )
            if trajectory_rule == "consensus":
                edge_direction = consensus_direction(
                    edge_direction,
                    groups,
                    gamma=gamma,
                )
            apply_local_credit(
                model,
                edge_direction,
                bias_direction,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )
        rows.append(
            checkpoint_row(
                model,
                circuit,
                seed=seed,
                trajectory_rule=trajectory_rule,
                checkpoint=checkpoint,
                frame_steps=frame_steps,
                step_size=step_size,
                power_iterations=power_iterations,
            )
        )
        previous = checkpoint
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare inference-map gain, saturation, and perturbation amplification along "
            "local versus weak type-pair-consensus training trajectories"
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--power-iterations", type=int, default=12)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, float | int | str | bool]] = []
    for trajectory_rule in ("local", "consensus"):
        rows.extend(
            run_trajectory(
                seed=args.seed,
                trajectory_rule=trajectory_rule,
                gamma=args.gamma,
                checkpoints=[0, 25, 50, 100],
                learning_rate=args.learning_rate,
                max_update=args.max_update,
                frame_steps=args.frame_steps,
                step_size=args.step_size,
                power_iterations=args.power_iterations,
            )
        )
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
