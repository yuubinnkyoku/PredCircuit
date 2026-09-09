from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_branch_nudge_alignment import cycle_branched_credit, free_frame_states
from diagnose_flyvis_temporal_ce_credit_alignment import cycle_ce_oracles, geometry
from run_flyvis_credit_residual_scaling import evaluate_metrics
from run_flyvis_retinotopic_contrastive import DIRECTIONS, output_nodes, render_motion_batch
from run_flyvis_temporal_coherence_gate import apply_local_credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def state_step_map(
    model: PredictiveCodingGraph,
    state: torch.Tensor,
    clamp_mask: torch.Tensor,
    clamp_values: torch.Tensor,
    *,
    step_size: float,
) -> torch.Tensor:
    mask = clamp_mask.unsqueeze(0)
    clamped = torch.where(mask, clamp_values, state)
    gradient = model._internal_gradient(clamped)
    gradient = torch.where(mask, torch.zeros_like(gradient), gradient)
    updated = clamped - step_size * gradient
    return torch.where(mask, clamp_values, updated)


def jacobian_metrics(
    model: PredictiveCodingGraph,
    state: torch.Tensor,
    clamp_mask: torch.Tensor,
    clamp_values: torch.Tensor,
    outs: list[int],
    *,
    step_size: float,
    seed: int,
) -> dict[str, float]:
    mask = clamp_mask.unsqueeze(0)

    def step_fn(current: torch.Tensor) -> torch.Tensor:
        return state_step_map(
            model,
            current,
            clamp_mask,
            clamp_values,
            step_size=step_size,
        )

    generator = torch.Generator().manual_seed(seed)
    vector = torch.randn(state.shape, generator=generator, dtype=state.dtype)
    vector = torch.where(mask, torch.zeros_like(vector), vector)
    vector = vector / torch.linalg.vector_norm(vector).clamp_min(1e-30)
    for _ in range(12):
        _, product = torch.autograd.functional.jvp(step_fn, state, vector)
        product = torch.where(mask, torch.zeros_like(product), product)
        vector = product / torch.linalg.vector_norm(product).clamp_min(1e-30)
    _, product = torch.autograd.functional.jvp(step_fn, state, vector)
    product = torch.where(mask, torch.zeros_like(product), product)
    dominant_magnitude = float(torch.linalg.vector_norm(product))
    dominant_rayleigh = float((vector * product).sum())

    outside_mask = torch.ones(state.shape[1], dtype=torch.bool)
    outside_mask[clamp_mask] = False
    outside_mask[outs] = False
    spread: dict[int, list[float]] = {1: [], 2: [], 4: [], 8: []}
    total: dict[int, list[float]] = {1: [], 2: [], 4: [], 8: []}
    for out in outs:
        tangent = torch.zeros_like(state)
        tangent[:, out] = 1.0
        for step in range(1, 9):
            _, tangent = torch.autograd.functional.jvp(step_fn, state, tangent)
            tangent = torch.where(mask, torch.zeros_like(tangent), tangent)
            if step in spread:
                spread[step].append(float(torch.linalg.vector_norm(tangent[:, outside_mask])))
                total[step].append(float(torch.linalg.vector_norm(tangent)))

    with torch.no_grad():
        activation_prime = model.activation_prime(state)
        saturation = (model.activation(state).abs() > 0.9).float()
        state_norm = float(torch.linalg.vector_norm(state))
        energy = float(model.energy(state))
    return {
        "step_jacobian_dominant_magnitude": dominant_magnitude,
        "step_jacobian_dominant_rayleigh": dominant_rayleigh,
        "state_norm": state_norm,
        "energy": energy,
        "mean_activation_prime": float(activation_prime.mean()),
        "saturation_fraction": float(saturation.mean()),
        **{f"output_spread_{step}": sum(values) / len(values) for step, values in spread.items()},
        **{
            f"output_total_gain_{step}": sum(values) / len(values) for step, values in total.items()
        },
    }


def checkpoint_state_metrics(
    model: PredictiveCodingGraph,
    circuit,
    *,
    seed: int,
    checkpoint: int,
) -> dict[str, float]:
    outs = output_nodes(circuit)
    rows: list[dict[str, float]] = []
    for sample_index, direction in enumerate(DIRECTIONS):
        stimulus = render_motion_batch(
            circuit,
            [direction],
            frames=7,
            width=0.5,
            jitter_seed=100_000 * seed + sample_index,
        )
        states, clamps, clamp_mask = free_frame_states(
            model,
            circuit,
            stimulus,
            frame_steps=2,
            step_size=0.015,
        )
        rows.append(
            jacobian_metrics(
                model,
                states[-1],
                clamp_mask,
                clamps[-1],
                outs,
                step_size=0.015,
                seed=10_000 * seed + 100 * checkpoint + sample_index,
            )
        )
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def run_seed(
    *,
    seed: int,
    checkpoints: list[int],
    learning_rate: float,
    train_nudge_steps: int,
    max_update: float,
    test_repeats: int,
) -> list[dict[str, float | int | bool]]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    rows: list[dict[str, float | int | bool]] = []
    previous_epoch = 0

    for checkpoint in checkpoints:
        for epoch in range(previous_epoch, checkpoint):
            edge_direction, bias_direction = cycle_branched_credit(
                model,
                circuit,
                seed=seed,
                epoch=epoch,
                frames=7,
                width=0.5,
                beta=0.03,
                frame_steps=2,
                nudge_steps=train_nudge_steps,
                step_size=0.015,
                terminal_only=False,
            )
            apply_local_credit(
                model,
                edge_direction,
                bias_direction,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )

        local_edge, local_bias = cycle_branched_credit(
            model,
            circuit,
            seed=seed,
            epoch=0,
            frames=7,
            width=0.5,
            beta=0.03,
            frame_steps=2,
            nudge_steps=1,
            step_size=0.015,
            terminal_only=False,
        )
        oracle_edge, oracle_bias = cycle_ce_oracles(
            model,
            circuit,
            seed=seed,
            epoch=0,
            frames=7,
            width=0.5,
            frame_steps=2,
            step_size=0.015,
        )["final_ce_oracle"]
        alignment = geometry(local_edge, local_bias, oracle_edge, oracle_bias)
        task = evaluate_metrics(
            model,
            circuit,
            repeats=test_repeats,
            frames=7,
            width=0.5,
            frame_steps=2,
            step_size=0.015,
            jitter_seed=900_000 + seed,
        )
        rows.append(
            {
                "seed": seed,
                "epoch": checkpoint,
                "learning_rate": learning_rate,
                "train_nudge_steps": train_nudge_steps,
                "final_cosine_nudge1": alignment["cosine"],
                "final_projection_nudge1": alignment["projection_coefficient"],
                "cross_entropy": task["cross_entropy"],
                "accuracy": task["accuracy"],
                "margin": task["margin"],
                "weight_norm": float(torch.linalg.vector_norm(model.weight)),
                "bias_norm": float(torch.linalg.vector_norm(model.bias)),
                **checkpoint_state_metrics(
                    model,
                    circuit,
                    seed=seed,
                    checkpoint=checkpoint,
                ),
                "finite": bool(torch.isfinite(model.weight).all())
                and bool(torch.isfinite(model.bias).all()),
            }
        )
        previous_epoch = checkpoint
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track state Jacobian and output perturbation propagation during local training"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--train-nudge-steps", type=int, default=2)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_checkpoint_state_jacobian.csv"),
    )
    args = parser.parse_args()

    frame = pd.DataFrame(
        run_seed(
            seed=args.seed,
            checkpoints=[0, 25, 50, 100],
            learning_rate=args.learning_rate,
            train_nudge_steps=args.train_nudge_steps,
            max_update=args.max_update,
            test_repeats=args.test_repeats,
        )
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
