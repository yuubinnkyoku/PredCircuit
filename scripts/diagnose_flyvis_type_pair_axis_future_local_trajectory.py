from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_axis_ce_geometry import candidate_directions
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit, evaluate
from run_flyvis_type_pair_consensus import type_pair_indices
from run_flyvis_type_pair_shuffle_control import shuffled_groups_like

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import RetinotopicFlyVisCircuit, graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


@torch.no_grad()
def clone_model(
    source: PredictiveCodingGraph,
    circuit: RetinotopicFlyVisCircuit,
    *,
    seed: int,
) -> PredictiveCodingGraph:
    clone = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    clone.weight.copy_(source.weight)
    clone.bias.copy_(source.bias)
    return clone


@torch.no_grad()
def perturb_model(
    model: PredictiveCodingGraph,
    direction: torch.Tensor,
    *,
    relative_step: float,
) -> float:
    base_norm = torch.linalg.vector_norm(model.weight).clamp_min(1e-30)
    unit = direction / torch.linalg.vector_norm(direction).clamp_min(1e-30)
    model.weight.add_(relative_step * base_norm * unit)
    model.weight.mul_(base_norm / torch.linalg.vector_norm(model.weight).clamp_min(1e-30))
    return float(torch.linalg.vector_norm(model.weight) / base_norm - 1.0)


def measure_future_trajectory(
    checkpoint_model: PredictiveCodingGraph,
    *,
    circuit: RetinotopicFlyVisCircuit,
    seed: int,
    checkpoint: int,
    biological_groups: list[torch.Tensor],
    shuffled_groups: list[torch.Tensor],
    learning_rate: float,
    max_update: float,
    relative_step: float,
    horizons: tuple[int, ...],
) -> list[dict[str, float | int | bool | str]]:
    edge, _ = credit(checkpoint_model, circuit, seed=seed, epoch=checkpoint)
    candidates = candidate_directions(edge, biological_groups, shuffled_groups)
    models = {
        "base": clone_model(checkpoint_model, circuit, seed=seed),
        **{
            condition: clone_model(checkpoint_model, circuit, seed=seed) for condition in candidates
        },
    }
    initial_norm_error = {condition: 0.0 for condition in models}
    for condition, (direction, _, _) in candidates.items():
        initial_norm_error[condition] = perturb_model(
            models[condition],
            direction,
            relative_step=relative_step,
        )

    target_horizons = set(horizons)
    max_horizon = max(horizons)
    rows: list[dict[str, float | int | bool | str]] = []
    for horizon in range(max_horizon + 1):
        if horizon in target_horizons:
            reference = models["base"]
            reference_weight_norm = torch.linalg.vector_norm(reference.weight).clamp_min(1e-30)
            reference_bias_norm = torch.linalg.vector_norm(reference.bias).clamp_min(1e-30)
            for condition, model in models.items():
                metrics = evaluate(model, circuit, seed)
                weight_norm = torch.linalg.vector_norm(model.weight)
                bias_delta = torch.linalg.vector_norm(model.bias - reference.bias)
                weight_delta = torch.linalg.vector_norm(model.weight - reference.weight)
                cosine = torch.dot(model.weight, reference.weight) / (
                    torch.linalg.vector_norm(model.weight).clamp_min(1e-30) * reference_weight_norm
                )
                rows.append(
                    {
                        "seed": seed,
                        "checkpoint": checkpoint,
                        "horizon": horizon,
                        "condition": condition,
                        "relative_step": relative_step,
                        "cross_entropy": metrics["cross_entropy"],
                        "accuracy": metrics["accuracy"],
                        "margin": metrics["margin"],
                        "initial_weight_norm_error": initial_norm_error[condition],
                        "relative_weight_norm_delta_vs_base": float(
                            weight_norm / reference_weight_norm - 1.0
                        ),
                        "relative_weight_vector_delta_vs_base": float(
                            weight_delta / reference_weight_norm
                        ),
                        "weight_cosine_vs_base": float(cosine),
                        "relative_bias_vector_delta_vs_base": float(
                            bias_delta / reference_bias_norm
                        ),
                        "finite": bool(torch.isfinite(model.weight).all())
                        and bool(torch.isfinite(model.bias).all())
                        and all(math.isfinite(float(metrics[name])) for name in metrics),
                    }
                )
        if horizon == max_horizon:
            break
        epoch = checkpoint + horizon
        for model in models.values():
            edge_direction, bias_direction = credit(model, circuit, seed=seed, epoch=epoch)
            apply_local_credit(
                model,
                edge_direction,
                bias_direction,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )
    return rows


def run_seed(
    *,
    seed: int,
    shuffle_seed: int,
    learning_rate: float,
    max_update: float,
    relative_step: float,
    horizons: tuple[int, ...],
) -> list[dict[str, float | int | bool | str]]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    biological_groups = type_pair_indices(circuit)
    shuffled_groups = shuffled_groups_like(
        biological_groups,
        edge_count=model.weight.numel(),
        shuffle_seed=shuffle_seed,
    )
    checkpoints = {0, 20, 100}
    rows: list[dict[str, float | int | bool | str]] = []
    for epoch in range(101):
        if epoch in checkpoints:
            rows.extend(
                measure_future_trajectory(
                    model,
                    circuit=circuit,
                    seed=seed,
                    checkpoint=epoch,
                    biological_groups=biological_groups,
                    shuffled_groups=shuffled_groups,
                    learning_rate=learning_rate,
                    max_update=max_update,
                    relative_step=relative_step,
                    horizons=horizons,
                )
            )
        if epoch == 100:
            break
        edge_direction, bias_direction = credit(model, circuit, seed=seed, epoch=epoch)
        apply_local_credit(
            model,
            edge_direction,
            bias_direction,
            learning_rate=learning_rate,
            weight_decay=0.0,
            max_update=max_update,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether a single type-pair-axis displacement enters a more favorable future "
            "trajectory under otherwise identical local learning"
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shuffle-seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--relative-step", type=float, default=0.01)
    parser.add_argument("--horizons", type=int, nargs="+", default=[0, 1, 5, 20])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.DataFrame(
        run_seed(
            seed=args.seed,
            shuffle_seed=args.shuffle_seed,
            learning_rate=args.learning_rate,
            max_update=args.max_update,
            relative_step=args.relative_step,
            horizons=tuple(args.horizons),
        )
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
