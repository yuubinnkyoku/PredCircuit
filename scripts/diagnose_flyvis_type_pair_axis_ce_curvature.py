from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from diagnose_flyvis_temporal_ce_credit_alignment import differentiable_frame_states
from diagnose_flyvis_type_pair_axis_ce_geometry import candidate_directions
from run_flyvis_retinotopic_contrastive import (
    DIRECTIONS,
    output_nodes,
    render_motion_batch,
    targets_for,
)
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit
from run_flyvis_type_pair_consensus import type_pair_indices
from run_flyvis_type_pair_shuffle_control import shuffled_groups_like

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import RetinotopicFlyVisCircuit, graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def cycle_ce_losses(
    circuit: RetinotopicFlyVisCircuit,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    seed: int,
    epoch: int,
    frames: int,
    width: float,
    frame_steps: int,
    step_size: float,
) -> dict[str, float]:
    final_losses: list[torch.Tensor] = []
    temporal_losses: list[torch.Tensor] = []
    directions = list(DIRECTIONS)
    random.Random(700_000 * seed + epoch).shuffle(directions)
    outs = output_nodes(circuit)
    for sample_index, direction in enumerate(directions):
        stimulus = render_motion_batch(
            circuit,
            [direction],
            frames=frames,
            width=width,
            jitter_seed=100_000 * seed + 100 * epoch + sample_index,
        )
        _, classes = targets_for([direction])
        states = differentiable_frame_states(
            circuit,
            stimulus,
            weight,
            bias,
            frame_steps=frame_steps,
            step_size=step_size,
        )
        losses = [F.cross_entropy(state[:, outs], classes) for state in states]
        final_losses.append(losses[-1])
        temporal_losses.append(torch.stack(losses).mean())
    return {
        "final_ce": float(torch.stack(final_losses).mean()),
        "temporal_ce": float(torch.stack(temporal_losses).mean()),
    }


def measure_checkpoint(
    model: PredictiveCodingGraph,
    *,
    circuit: RetinotopicFlyVisCircuit,
    seed: int,
    checkpoint: int,
    biological_groups: list[torch.Tensor],
    shuffled_groups: list[torch.Tensor],
    relative_steps: tuple[float, ...],
) -> list[dict[str, float | int | str]]:
    edge, _ = credit(model, circuit, seed=seed, epoch=checkpoint)
    candidates = candidate_directions(edge, biological_groups, shuffled_groups)
    base_weight = model.weight.detach().clone()
    base_bias = model.bias.detach().clone()
    base_losses = cycle_ce_losses(
        circuit,
        base_weight,
        base_bias,
        seed=seed,
        epoch=checkpoint,
        frames=7,
        width=0.5,
        frame_steps=2,
        step_size=0.015,
    )
    weight_norm = torch.linalg.vector_norm(base_weight).clamp_min(1e-30)
    rows: list[dict[str, float | int | str]] = []
    for condition, (direction, change, effective_gain) in candidates.items():
        unit = direction / torch.linalg.vector_norm(direction).clamp_min(1e-30)
        for relative_step in relative_steps:
            step_norm = float(weight_norm) * relative_step
            plus_losses = cycle_ce_losses(
                circuit,
                base_weight + step_norm * unit,
                base_bias,
                seed=seed,
                epoch=checkpoint,
                frames=7,
                width=0.5,
                frame_steps=2,
                step_size=0.015,
            )
            minus_losses = cycle_ce_losses(
                circuit,
                base_weight - step_norm * unit,
                base_bias,
                seed=seed,
                epoch=checkpoint,
                frames=7,
                width=0.5,
                frame_steps=2,
                step_size=0.015,
            )
            for objective in ("final_ce", "temporal_ce"):
                base = base_losses[objective]
                plus = plus_losses[objective]
                minus = minus_losses[objective]
                rows.append(
                    {
                        "seed": seed,
                        "checkpoint": checkpoint,
                        "condition": condition,
                        "objective": objective,
                        "relative_step": relative_step,
                        "step_norm": step_norm,
                        "relative_direction_change": change,
                        "effective_gain": effective_gain,
                        "base_loss": base,
                        "plus_loss": plus,
                        "minus_loss": minus,
                        "one_sided_improvement": base - plus,
                        "central_slope": (plus - minus) / (2.0 * step_norm),
                        "directional_curvature": (plus - 2.0 * base + minus) / (step_norm**2),
                    }
                )
    return rows


def run_seed(
    *,
    seed: int,
    shuffle_seed: int,
    learning_rate: float,
    max_update: float,
    relative_steps: tuple[float, ...],
) -> list[dict[str, float | int | str]]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    biological_groups = type_pair_indices(circuit)
    shuffled_groups = shuffled_groups_like(
        biological_groups,
        edge_count=model.weight.numel(),
        shuffle_seed=shuffle_seed,
    )
    rows: list[dict[str, float | int | str]] = []
    checkpoints = {0, 20, 100}
    for epoch in range(101):
        if epoch in checkpoints:
            rows.extend(
                measure_checkpoint(
                    model,
                    circuit=circuit,
                    seed=seed,
                    checkpoint=epoch,
                    biological_groups=biological_groups,
                    shuffled_groups=shuffled_groups,
                    relative_steps=relative_steps,
                )
            )
        if epoch == 100:
            break
        edge, bias = credit(model, circuit, seed=seed, epoch=epoch)
        apply_local_credit(
            model,
            edge,
            bias,
            learning_rate=learning_rate,
            weight_decay=0.0,
            max_update=max_update,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure finite-step CE response and directional curvature for type-pair versus "
            "angle-matched shuffled credit axes at identical local checkpoints"
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shuffle-seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--relative-steps", type=float, nargs="+", default=[0.001, 0.003, 0.01])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.DataFrame(
        run_seed(
            seed=args.seed,
            shuffle_seed=args.shuffle_seed,
            learning_rate=args.learning_rate,
            max_update=args.max_update,
            relative_steps=tuple(args.relative_steps),
        )
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
