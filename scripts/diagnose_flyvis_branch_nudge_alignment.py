from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_temporal_ce_credit_alignment import cycle_ce_oracles, geometry
from run_flyvis_retinotopic_contrastive import (
    DIRECTIONS,
    output_nodes,
    render_motion_batch,
    targets_for,
)
from run_flyvis_temporal_classification_nudge import (
    cycle_classification_credit,
    infer_classification_nudged,
)

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import RetinotopicFlyVisCircuit, graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


@torch.no_grad()
def free_frame_states(
    model: PredictiveCodingGraph,
    circuit: RetinotopicFlyVisCircuit,
    stimulus: torch.Tensor,
    *,
    frame_steps: int,
    step_size: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    state = torch.zeros(stimulus.shape[0], circuit.graph.num_nodes, dtype=torch.float32)
    clamp_mask = torch.zeros(circuit.graph.num_nodes, dtype=torch.bool)
    clamp_mask[list(circuit.input_nodes)] = True
    states: list[torch.Tensor] = []
    clamps: list[torch.Tensor] = []

    for frame in stimulus.unbind(dim=1):
        clamp_values = torch.zeros_like(state)
        clamp_values[:, list(circuit.input_nodes)] = frame
        state, _ = model.infer(
            state,
            clamp_mask=clamp_mask,
            clamp_values=clamp_values,
            steps=frame_steps,
            step_size=step_size,
        )
        states.append(state.clone())
        clamps.append(clamp_values)
    return states, clamps, clamp_mask


@torch.no_grad()
def branched_local_direction(
    model: PredictiveCodingGraph,
    circuit: RetinotopicFlyVisCircuit,
    stimulus: torch.Tensor,
    classes: torch.Tensor,
    *,
    beta: float,
    frame_steps: int,
    nudge_steps: int,
    step_size: float,
    terminal_only: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    states, clamps, clamp_mask = free_frame_states(
        model,
        circuit,
        stimulus,
        frame_steps=frame_steps,
        step_size=step_size,
    )
    indices = [len(states) - 1] if terminal_only else list(range(len(states)))
    edge_sum = torch.zeros_like(model.weight)
    bias_sum = torch.zeros_like(model.bias)
    outs = output_nodes(circuit)

    for index in indices:
        branch_state = states[index]
        continued_free_state, _ = model.infer(
            branch_state,
            clamp_mask=clamp_mask,
            clamp_values=clamps[index],
            steps=nudge_steps,
            step_size=step_size,
        )
        nudged_state = infer_classification_nudged(
            model,
            branch_state,
            clamp_mask=clamp_mask,
            clamp_values=clamps[index],
            nudged_nodes=outs,
            classes=classes,
            beta=beta,
            steps=nudge_steps,
            step_size=step_size,
        )
        free_edge = model.local_edge_statistics(continued_free_state)
        nudged_edge = model.local_edge_statistics(nudged_state)
        free_bias = model.errors(continued_free_state).mean(dim=0)
        nudged_bias = model.errors(nudged_state).mean(dim=0)
        edge_sum += (nudged_edge - free_edge) / beta
        bias_sum += (nudged_bias - free_bias) / beta

    scale = 1.0 / len(indices)
    return edge_sum * scale, bias_sum * scale


def cycle_branched_credit(
    model: PredictiveCodingGraph,
    circuit: RetinotopicFlyVisCircuit,
    *,
    seed: int,
    epoch: int,
    frames: int,
    width: float,
    beta: float,
    frame_steps: int,
    nudge_steps: int,
    step_size: float,
    terminal_only: bool,
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
        edge, bias = branched_local_direction(
            model,
            circuit,
            stimulus,
            classes,
            beta=beta,
            frame_steps=frame_steps,
            nudge_steps=nudge_steps,
            step_size=step_size,
            terminal_only=terminal_only,
        )
        edge_sum += edge
        bias_sum += bias
    return edge_sum, bias_sum


def measure(
    circuit: RetinotopicFlyVisCircuit,
    *,
    seed: int,
    repeat: int,
    frames: int,
    width: float,
    beta: float,
    frame_steps: int,
    nudge_steps: int,
    step_size: float,
) -> list[dict[str, float | int | str]]:
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    current_edge, current_bias, _ = cycle_classification_credit(
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
    temporal_edge, temporal_bias = cycle_branched_credit(
        model,
        circuit,
        seed=seed,
        epoch=repeat,
        frames=frames,
        width=width,
        beta=beta,
        frame_steps=frame_steps,
        nudge_steps=nudge_steps,
        step_size=step_size,
        terminal_only=False,
    )
    terminal_edge, terminal_bias = cycle_branched_credit(
        model,
        circuit,
        seed=seed,
        epoch=repeat,
        frames=frames,
        width=width,
        beta=beta,
        frame_steps=frame_steps,
        nudge_steps=nudge_steps,
        step_size=step_size,
        terminal_only=True,
    )
    oracles = cycle_ce_oracles(
        model,
        circuit,
        seed=seed,
        epoch=repeat,
        frames=frames,
        width=width,
        frame_steps=frame_steps,
        step_size=step_size,
    )
    locals_ = {
        "independent_trajectory": (current_edge, current_bias),
        "per_frame_branch": (temporal_edge, temporal_bias),
        "terminal_branch": (terminal_edge, terminal_bias),
    }

    rows: list[dict[str, float | int | str]] = []
    for local_name, (edge, bias) in locals_.items():
        for oracle_name, (oracle_edge, oracle_bias) in oracles.items():
            rows.append(
                {
                    "comparison": f"{local_name}_vs_{oracle_name}",
                    "local_rule": local_name,
                    "oracle_objective": oracle_name,
                    "seed": seed,
                    "repeat": repeat,
                    "frame_steps": frame_steps,
                    "nudge_steps": nudge_steps,
                    **geometry(edge, bias, oracle_edge, oracle_bias),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare independent free/nudged trajectories with nudges branched from free states "
            "against exact temporal and final-frame CE gradients"
        )
    )
    parser.add_argument("--extent", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--nudge-steps", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_branch_nudge_alignment.csv"),
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
                    nudge_steps=args.nudge_steps,
                    step_size=args.step_size,
                )
            )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = frame.groupby("comparison")[["cosine", "residual_fraction", "norm_ratio"]].agg(
        ["mean", "median", "std"]
    )
    print(
        f"Branched nudge alignment: extent={args.extent}, frame_steps={args.frame_steps}, "
        f"nudge_steps={args.nudge_steps}, seeds={args.seeds}"
    )
    print(summary.to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
