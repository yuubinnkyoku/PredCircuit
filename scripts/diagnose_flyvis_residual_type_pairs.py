from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_branch_nudge_alignment import cycle_branched_credit
from diagnose_flyvis_temporal_ce_credit_alignment import cycle_ce_oracles
from run_flyvis_credit_residual_scaling import evaluate_metrics
from run_flyvis_retinotopic_temporal_adam import local_adam_step

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import RetinotopicFlyVisCircuit, graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float(torch.dot(left, right) / denominator.clamp_min(1e-30))


def weighted_sign_agreement(candidate: torch.Tensor, oracle: torch.Tensor) -> float:
    weights = oracle.abs()
    same = (candidate * oracle > 0).to(weights.dtype)
    return float((weights * same).sum() / weights.sum().clamp_min(1e-30))


def edge_blocks(circuit: RetinotopicFlyVisCircuit) -> dict[tuple[str, str], torch.Tensor]:
    blocks: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (source, target) in enumerate(circuit.graph.edge_index.t().tolist()):
        blocks[(circuit.node_types[source], circuit.node_types[target])].append(index)
    return {key: torch.tensor(indices, dtype=torch.long) for key, indices in blocks.items()}


def bias_blocks(circuit: RetinotopicFlyVisCircuit) -> dict[str, torch.Tensor]:
    blocks: dict[str, list[int]] = defaultdict(list)
    for index, cell_type in enumerate(circuit.node_types):
        blocks[cell_type].append(index)
    return {key: torch.tensor(indices, dtype=torch.long) for key, indices in blocks.items()}


def block_metrics(
    local: torch.Tensor,
    oracle: torch.Tensor,
    residual: torch.Tensor,
    *,
    alpha: torch.Tensor,
    total_residual_energy: torch.Tensor,
    total_oracle_energy: torch.Tensor,
) -> dict[str, float]:
    oracle_norm_sq = torch.dot(oracle, oracle)
    projection = torch.dot(local, oracle) / oracle_norm_sq.clamp_min(1e-30)
    parallel = alpha * oracle
    return {
        "local_norm": float(torch.linalg.vector_norm(local)),
        "oracle_norm": float(torch.linalg.vector_norm(oracle)),
        "residual_norm": float(torch.linalg.vector_norm(residual)),
        "residual_energy_fraction": float(
            torch.dot(residual, residual) / total_residual_energy.clamp_min(1e-30)
        ),
        "oracle_energy_fraction": float(
            oracle_norm_sq / total_oracle_energy.clamp_min(1e-30)
        ),
        "local_oracle_cosine": cosine(local, oracle),
        "projection_coefficient": float(projection),
        "residual_to_parallel_norm": float(
            torch.linalg.vector_norm(residual)
            / torch.linalg.vector_norm(parallel).clamp_min(1e-30)
        ),
        "oracle_weighted_sign_agreement": weighted_sign_agreement(local, oracle),
    }


def measure_blocks(
    circuit: RetinotopicFlyVisCircuit,
    local_edge: torch.Tensor,
    local_bias: torch.Tensor,
    oracle_edge: torch.Tensor,
    oracle_bias: torch.Tensor,
    *,
    seed: int,
    epoch: int,
    nudge_steps: int,
    cross_entropy: float,
    accuracy: float,
) -> list[dict[str, float | int | str | bool]]:
    local = torch.cat((local_edge.reshape(-1), local_bias.reshape(-1)))
    oracle = torch.cat((oracle_edge.reshape(-1), oracle_bias.reshape(-1)))
    alpha = torch.dot(local, oracle) / torch.dot(oracle, oracle).clamp_min(1e-30)
    residual_edge = local_edge - alpha * oracle_edge
    residual_bias = local_bias - alpha * oracle_bias
    total_residual_energy = torch.dot(residual_edge, residual_edge) + torch.dot(
        residual_bias, residual_bias
    )
    total_oracle_energy = torch.dot(oracle_edge, oracle_edge) + torch.dot(oracle_bias, oracle_bias)
    global_cosine = cosine(local, oracle)

    input_types = {circuit.node_types[index] for index in circuit.input_nodes}
    output_types = {circuit.node_types[index] for index in circuit.output_nodes}
    pair_keys = set(edge_blocks(circuit))
    rows: list[dict[str, float | int | str | bool]] = []

    for (source_type, target_type), indices in edge_blocks(circuit).items():
        metrics = block_metrics(
            local_edge[indices],
            oracle_edge[indices],
            residual_edge[indices],
            alpha=alpha,
            total_residual_energy=total_residual_energy,
            total_oracle_energy=total_oracle_energy,
        )
        rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "nudge_steps": nudge_steps,
                "cross_entropy": cross_entropy,
                "accuracy": accuracy,
                "global_cosine": global_cosine,
                "global_projection_coefficient": float(alpha),
                "kind": "edge",
                "source_type": source_type,
                "target_type": target_type,
                "cell_type": "",
                "count": int(indices.numel()),
                "reciprocal_type_pair": (target_type, source_type) in pair_keys,
                "source_is_input": source_type in input_types,
                "target_is_output": target_type in output_types,
                **metrics,
            }
        )

    for cell_type, indices in bias_blocks(circuit).items():
        metrics = block_metrics(
            local_bias[indices],
            oracle_bias[indices],
            residual_bias[indices],
            alpha=alpha,
            total_residual_energy=total_residual_energy,
            total_oracle_energy=total_oracle_energy,
        )
        rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "nudge_steps": nudge_steps,
                "cross_entropy": cross_entropy,
                "accuracy": accuracy,
                "global_cosine": global_cosine,
                "global_projection_coefficient": float(alpha),
                "kind": "bias",
                "source_type": "",
                "target_type": "",
                "cell_type": cell_type,
                "count": int(indices.numel()),
                "reciprocal_type_pair": False,
                "source_is_input": cell_type in input_types,
                "target_is_output": cell_type in output_types,
                **metrics,
            }
        )
    return rows


def run_seed(
    *,
    seed: int,
    epochs: int,
    checkpoint_every: int,
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
) -> list[dict[str, float | int | str | bool]]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    edge_m = torch.zeros_like(model.weight)
    edge_v = torch.zeros_like(model.weight)
    bias_m = torch.zeros_like(model.bias)
    bias_v = torch.zeros_like(model.bias)
    rows: list[dict[str, float | int | str | bool]] = []

    for epoch in range(epochs):
        local_edge, local_bias = cycle_branched_credit(
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
        if epoch % checkpoint_every == 0 or epoch == epochs - 1:
            oracle_edge, oracle_bias = cycle_ce_oracles(
                model,
                circuit,
                seed=seed,
                epoch=epoch,
                frames=frames,
                width=width,
                frame_steps=frame_steps,
                step_size=step_size,
            )["final_ce_oracle"]
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
            rows.extend(
                measure_blocks(
                    circuit,
                    local_edge,
                    local_bias,
                    oracle_edge,
                    oracle_bias,
                    seed=seed,
                    epoch=epoch,
                    nudge_steps=nudge_steps,
                    cross_entropy=task["cross_entropy"],
                    accuracy=task["accuracy"],
                )
            )

        local_adam_step(
            model.weight,
            local_edge,
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
            local_bias,
            bias_m,
            bias_v,
            step=epoch + 1,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=adam_epsilon,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decompose branched local-credit residuals by FlyVis cell-type pair"
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--nudge-steps", type=int, required=True)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--test-repeats", type=int, default=4)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_residual_type_pairs.csv"),
    )
    args = parser.parse_args()

    rows: list[dict[str, float | int | str | bool]] = []
    for seed in range(args.seeds):
        rows.extend(
            run_seed(
                seed=seed,
                epochs=args.epochs,
                checkpoint_every=args.checkpoint_every,
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
    edges = frame[frame["kind"] == "edge"]
    summary = (
        edges.groupby(["source_type", "target_type"])["residual_energy_fraction"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
    )
    print(f"Residual type-pair diagnostic: nudge_steps={args.nudge_steps}, seeds={args.seeds}")
    print(summary.to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
