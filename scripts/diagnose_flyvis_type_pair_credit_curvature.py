from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_branch_nudge_alignment import cycle_branched_credit
from diagnose_flyvis_strict_curvature import exact_cycle_loss
from diagnose_flyvis_temporal_ce_credit_alignment import cycle_ce_oracles, geometry
from run_flyvis_credit_residual_scaling import evaluate_metrics
from run_flyvis_temporal_coherence_gate import apply_local_credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import RetinotopicFlyVisCircuit, graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def safe_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) <= 1e-30:
        return float("nan")
    return float(torch.dot(left, right) / denominator)


def type_pair_indices(
    circuit: RetinotopicFlyVisCircuit,
) -> dict[tuple[str, str], torch.Tensor]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (source, target) in enumerate(circuit.graph.edge_index.t().tolist()):
        groups[(circuit.node_types[source], circuit.node_types[target])].append(index)
    return {pair: torch.tensor(indices, dtype=torch.long) for pair, indices in groups.items()}


def checkpoint_rows(
    model: PredictiveCodingGraph,
    circuit: RetinotopicFlyVisCircuit,
    *,
    seed: int,
    checkpoint: int,
    nudge_steps: int,
    test_repeats: int,
) -> list[dict[str, float | int | str | bool]]:
    local_edge, local_bias = cycle_branched_credit(
        model,
        circuit,
        seed=seed,
        epoch=0,
        frames=7,
        width=0.5,
        beta=0.03,
        frame_steps=2,
        nudge_steps=nudge_steps,
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
    full_geometry = geometry(local_edge, local_bias, oracle_edge, oracle_bias)

    oracle_norm_sq = torch.dot(oracle_edge, oracle_edge) + torch.dot(oracle_bias, oracle_bias)
    alpha = (
        torch.dot(local_edge, oracle_edge) + torch.dot(local_bias, oracle_bias)
    ) / oracle_norm_sq.clamp_min(1e-30)
    parallel_edge = alpha * oracle_edge
    parallel_bias = alpha * oracle_bias
    residual_edge = local_edge - parallel_edge
    residual_bias = local_bias - parallel_bias
    full_residual_norm_sq = torch.dot(residual_edge, residual_edge) + torch.dot(
        residual_bias, residual_bias
    )
    edge_residual_norm_sq = torch.dot(residual_edge, residual_edge)

    weight = model.weight.detach().clone().requires_grad_(True)
    bias = model.bias.detach().clone().requires_grad_(True)
    loss = exact_cycle_loss(circuit, weight, bias, seed=seed, epoch=0)
    grad_edge, grad_bias = torch.autograd.grad(loss, (weight, bias), create_graph=True)
    oracle_reconstruction_error = torch.linalg.vector_norm(
        torch.cat((grad_edge + oracle_edge, grad_bias + oracle_bias))
    ) / torch.linalg.vector_norm(torch.cat((oracle_edge, oracle_bias))).clamp_min(1e-30)

    residual_derivative = torch.dot(grad_edge, residual_edge) + torch.dot(grad_bias, residual_bias)
    hvp_edge, hvp_bias = torch.autograd.grad(
        residual_derivative,
        (weight, bias),
    )
    total_residual_curvature = torch.dot(residual_edge, hvp_edge) + torch.dot(
        residual_bias, hvp_bias
    )
    total_cross_curvature = 2.0 * (
        torch.dot(parallel_edge, hvp_edge) + torch.dot(parallel_bias, hvp_bias)
    )

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
    rows: list[dict[str, float | int | str | bool]] = []
    for (source_type, target_type), indices in type_pair_indices(circuit).items():
        local = local_edge[indices]
        oracle = oracle_edge[indices]
        parallel = parallel_edge[indices]
        residual = residual_edge[indices]
        residual_hvp = hvp_edge[indices]
        centered_local = local - local.mean()
        centered_oracle = oracle - oracle.mean()
        centered_residual = residual - residual.mean()
        residual_norm_sq = torch.dot(residual, residual)
        group_dot = torch.dot(local, oracle)
        group_oracle_norm_sq = torch.dot(oracle, oracle)
        rows.append(
            {
                "component": "edge_type_pair",
                "seed": seed,
                "epoch": checkpoint,
                "nudge_steps": nudge_steps,
                "source_type": source_type,
                "target_type": target_type,
                "edge_count": int(indices.numel()),
                "cross_entropy": task["cross_entropy"],
                "accuracy": task["accuracy"],
                "margin": task["margin"],
                "full_cosine": full_geometry["cosine"],
                "global_alpha": float(alpha),
                "local_norm": float(torch.linalg.vector_norm(local)),
                "oracle_norm": float(torch.linalg.vector_norm(oracle)),
                "pair_cosine": safe_cosine(local, oracle),
                "pair_projection_coefficient": float(
                    group_dot / group_oracle_norm_sq.clamp_min(1e-30)
                ),
                "pair_alignment_dot": float(group_dot),
                "centered_pair_cosine": safe_cosine(centered_local, centered_oracle),
                "centered_local_norm": float(torch.linalg.vector_norm(centered_local)),
                "centered_oracle_norm": float(torch.linalg.vector_norm(centered_oracle)),
                "residual_norm": float(torch.sqrt(residual_norm_sq)),
                "residual_edge_fraction_sq": float(
                    residual_norm_sq / edge_residual_norm_sq.clamp_min(1e-30)
                ),
                "residual_full_fraction_sq": float(
                    residual_norm_sq / full_residual_norm_sq.clamp_min(1e-30)
                ),
                "residual_mean": float(residual.mean()),
                "centered_residual_norm": float(torch.linalg.vector_norm(centered_residual)),
                "centered_residual_fraction": float(
                    torch.linalg.vector_norm(centered_residual)
                    / torch.linalg.vector_norm(residual).clamp_min(1e-30)
                ),
                "residual_curvature_contribution": float(torch.dot(residual, residual_hvp)),
                "cross_curvature_contribution": float(2.0 * torch.dot(parallel, residual_hvp)),
                "total_residual_curvature": float(total_residual_curvature),
                "total_cross_curvature": float(total_cross_curvature),
                "oracle_reconstruction_relative_error": float(oracle_reconstruction_error),
                "finite": bool(torch.isfinite(residual_hvp).all()),
            }
        )

    bias_residual_curvature = torch.dot(residual_bias, hvp_bias)
    bias_cross_curvature = 2.0 * torch.dot(parallel_bias, hvp_bias)
    rows.append(
        {
            "component": "bias",
            "seed": seed,
            "epoch": checkpoint,
            "nudge_steps": nudge_steps,
            "source_type": "__bias__",
            "target_type": "__bias__",
            "edge_count": int(local_bias.numel()),
            "cross_entropy": task["cross_entropy"],
            "accuracy": task["accuracy"],
            "margin": task["margin"],
            "full_cosine": full_geometry["cosine"],
            "global_alpha": float(alpha),
            "local_norm": float(torch.linalg.vector_norm(local_bias)),
            "oracle_norm": float(torch.linalg.vector_norm(oracle_bias)),
            "pair_cosine": safe_cosine(local_bias, oracle_bias),
            "pair_projection_coefficient": float(
                torch.dot(local_bias, oracle_bias)
                / torch.dot(oracle_bias, oracle_bias).clamp_min(1e-30)
            ),
            "pair_alignment_dot": float(torch.dot(local_bias, oracle_bias)),
            "centered_pair_cosine": float("nan"),
            "centered_local_norm": float("nan"),
            "centered_oracle_norm": float("nan"),
            "residual_norm": float(torch.linalg.vector_norm(residual_bias)),
            "residual_edge_fraction_sq": float("nan"),
            "residual_full_fraction_sq": float(
                torch.dot(residual_bias, residual_bias) / full_residual_norm_sq.clamp_min(1e-30)
            ),
            "residual_mean": float(residual_bias.mean()),
            "centered_residual_norm": float("nan"),
            "centered_residual_fraction": float("nan"),
            "residual_curvature_contribution": float(bias_residual_curvature),
            "cross_curvature_contribution": float(bias_cross_curvature),
            "total_residual_curvature": float(total_residual_curvature),
            "total_cross_curvature": float(total_cross_curvature),
            "oracle_reconstruction_relative_error": float(oracle_reconstruction_error),
            "finite": bool(torch.isfinite(hvp_bias).all()),
        }
    )
    return rows


def run_seed(
    *,
    seed: int,
    checkpoints: list[int],
    train_nudge_steps: int,
    eval_nudge_steps: int,
    learning_rate: float,
    max_update: float,
    test_repeats: int,
) -> list[dict[str, float | int | str | bool]]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    rows: list[dict[str, float | int | str | bool]] = []
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
        rows.extend(
            checkpoint_rows(
                model,
                circuit,
                seed=seed,
                checkpoint=checkpoint,
                nudge_steps=eval_nudge_steps,
                test_repeats=test_repeats,
            )
        )
        previous_epoch = checkpoint
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decompose FlyVis local-credit residual and Hessian coupling by cell-type pair"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-nudge-steps", type=int, default=2)
    parser.add_argument("--eval-nudge-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_type_pair_credit_curvature.csv"),
    )
    args = parser.parse_args()

    frame = pd.DataFrame(
        run_seed(
            seed=args.seed,
            checkpoints=[0, 25, 50, 100],
            train_nudge_steps=args.train_nudge_steps,
            eval_nudge_steps=args.eval_nudge_steps,
            learning_rate=args.learning_rate,
            max_update=args.max_update,
            test_repeats=args.test_repeats,
        )
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    edges = frame[frame["component"] == "edge_type_pair"].copy()
    print(
        edges.sort_values(
            ["epoch", "residual_curvature_contribution"],
            ascending=[True, False],
        )[
            [
                "epoch",
                "source_type",
                "target_type",
                "edge_count",
                "pair_cosine",
                "centered_pair_cosine",
                "residual_full_fraction_sq",
                "residual_curvature_contribution",
                "cross_curvature_contribution",
            ]
        ]
        .groupby("epoch")
        .head(12)
        .to_string(index=False)
    )
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
