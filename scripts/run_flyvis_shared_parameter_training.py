from __future__ import annotations

import argparse
import math
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


class SharedFlyVisParameters:
    """FlyVis-like type-shared gains over fixed retinotopic connectome coefficients."""

    def __init__(self, circuit: RetinotopicFlyVisCircuit, *, gain_init: float = 0.08) -> None:
        graph = circuit.graph
        if graph.edge_weight is None:
            raise ValueError("FlyVis shared parameterization requires biological edge weights")

        pair_to_group: dict[tuple[str, str], int] = {}
        edge_groups: list[int] = []
        for source, target in graph.edge_index.t().tolist():
            pair = (circuit.node_types[source], circuit.node_types[target])
            if pair not in pair_to_group:
                pair_to_group[pair] = len(pair_to_group)
            edge_groups.append(pair_to_group[pair])
        self.edge_group = torch.tensor(edge_groups, dtype=torch.long)
        self.edge_group_count = len(pair_to_group)

        base = graph.edge_weight.float()
        mean_abs = torch.zeros(self.edge_group_count, dtype=base.dtype)
        counts = torch.zeros(self.edge_group_count, dtype=base.dtype)
        mean_abs.index_add_(0, self.edge_group, base.abs())
        counts.index_add_(0, self.edge_group, torch.ones_like(base))
        mean_abs = mean_abs / counts.clamp_min(1.0)
        self.edge_basis = base / mean_abs[self.edge_group].clamp_min(1e-8)
        self.gain = torch.full((self.edge_group_count,), gain_init, dtype=torch.float32)

        type_to_group: dict[str, int] = {}
        node_groups: list[int] = []
        for cell_type in circuit.node_types:
            if cell_type not in type_to_group:
                type_to_group[cell_type] = len(type_to_group)
            node_groups.append(type_to_group[cell_type])
        self.node_group = torch.tensor(node_groups, dtype=torch.long)
        self.node_group_count = len(type_to_group)
        self.bias = torch.zeros(self.node_group_count, dtype=torch.float32)

    def write_to_model(self, model: PredictiveCodingGraph) -> None:
        with torch.no_grad():
            model.weight.copy_(self.edge_basis * self.gain[self.edge_group])
            model.bias.copy_(self.bias[self.node_group])

    def project_direction(
        self,
        edge_direction: torch.Tensor,
        bias_direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gain_direction = torch.zeros_like(self.gain)
        gain_direction.index_add_(
            0,
            self.edge_group,
            self.edge_basis * edge_direction,
        )
        shared_bias_direction = torch.zeros_like(self.bias)
        shared_bias_direction.index_add_(0, self.node_group, bias_direction)
        return gain_direction, shared_bias_direction


def vector_geometry(candidate: torch.Tensor, oracle: torch.Tensor) -> tuple[float, float, float]:
    candidate_norm = torch.linalg.vector_norm(candidate).clamp_min(1e-30)
    oracle_norm = torch.linalg.vector_norm(oracle).clamp_min(1e-30)
    cosine = float(torch.dot(candidate, oracle) / (candidate_norm * oracle_norm))
    projection = float(torch.dot(candidate, oracle) / torch.dot(oracle, oracle).clamp_min(1e-30))
    norm_ratio = float(candidate_norm / oracle_norm)
    return cosine, projection, norm_ratio


def run_one(
    *,
    seed: int,
    mode: str,
    epochs: int,
    nudge_steps: int,
    learning_rate: float,
    test_repeats: int,
) -> dict[str, float | int | str | bool]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    shared = SharedFlyVisParameters(circuit, gain_init=0.08)
    shared.write_to_model(model)

    gain_m = torch.zeros_like(shared.gain)
    gain_v = torch.zeros_like(shared.gain)
    bias_m = torch.zeros_like(shared.bias)
    bias_v = torch.zeros_like(shared.bias)

    before = evaluate_metrics(
        model,
        circuit,
        repeats=test_repeats,
        frames=7,
        width=0.5,
        frame_steps=2,
        step_size=0.015,
        jitter_seed=900_000 + seed,
    )
    cosine_sum = 0.0
    projection_sum = 0.0
    norm_ratio_sum = 0.0

    for epoch in range(epochs):
        oracle_edge, oracle_bias = cycle_ce_oracles(
            model,
            circuit,
            seed=seed,
            epoch=epoch,
            frames=7,
            width=0.5,
            frame_steps=2,
            step_size=0.015,
        )["final_ce_oracle"]
        oracle_gain, oracle_shared_bias = shared.project_direction(oracle_edge, oracle_bias)

        if mode == "local":
            edge_direction, bias_direction = cycle_branched_credit(
                model,
                circuit,
                seed=seed,
                epoch=epoch,
                frames=7,
                width=0.5,
                beta=0.03,
                frame_steps=2,
                nudge_steps=nudge_steps,
                step_size=0.015,
                terminal_only=False,
            )
            gain_direction, shared_bias_direction = shared.project_direction(
                edge_direction, bias_direction
            )
        elif mode == "exact":
            gain_direction = oracle_gain
            shared_bias_direction = oracle_shared_bias
        else:
            raise ValueError(f"unsupported mode: {mode}")

        candidate = torch.cat((gain_direction, shared_bias_direction))
        oracle = torch.cat((oracle_gain, oracle_shared_bias))
        cosine, projection, norm_ratio = vector_geometry(candidate, oracle)
        cosine_sum += cosine
        projection_sum += projection
        norm_ratio_sum += norm_ratio

        local_adam_step(
            shared.gain,
            gain_direction,
            gain_m,
            gain_v,
            step=epoch + 1,
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
        )
        shared.gain.clamp_(min=0.0)
        local_adam_step(
            shared.bias,
            shared_bias_direction,
            bias_m,
            bias_v,
            step=epoch + 1,
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
        )
        shared.write_to_model(model)

    after = evaluate_metrics(
        model,
        circuit,
        repeats=test_repeats,
        frames=7,
        width=0.5,
        frame_steps=2,
        step_size=0.015,
        jitter_seed=900_000 + seed,
    )
    count = max(epochs, 1)
    return {
        "mode": mode,
        "seed": seed,
        "epochs": epochs,
        "nudge_steps": nudge_steps,
        "learning_rate": learning_rate,
        "edge_group_count": shared.edge_group_count,
        "node_group_count": shared.node_group_count,
        "accuracy_before": before["accuracy"],
        "accuracy_after": after["accuracy"],
        "cross_entropy_before": before["cross_entropy"],
        "cross_entropy_after": after["cross_entropy"],
        "cross_entropy_improvement": before["cross_entropy"] - after["cross_entropy"],
        "margin_after": after["margin"],
        "mean_shared_cosine": cosine_sum / count,
        "mean_shared_projection_coefficient": projection_sum / count,
        "mean_shared_norm_ratio": norm_ratio_sum / count,
        "gain_min": float(shared.gain.min()),
        "gain_max": float(shared.gain.max()),
        "gain_mean": float(shared.gain.mean()),
        "finite": bool(torch.isfinite(shared.gain).all())
        and bool(torch.isfinite(shared.bias).all())
        and math.isfinite(after["cross_entropy"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train predictive coding with FlyVis-like type-shared synaptic gains and biases"
    )
    parser.add_argument("--mode", choices=("local", "exact"), required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--nudge-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_shared_parameter_training.csv"),
    )
    args = parser.parse_args()

    rows = [
        run_one(
            seed=seed,
            mode=args.mode,
            epochs=args.epochs,
            nudge_steps=args.nudge_steps,
            learning_rate=args.learning_rate,
            test_repeats=args.test_repeats,
        )
        for seed in range(args.seeds)
    ]
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.select_dtypes(include="number").agg(["mean", "median", "std"]).to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
