from __future__ import annotations

import argparse
import copy
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_cross_curvature import _heldout_ce
from diagnose_flyvis_type_pair_horizon_causal import _match_local_state, _mixed_direction
from diagnose_flyvis_type_pair_state_geometry import _group_constant_projection
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit
from run_flyvis_type_pair_shuffle_control import shuffled_groups_like

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import RetinotopicFlyVisCircuit, graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

HORIZONS = (60, 80, 100)
EVAL_REPS = 2
EVAL_JITTER_BASE = 2_700_000


def _named_type_pair_groups(
    circuit: RetinotopicFlyVisCircuit,
) -> list[tuple[str, str, torch.Tensor]]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (source, target) in enumerate(circuit.graph.edge_index.t().tolist()):
        groups[(circuit.node_types[source], circuit.node_types[target])].append(index)
    return [
        (source_type, target_type, torch.tensor(indices, dtype=torch.long))
        for (source_type, target_type), indices in groups.items()
    ]


def _group_projection(vector: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(vector)
    result[indices] = vector[indices].mean()
    return result


def run_seed(
    *,
    seed: int,
    shuffle_seed: int,
    learning_rate: float,
    max_update: float,
) -> pd.DataFrame:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    base = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    local = copy.deepcopy(base)
    type_ = copy.deepcopy(base)
    named_groups = _named_type_pair_groups(circuit)
    biological_groups = [indices for _, _, indices in named_groups]
    shuffled_groups = shuffled_groups_like(
        biological_groups,
        edge_count=base.weight.numel(),
        shuffle_seed=shuffle_seed,
    )
    rows: list[dict[str, float | int | bool | str]] = []

    for step in range(1, max(HORIZONS) + 1):
        epoch = step - 1
        local_edge, local_bias = credit(local, circuit, seed=seed, epoch=epoch)
        apply_local_credit(
            local,
            local_edge,
            local_bias,
            learning_rate=learning_rate,
            weight_decay=0.0,
            max_update=max_update,
        )

        raw_edge, raw_bias = credit(type_, circuit, seed=seed, epoch=epoch)
        direction = _mixed_direction(raw_edge, biological_groups)
        apply_local_credit(
            type_,
            direction,
            raw_bias,
            learning_rate=learning_rate,
            weight_decay=0.0,
            max_update=max_update,
        )
        _match_local_state(type_, local)

        if step not in HORIZONS:
            continue

        local_weight = local.weight.detach()
        delta = type_.weight.detach() - local_weight
        bias = local.bias.detach().clone()

        for eval_rep in range(EVAL_REPS):
            jitter_seed = EVAL_JITTER_BASE + eval_rep
            for grouping in ("bio", "shuffle"):
                if grouping == "bio":
                    groups = biological_groups
                else:
                    groups = shuffled_groups
                projection = _group_constant_projection(delta, groups).detach()
                residual = (delta - projection).detach()

                weight = local_weight.clone().requires_grad_(True)
                loss = _heldout_ce(circuit, weight, bias, jitter_seed=jitter_seed)
                (gradient,) = torch.autograd.grad(loss, weight, create_graph=True)
                grad_residual = torch.dot(gradient, residual)
                residual_hvp = torch.autograd.grad(grad_residual, weight)[0].detach()
                total_synergy = float(-torch.dot(projection, residual_hvp))

                contribution_sum = 0.0
                for group_id, indices in enumerate(groups):
                    group_projection = _group_projection(delta, indices)
                    contribution = float(-torch.dot(group_projection, residual_hvp))
                    contribution_sum += contribution
                    if grouping == "bio":
                        source_type, target_type, _ = named_groups[group_id]
                    else:
                        source_type = "shuffle"
                        target_type = "shuffle"
                    values = {
                        "total_predicted_synergy": total_synergy,
                        "group_synergy_contribution": contribution,
                        "group_projection_norm": float(torch.linalg.vector_norm(group_projection)),
                        "group_residual_hvp_norm": float(
                            torch.linalg.vector_norm(residual_hvp[indices])
                        ),
                    }
                    finite = (
                        bool(torch.isfinite(local.weight).all())
                        and bool(torch.isfinite(type_.weight).all())
                        and all(math.isfinite(value) for value in values.values())
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "horizon": step,
                            "eval_rep": eval_rep,
                            "grouping": grouping,
                            "group_id": group_id,
                            "source_type": source_type,
                            "target_type": target_type,
                            "edge_count": int(indices.numel()),
                            **values,
                            "finite": finite,
                        }
                    )

                if not math.isclose(
                    contribution_sum,
                    total_synergy,
                    rel_tol=2e-5,
                    abs_tol=2e-8,
                ):
                    raise RuntimeError(
                        f"group contributions do not sum to total: "
                        f"{contribution_sum} vs {total_synergy}"
                    )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Attribute the held-out CE projection-residual Hessian synergy to individual "
            "biological type-pair groups and size-matched shuffled groups"
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shuffle-seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frame = run_seed(
        seed=args.seed,
        shuffle_seed=args.shuffle_seed,
        learning_rate=args.learning_rate,
        max_update=args.max_update,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
