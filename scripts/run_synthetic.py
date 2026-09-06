from __future__ import annotations

import argparse
from pathlib import Path

from predcircuit.experiment import (
    results_frame,
    train_bptt_linear_mapping,
    train_linear_mapping,
)
from predcircuit.topology import erdos_renyi_matched, layered_graph


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run topology-controlled predictive-coding sanity checks"
    )
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--pc-only", action="store_true", help="Skip the same-topology BPTT control")
    p.add_argument("--out", type=Path, default=Path("results/generated/synthetic_mapping.csv"))
    args = p.parse_args()

    results = []
    for seed in range(args.seeds):
        base = layered_graph(
            [2, 8, 1], recurrent_probability=0.2, feedback_probability=0.08, seed=seed
        )
        variants = {
            "base": base,
            "degree_preserving_rewire": base.degree_preserving_rewire(
                max(base.num_edges * 3, 1), seed=10_000 + seed
            ),
            "er_matched": erdos_renyi_matched(base, seed=20_000 + seed),
            "no_feedback": base.remove_feedback_edges(),
            "no_reciprocal": base.remove_reciprocal_edges(),
        }
        for name, graph in variants.items():
            results.append(
                train_linear_mapping(graph, topology_name=name, seed=seed, epochs=args.epochs)
            )
            if not args.pc_only:
                results.append(
                    train_bptt_linear_mapping(
                        graph, topology_name=name, seed=seed, epochs=args.epochs
                    )
                )

    df = results_frame(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    summary = (
        df.groupby(["learning_rule", "topology"])["mse_after"]
        .agg(["mean", "std", "min", "max"])
        .sort_index()
    )
    print(df.to_string(index=False))
    print("\nPost-training MSE summary:\n", summary)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
