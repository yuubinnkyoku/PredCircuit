from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.locality_cost import estimate_two_phase_pc_vs_bptt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate temporary training-state storage for local PC and BPTT"
    )
    parser.add_argument("--extents", default="1,2,3")
    parser.add_argument("--frames", default="5,7,9")
    parser.add_argument("--frame-steps", default="1,2,3,4")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bits", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_training_locality.csv"),
    )
    args = parser.parse_args()

    extents = [int(value) for value in args.extents.split(",")]
    frames_values = [int(value) for value in args.frames.split(",")]
    frame_steps_values = [int(value) for value in args.frame_steps.split(",")]
    bytes_per_value = args.bits // 8
    spec = load_flyvis_spec()

    rows: list[dict[str, float | int]] = []
    for extent in extents:
        circuit = graph_from_flyvis_retinotopy(spec, extent=extent)
        for frames in frames_values:
            for frame_steps in frame_steps_values:
                recurrent_steps = frames * frame_steps
                cost = estimate_two_phase_pc_vs_bptt(
                    nodes=circuit.graph.num_nodes,
                    edges=circuit.graph.num_edges,
                    batch_size=args.batch_size,
                    recurrent_steps=recurrent_steps,
                    bytes_per_value=bytes_per_value,
                )
                rows.append(
                    {
                        "extent": extent,
                        "nodes": circuit.graph.num_nodes,
                        "edges": circuit.graph.num_edges,
                        "frames": frames,
                        "frame_steps": frame_steps,
                        "recurrent_steps": recurrent_steps,
                        "bits": args.bits,
                        "local_bytes": cost.local_bytes,
                        "bptt_state_lower_bound_bytes": cost.bptt_state_lower_bound_bytes,
                        "bptt_over_local_ratio": cost.bptt_over_local_ratio,
                        "crossover_recurrent_steps": cost.crossover_recurrent_steps,
                    }
                )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(
        frame[
            [
                "extent",
                "nodes",
                "edges",
                "recurrent_steps",
                "local_bytes",
                "bptt_state_lower_bound_bytes",
                "bptt_over_local_ratio",
                "crossover_recurrent_steps",
            ]
        ].to_string(index=False)
    )
    print(
        "\nBPTT numbers are state-history lower bounds; real autodiff retains additional intermediates."
    )
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
