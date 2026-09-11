from __future__ import annotations

import argparse
from pathlib import Path

import diagnose_flyvis_type_pair_axis_burst_schedule as burst_schedule
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Match type-pair pulse count and temporal center while contrasting contiguous "
            "and evenly spaced schedules"
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shuffle-seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    burst_schedule.SCHEDULES = {
        "centered20": lambda epoch: 40 <= epoch < 60,
        "spread20": lambda epoch: epoch % 5 == 0,
        "centered50": lambda epoch: 25 <= epoch < 75,
        "spread50": lambda epoch: epoch % 2 == 0,
        "all100": lambda epoch: True,
    }
    frame = pd.DataFrame(
        burst_schedule.run_seed(
            seed=args.seed,
            shuffle_seed=args.shuffle_seed,
            learning_rate=args.learning_rate,
            max_update=args.max_update,
        )
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
