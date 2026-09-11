from __future__ import annotations

import argparse
from pathlib import Path

import diagnose_flyvis_type_pair_axis_postburst_donor_transfer as donor_transfer
import pandas as pd


HORIZONS = (0, 1, 2, 5, 10, 25)
CHECKPOINTS = tuple(donor_transfer.BURST_END + horizon for horizon in HORIZONS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how the post-burst type-state/type-credit compatibility interaction "
            "accumulates over 0, 1, 2, 5, 10, and 25 local-learning steps"
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shuffle-seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    donor_transfer.CHECKPOINTS = CHECKPOINTS
    frame = pd.DataFrame(
        donor_transfer.run_seed(
            seed=args.seed,
            shuffle_seed=args.shuffle_seed,
            learning_rate=args.learning_rate,
            max_update=args.max_update,
        )
    )
    frame["horizon"] = frame["epoch"] - donor_transfer.BURST_END
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
