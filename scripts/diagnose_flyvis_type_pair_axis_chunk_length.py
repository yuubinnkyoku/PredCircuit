from __future__ import annotations

import argparse
from pathlib import Path

import diagnose_flyvis_type_pair_axis_burst_schedule as burst_schedule
import pandas as pd


def _symmetric_mask(first_half: set[int]) -> set[int]:
    return first_half | {99 - epoch for epoch in first_half}


MASKS = {
    "chunk1": _symmetric_mask(set(range(0, 50, 2))),
    "chunk5": _symmetric_mask(
        {epoch for start in (0, 10, 20, 30, 40) for epoch in range(start, start + 5)}
    ),
    "chunk25": _symmetric_mask(set(range(13, 38))),
    "chunk50": set(range(25, 75)),
}

for name, mask in MASKS.items():
    if len(mask) != 50:
        raise RuntimeError(f"{name} expected 50 pulses, got {len(mask)}")
    if sum(mask) / len(mask) != 49.5:
        raise RuntimeError(f"{name} is not centered at epoch 49.5")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Hold pulse count and temporal center fixed while varying contiguous type-pair "
            "credit chunk length"
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shuffle-seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    burst_schedule.SCHEDULES = {
        name: (lambda epoch, mask=mask: epoch in mask) for name, mask in MASKS.items()
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
