from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import diagnose_flyvis_type_pair_axis_pulse_frequency as pulse_frequency
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add only the missing every-2-epochs condition to the existing burst-schedule seeds"
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shuffle-seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cast(Any, pulse_frequency).PULSE_INTERVALS = (2,)
    frame = pd.DataFrame(
        pulse_frequency.run_seed(
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
