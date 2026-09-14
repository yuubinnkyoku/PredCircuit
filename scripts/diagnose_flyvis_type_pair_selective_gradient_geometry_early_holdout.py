from __future__ import annotations

import argparse
from pathlib import Path

from diagnose_flyvis_type_pair_selective_gradient_geometry import run_seed

EARLY_HORIZONS = (120, 140)
INIT_SCALES = (0.115,)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fresh early-horizon holdout of type-pair selective gradient geometry at "
            "epochs 120 and 140 for biological-strength init scale 0.115."
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frame = run_seed(
        seed=args.seed,
        learning_rate=args.learning_rate,
        max_update=args.max_update,
        init_scales=INIT_SCALES,
        horizons=EARLY_HORIZONS,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
