from __future__ import annotations

import argparse
from pathlib import Path

from diagnose_flyvis_type_pair_selective_gradient_geometry import run_seed

DYNAMIC_HORIZONS = tuple(range(140, 160))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the full epoch-140..159 trajectory of L2-matched type-pair "
            "selective gradient geometry on the standard 4m+r path."
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = run_seed(
        seed=args.seed,
        init_scales=(0.115,),
        horizons=DYNAMIC_HORIZONS,
    )
    if len(frame) != len(DYNAMIC_HORIZONS) or not frame["finite"].all():
        raise RuntimeError("incomplete or non-finite dynamic geometry diagnostic")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
