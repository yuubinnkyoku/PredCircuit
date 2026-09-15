"""Sanity check: sPC/ePC and PC-ALM do not share the same fixed-point target.

For the scalar one-hidden-layer linear model

    h_pred = a
    y_pred = b h

ordinary predictive coding minimizes

    E_PC(h) = 1/2 (h-a)^2 + 1/2 (b h-y)^2,

whereas PC-ALM enforces the feed-forward constraint h-a=0 and stores the
supervised credit in the dual variable.  This tiny exact case prevents us from
using "distance to the PC equilibrium" as a supposedly common stopping metric
for sPC/ePC/PC-ALM.
"""

from __future__ import annotations

import argparse
import json


def exact_pc(a: float, b: float, y: float) -> float:
    return (a + b * y) / (1.0 + b * b)


def exact_pcalm(a: float, b: float, y: float) -> tuple[float, float]:
    h = a
    # L = 1/2 (b h-y)^2 + lambda (h-a): dL/dh = 0 at the KKT point.
    lam = -b * (b * h - y)
    return h, lam


def run_spc(a: float, b: float, y: float, *, lr: float, steps: int) -> float:
    h = a
    for _ in range(steps):
        grad = (h - a) + b * (b * h - y)
        h -= lr * grad
    return h


def run_pcalm(
    a: float,
    b: float,
    y: float,
    *,
    lr: float,
    alpha: float,
    rho: float,
    steps: int,
) -> tuple[float, float]:
    h = a
    lam = 0.0
    for _ in range(steps):
        grad = b * (b * h - y) + lam + rho * (h - a)
        h -= lr * grad
        lam += alpha * (h - a)
    return h, lam


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=float, default=0.7)
    parser.add_argument("--b", type=float, default=1.2)
    parser.add_argument("--y", type=float, default=-0.4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    pc_star = exact_pc(args.a, args.b, args.y)
    pcalm_h_star, pcalm_lam_star = exact_pcalm(args.a, args.b, args.y)
    spc_h = run_spc(args.a, args.b, args.y, lr=args.lr, steps=args.steps)
    pcalm_h, pcalm_lam = run_pcalm(
        args.a,
        args.b,
        args.y,
        lr=args.lr,
        alpha=args.alpha,
        rho=args.rho,
        steps=args.steps,
    )

    result = {
        "exact_pc_h": pc_star,
        "exact_pcalm_h": pcalm_h_star,
        "exact_pcalm_lambda": pcalm_lam_star,
        "pc_vs_pcalm_target_gap": abs(pc_star - pcalm_h_star),
        "spc_h": spc_h,
        "spc_error_to_pc": abs(spc_h - pc_star),
        "pcalm_h": pcalm_h,
        "pcalm_lambda": pcalm_lam,
        "pcalm_error_to_kkt_h": abs(pcalm_h - pcalm_h_star),
        "pcalm_error_to_kkt_lambda": abs(pcalm_lam - pcalm_lam_star),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
