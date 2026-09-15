"""Measure how PC-ALM forms the exact BP/KKT credit in a deep linear chain.

The model is scalar at every layer:

    z_l = w_l z_{l-1},   l=1..L
    loss = 1/2 (w_L z_{L-1} - y)^2.

Hidden activations z_1..z_{L-1} are primal variables.  At the constrained
optimum they equal the ordinary forward pass.  The exact KKT multipliers obey

    lambda_{L-1} = -w_L (y_hat-y)
    lambda_l = w_{l+1} lambda_{l+1}.

Thus -lambda_l is exactly the BP adjoint d loss / d z_l.  This script tracks
how many local PC-ALM outer iterations are needed to form that credit.
"""

from __future__ import annotations

import argparse
import json
import math


def forward_hidden(x: float, weights: list[float]) -> tuple[list[float], float]:
    hidden: list[float] = []
    z = x
    for w in weights[:-1]:
        z = w * z
        hidden.append(z)
    return hidden, weights[-1] * z


def exact_duals(x: float, y: float, weights: list[float]) -> list[float]:
    hidden, y_hat = forward_hidden(x, weights)
    del hidden
    lam = [0.0] * (len(weights) - 1)
    lam[-1] = -weights[-1] * (y_hat - y)
    for layer in range(len(lam) - 2, -1, -1):
        lam[layer] = weights[layer + 1] * lam[layer + 1]
    return lam


def residuals(x: float, weights: list[float], hidden: list[float]) -> list[float]:
    out: list[float] = []
    prev = x
    for layer, z in enumerate(hidden):
        out.append(z - weights[layer] * prev)
        prev = z
    return out


def state_gradient(
    x: float,
    y: float,
    weights: list[float],
    hidden: list[float],
    duals: list[float],
    rho: float,
) -> list[float]:
    r = residuals(x, weights, hidden)
    shifted = [duals[i] + rho * r[i] for i in range(len(r))]
    grad = [0.0] * len(hidden)
    for layer in range(len(hidden)):
        grad[layer] += shifted[layer]
        if layer + 1 < len(hidden):
            grad[layer] -= weights[layer + 1] * shifted[layer + 1]
        else:
            y_hat = weights[-1] * hidden[-1]
            grad[layer] += weights[-1] * (y_hat - y)
    return grad


def norm(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


def run(
    *,
    depth: int,
    x: float,
    y: float,
    weight: float,
    state_lr: float,
    alpha: float,
    rho: float,
    steps: int,
    tolerance: float,
) -> dict[str, float | int | None]:
    if depth < 2:
        raise ValueError("depth must be at least 2")
    weights = [weight] * depth
    hidden, _ = forward_hidden(x, weights)
    duals = [0.0] * (depth - 1)
    target = exact_duals(x, y, weights)
    target_norm = max(norm(target), 1e-30)
    hit: int | None = None

    for step in range(1, steps + 1):
        grad = state_gradient(x, y, weights, hidden, duals, rho)
        hidden = [z - state_lr * g for z, g in zip(hidden, grad, strict=True)]
        r = residuals(x, weights, hidden)
        duals = [lam + alpha * ri for lam, ri in zip(duals, r, strict=True)]
        rel_dual_error = norm([a - b for a, b in zip(duals, target, strict=True)]) / target_norm
        if hit is None and rel_dual_error <= tolerance:
            hit = step

    r = residuals(x, weights, hidden)
    rel_dual_error = norm([a - b for a, b in zip(duals, target, strict=True)]) / target_norm
    return {
        "depth": depth,
        "steps_to_tolerance": hit,
        "final_relative_dual_error": rel_dual_error,
        "final_constraint_norm": norm(r),
        "target_dual_norm": norm(target),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--x", type=float, default=0.7)
    parser.add_argument("--y", type=float, default=-0.4)
    parser.add_argument("--weight", type=float, default=0.9)
    parser.add_argument("--state-lr", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    rows = [
        run(
            depth=depth,
            x=args.x,
            y=args.y,
            weight=args.weight,
            state_lr=args.state_lr,
            alpha=args.alpha,
            rho=args.rho,
            steps=args.steps,
            tolerance=args.tolerance,
        )
        for depth in args.depths
    ]
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
