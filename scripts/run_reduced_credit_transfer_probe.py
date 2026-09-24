from __future__ import annotations

import argparse
import json

import torch

from predcircuit.pcalm import ResidualMLP


def low_rank(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    r = min(rank, s.numel())
    return (u[:, :r] * s[:r]) @ vh[:r]


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm() * b.norm()
    if float(denom) == 0.0:
        return float("nan")
    return float(torch.dot(a, b) / denom)


def local_transfer(
    model: ResidualMLP,
    layer: int,
    z_prev: torch.Tensor,
    z: torch.Tensor,
    *,
    state_lr: float,
    rho: float,
    alpha: float,
) -> tuple[torch.Tensor, float]:
    """Eliminate one layer's local (z, lambda) state from a linearized PC-ALM step.

    Input c is an abstract downstream credit coupled linearly to z.  The output is
    the upstream credit induced through the local prediction constraint.  We form
    the exact one-step Jacobian at the free trajectory, partition it as
        s' = A s + B c,  c_up = C s + D c,
    and return D + C (I-A)^-1 B.  solve() is used rather than an explicit inverse.
    """
    n = z.numel()
    base = torch.cat(
        [z.reshape(-1), torch.zeros_like(z).reshape(-1), torch.zeros(n, dtype=z.dtype)]
    )

    def step(q: torch.Tensor) -> torch.Tensor:
        z_q = q[:n].reshape_as(z)
        lam_q = q[n : 2 * n].reshape_as(z)
        credit = q[2 * n :]
        pred = model.block_pred(layer, z_prev)
        residual = z_q - pred
        energy = 0.5 * rho * (residual + lam_q / rho).square().sum() + torch.dot(
            credit, z_q.reshape(-1)
        )
        grad_z = torch.autograd.grad(energy, z_q, create_graph=True)[0]
        z_new = z_q - state_lr * grad_z
        residual_new = z_new - pred
        lam_new = lam_q + alpha * residual_new
        # Upstream credit is the adjoint of the local prediction constraint.
        upstream = torch.autograd.grad(
            pred,
            z_prev,
            grad_outputs=(rho * residual_new + lam_new),
            create_graph=True,
            retain_graph=True,
        )[0]
        return torch.cat([z_new.reshape(-1), lam_new.reshape(-1), upstream.reshape(-1)])

    jac = torch.autograd.functional.jacobian(step, base, vectorize=True)
    state_n = 2 * n
    a = jac[:state_n, :state_n]
    b = jac[:state_n, state_n:]
    c = jac[state_n:, :state_n]
    d = jac[state_n:, state_n:]
    eye = torch.eye(state_n, dtype=jac.dtype)
    system = eye - a
    cond = float(torch.linalg.cond(system))
    transfer = d + c @ torch.linalg.solve(system, b)
    return transfer.detach(), cond


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--seed", type=int, default=980)
    parser.add_argument("--state-lr", type=float, default=0.05)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.2)
    args = parser.parse_args()

    torch.manual_seed(args.seed + 100_000)
    model = ResidualMLP(
        depth=args.depth,
        width=args.width,
        input_dim=args.width,
        output_dim=args.width,
        activation="relu",
        seed=args.seed,
        dtype=torch.float64,
    )
    x = torch.randn(1, args.width, dtype=torch.float64)
    acts = model.forward_activations(x)

    transfers: list[torch.Tensor] = []
    conditions: list[float] = []
    # Restrict to width-preserving residual hidden blocks; boundary layers are
    # intentionally excluded so every transfer has the same 8x8 interface.
    for layer in range(1, args.depth - 1):
        z_prev = acts[layer - 1].detach().requires_grad_(True)
        z = acts[layer].detach().requires_grad_(True)
        transfer, cond = local_transfer(
            model,
            layer,
            z_prev,
            z,
            state_lr=args.state_lr,
            rho=args.rho,
            alpha=args.alpha,
        )
        transfers.append(transfer)
        conditions.append(cond)

    gen = torch.Generator().manual_seed(args.seed + 200_000)
    start = torch.randn(args.width, generator=gen, dtype=torch.float64)
    exact = start.clone()
    for transfer in reversed(transfers):
        exact = transfer.T @ exact

    rows = []
    for rank in (1, 2, 4, args.width):
        approx = start.clone()
        for transfer in reversed(transfers):
            approx = low_rank(transfer, rank).T @ approx
        rows.append(
            {
                "rank": rank,
                "cosine": cosine(approx, exact),
                "relative_error": float((approx - exact).norm() / (exact.norm() + 1e-30)),
                "norm_ratio": float(approx.norm() / (exact.norm() + 1e-30)),
            }
        )

    print(
        json.dumps(
            {
                "depth": args.depth,
                "width": args.width,
                "seed": args.seed,
                "layers": len(transfers),
                "max_condition_I_minus_A": max(conditions, default=float("nan")),
                "rows": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
