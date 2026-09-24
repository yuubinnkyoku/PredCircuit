from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from predcircuit.pcalm import ResidualMLP


def truncated_residual_jacobian(jac: torch.Tensor, rank: int) -> torch.Tensor:
    """Keep the identity skip exactly and truncate only J-I."""
    eye = torch.eye(jac.shape[0], dtype=jac.dtype, device=jac.device)
    residual = jac - eye
    u, s, vh = torch.linalg.svd(residual, full_matrices=False)
    return eye + (u[:, :rank] * s[:rank]) @ vh[:rank]


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30))


def main() -> None:
    p = argparse.ArgumentParser(description="Probe compressibility of deep residual credit transport.")
    p.add_argument("--seeds", default="980,981,982,983")
    p.add_argument("--depth", type=int, default=32)
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--ranks", default="1,2,4,8")
    p.add_argument("--out", type=Path, default=Path("results/generated/credit_jacobian_rank_probe.csv"))
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    ranks = [int(x) for x in args.ranks.split(",")]
    rows = []
    for seed in seeds:
        model = ResidualMLP(depth=args.depth, width=args.width, input_dim=args.width, output_dim=4,
                            activation="relu", seed=seed + args.depth, dtype=torch.float64)
        gen = torch.Generator().manual_seed(seed + 10_000 + args.depth)
        x = torch.randn(args.batch_size, args.width, generator=gen, dtype=torch.float64)
        y = torch.randn(args.batch_size, 4, generator=gen, dtype=torch.float64)
        zs = model.forward_activations(x)
        for b in range(args.batch_size):
            out_mask = (zs[-2][b] > 0).to(torch.float64)
            jout = model.scales[-1] * model.weights[-1].detach() @ torch.diag(out_mask)
            start = jout.T @ (zs[-1][b].detach() - y[b])
            jacobians = []
            for layer in range(1, args.depth - 1):
                mask = (zs[layer - 1][b] > 0).to(torch.float64)
                jac = torch.eye(args.width, dtype=torch.float64)
                jac = jac + model.scales[layer] * model.weights[layer].detach() @ torch.diag(mask)
                jacobians.append(jac)
            exact = start.clone()
            for jac in reversed(jacobians):
                exact = jac.T @ exact
            diag = start.clone()
            for jac in reversed(jacobians):
                diag = torch.diag(torch.diag(jac)).T @ diag
            rows.append({"seed": seed, "sample": b, "approx": "diag", "rank": 0,
                         "cosine": cosine(diag, exact), "relative_error": float((diag-exact).norm()/(exact.norm()+1e-30)),
                         "norm_ratio": float(diag.norm()/(exact.norm()+1e-30))})
            for rank in ranks:
                approx = start.clone()
                for jac in reversed(jacobians):
                    ja = jac if rank >= args.width else truncated_residual_jacobian(jac, rank)
                    approx = ja.T @ approx
                rows.append({"seed": seed, "sample": b, "approx": f"residual_rank_{rank}", "rank": rank,
                             "cosine": cosine(approx, exact), "relative_error": float((approx-exact).norm()/(exact.norm()+1e-30)),
                             "norm_ratio": float(approx.norm()/(exact.norm()+1e-30))})
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.groupby("approx")[["cosine", "relative_error", "norm_ratio"]].agg(["mean", "median"]).to_string())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
