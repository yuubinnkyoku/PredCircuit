from __future__ import annotations

import torch


def xor_task() -> tuple[torch.Tensor, torch.Tensor]:
    """XOR inputs/targets, kept as a nonlinear diagnostic rather than a smoke test."""
    x = torch.tensor(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], dtype=torch.float32
    )
    y = torch.tensor([[-0.8], [0.8], [0.8], [-0.8]], dtype=torch.float32)
    return x, y


def linear_mapping_task() -> tuple[torch.Tensor, torch.Tensor]:
    """A tiny supervised regression task used only to verify learning machinery."""
    x = torch.tensor(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], dtype=torch.float32
    )
    y = 0.5 * x[:, :1] - 0.25 * x[:, 1:2]
    return x, y


def parity_task(bits: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    if bits < 1:
        raise ValueError("bits must be positive")
    rows = 2**bits
    raw = torch.tensor(
        [[(i >> j) & 1 for j in range(bits)] for i in range(rows)], dtype=torch.float32
    )
    x = raw * 2.0 - 1.0
    parity = (raw.sum(dim=1).remainder(2) * 2.0 - 1.0).unsqueeze(1) * 0.8
    return x, parity
