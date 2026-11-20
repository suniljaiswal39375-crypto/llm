"""Bias-free SwiGLU Feed-Forward Network."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """Modern bias-free SwiGLU MLP: down(silu(gate(x)) * up(x))."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.w_gate(x)) * self.w_up(x)
        if self.dropout > 0.0 and self.training:
            hidden = F.dropout(hidden, p=self.dropout, training=True)
        return self.w_down(hidden)
