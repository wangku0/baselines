from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=t.dtype) * (-math.log(10000.0) / max(half - 1, 1))
        )
        args = t * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class FlowVectorField(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        cond_dim: int,
        num_layers: int = 32,
        hidden_width: int = 1024,
        time_embedding_dim: int = 128,
        layer_embedding_dim: int = 16,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.cond_dim = int(cond_dim)
        self.num_layers = int(num_layers)
        self.time_embedding_dim = int(time_embedding_dim)
        self.layer_embedding_dim = int(layer_embedding_dim)
        self.x_norm = nn.LayerNorm(self.hidden_dim)
        self.time = SinusoidalTimeEmbedding(self.time_embedding_dim)
        self.layer_emb = nn.Embedding(max(self.num_layers + 1, 1), self.layer_embedding_dim)
        in_dim = self.hidden_dim + self.cond_dim + self.time_embedding_dim + self.layer_embedding_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, hidden_width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, self.hidden_dim),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        layer_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x_t.ndim != 2:
            raise ValueError(f"FlowVectorField expects x_t [B,D], got {tuple(x_t.shape)}")
        if cond.ndim != 2:
            raise ValueError(f"FlowVectorField expects cond [B,C], got {tuple(cond.shape)}")
        if layer_id is None:
            layer_id = torch.zeros(x_t.shape[0], dtype=torch.long, device=x_t.device)
        layer_id = layer_id.to(device=x_t.device, dtype=torch.long).clamp(0, self.num_layers)
        h = torch.cat([self.x_norm(x_t), cond.to(x_t.dtype), self.time(t.to(x_t.dtype)), self.layer_emb(layer_id)], dim=-1)
        return self.net(h)


@torch.no_grad()
def euler_integrate_flow(
    model: FlowVectorField,
    x0: torch.Tensor,
    cond: torch.Tensor,
    *,
    layer_id: Optional[torch.Tensor] = None,
    steps: int = 8,
) -> torch.Tensor:
    x = x0
    steps = max(1, int(steps))
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((x.shape[0], 1), i / steps, device=x.device, dtype=x.dtype)
        x = x + dt * model(x, t, cond, layer_id)
    return x


def euler_integrate_flow_trainable(
    model: FlowVectorField,
    x0: torch.Tensor,
    cond: torch.Tensor,
    *,
    layer_id: Optional[torch.Tensor] = None,
    steps: int = 8,
) -> torch.Tensor:
    x = x0
    steps = max(1, int(steps))
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((x.shape[0], 1), i / steps, device=x.device, dtype=x.dtype)
        x = x + dt * model(x, t, cond, layer_id)
    return x

