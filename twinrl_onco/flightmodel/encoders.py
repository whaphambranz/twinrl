from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from twinrl_onco.flightplan import ModelPlan
from twinrl_onco.simenv.environment import PATHWAY


class GeneAttention(nn.Module):
    def __init__(self, gene_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(gene_dim, gene_dim // 8),
            nn.GELU(),
            nn.Linear(gene_dim // 8, gene_dim),
        )
        prior = torch.zeros(gene_dim)
        prior[:PATHWAY] = 1.0
        self.pathway_prior = nn.Parameter(prior, requires_grad=False)
        self.bias_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, gene: torch.Tensor) -> torch.Tensor:
        logits = self.gate(gene) + self.bias_scale * self.pathway_prior
        weights = torch.sigmoid(logits)
        return gene * weights


class GeneEncoder(nn.Module):
    def __init__(self, model: ModelPlan) -> None:
        super().__init__()
        self.use_attention = model.use_gene_attention
        self.use_multihead = model.use_multihead_encoder
        self.attention = GeneAttention(model.gene_dim) if self.use_attention else None
        out = model.fuse_dim
        self.proj: nn.Module
        if self.use_multihead:
            heads = model.n_heads
            per = out // heads
            self.heads = nn.ModuleList([nn.Linear(model.gene_dim, per) for _ in range(heads)])
            self.proj = nn.Linear(per * heads, out)
        else:
            self.heads = nn.ModuleList([nn.Linear(model.gene_dim, out)])
            self.proj = nn.Identity()

    def forward(self, gene: torch.Tensor) -> torch.Tensor:
        if self.attention is not None:
            gene = self.attention(gene)
        parts = [head(gene) for head in self.heads]
        return self.proj(torch.cat(parts, dim=-1))


class ObservationEncoder(nn.Module):
    def __init__(self, model: ModelPlan) -> None:
        super().__init__()
        self.gene = GeneEncoder(model)
        self.flow = nn.Linear(model.flow_dim, model.fuse_dim)
        self.clinical = nn.Linear(model.clinical_dim, model.fuse_dim)
        self.fuse = nn.Sequential(
            nn.Linear(model.fuse_dim * 3, model.fuse_dim),
            nn.GELU(),
            nn.Linear(model.fuse_dim, model.fuse_dim),
        )

    def forward(
        self, gene: torch.Tensor, flow: torch.Tensor, clinical: torch.Tensor
    ) -> torch.Tensor:
        g = F.gelu(self.gene(gene))
        f = F.gelu(self.flow(flow))
        c = F.gelu(self.clinical(clinical))
        return self.fuse(torch.cat([g, f, c], dim=-1))


class ActionEncoder(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(feature_dim, action_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(features))
