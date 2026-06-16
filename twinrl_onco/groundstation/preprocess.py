from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FoldPreprocessor:
    n_select: int
    grand_mean: torch.Tensor | None = None
    grand_std: torch.Tensor | None = None
    batch_mean: dict[int, torch.Tensor] | None = None
    batch_std: dict[int, torch.Tensor] | None = None
    select_index: torch.Tensor | None = None
    feat_mean: torch.Tensor | None = None
    feat_std: torch.Tensor | None = None

    def fit(self, gene: torch.Tensor, batch_ids: torch.Tensor) -> FoldPreprocessor:
        eps = 1e-6
        self.grand_mean = gene.mean(dim=0)
        self.grand_std = gene.std(dim=0).clamp_min(eps)
        self.batch_mean = {}
        self.batch_std = {}
        for bid in torch.unique(batch_ids).tolist():
            mask = batch_ids == bid
            self.batch_mean[int(bid)] = gene[mask].mean(dim=0)
            self.batch_std[int(bid)] = gene[mask].std(dim=0).clamp_min(eps)
        corrected = self._combat(gene, batch_ids)
        variance = corrected.var(dim=0)
        k = min(self.n_select, variance.numel())
        self.select_index = torch.topk(variance, k).indices.sort().values
        selected = corrected.index_select(1, self.select_index)
        self.feat_mean = selected.mean(dim=0)
        self.feat_std = selected.std(dim=0).clamp_min(eps)
        return self

    def _combat(self, gene: torch.Tensor, batch_ids: torch.Tensor) -> torch.Tensor:
        assert self.grand_mean is not None and self.grand_std is not None
        assert self.batch_mean is not None and self.batch_std is not None
        out = gene.clone()
        for bid in torch.unique(batch_ids).tolist():
            key = int(bid)
            if key not in self.batch_mean:
                continue
            mask = batch_ids == bid
            standardized = (gene[mask] - self.batch_mean[key]) / self.batch_std[key]
            out[mask] = standardized * self.grand_std + self.grand_mean
        return out

    def transform(self, gene: torch.Tensor, batch_ids: torch.Tensor) -> torch.Tensor:
        if self.select_index is None or self.feat_mean is None or self.feat_std is None:
            raise RuntimeError("preprocessor must be fit before transform")
        corrected = self._combat(gene, batch_ids)
        selected = corrected.index_select(1, self.select_index)
        return (selected - self.feat_mean) / self.feat_std
