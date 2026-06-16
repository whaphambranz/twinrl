from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from twinrl_onco.groundstation.cohort import Cohort


def stratum_labels(cohort: Cohort) -> np.ndarray:
    ighv = cohort.labels["ighv_unmutated"].numpy().astype(np.int64)
    source = cohort.dataset_id.numpy().astype(np.int64)
    return ighv * 100 + source


def fold_indices(
    cohort: Cohort, n_folds: int, seed: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    strata = stratum_labels(cohort)
    n = strata.shape[0]
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds: list[tuple[torch.Tensor, torch.Tensor]] = []
    for train_idx, test_idx in splitter.split(np.zeros(n), strata):
        folds.append(
            (
                torch.from_numpy(train_idx.astype(np.int64)),
                torch.from_numpy(test_idx.astype(np.int64)),
            )
        )
    return folds


def holdout_split(
    cohort: Cohort, datasets: tuple[str, ...], holdout: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if holdout not in datasets:
        raise ValueError(f"holdout {holdout!r} not among {datasets}")
    held = datasets.index(holdout)
    source = cohort.dataset_id
    test_idx = torch.nonzero(source == held, as_tuple=False).squeeze(-1)
    train_idx = torch.nonzero(source != held, as_tuple=False).squeeze(-1)
    return train_idx, test_idx


def split_train_val(
    train_idx: torch.Tensor, val_fraction: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    perm = train_idx[torch.randperm(train_idx.numel(), generator=gen)]
    n_val = max(1, int(round(perm.numel() * val_fraction)))
    return perm[n_val:], perm[:n_val]
