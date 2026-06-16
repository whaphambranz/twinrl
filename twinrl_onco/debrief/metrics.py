from __future__ import annotations

import numpy as np


def auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true).astype(np.float64)
    s = np.asarray(score).astype(np.float64)
    pos = s[y == 1]
    neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1)
    sorted_s = s[order]
    i = 0
    while i < sorted_s.size:
        j = i
        while j + 1 < sorted_s.size and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    rank_pos = ranks[y == 1].sum()
    n_pos = pos.size
    n_neg = neg.size
    return float((rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auprc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true).astype(np.float64)
    s = np.asarray(score).astype(np.float64)
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    precision = tp / np.maximum(tp + fp, 1e-12)
    total_pos = y.sum()
    if total_pos == 0:
        return float("nan")
    recall = tp / total_pos
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def brier(y_true: np.ndarray, prob: np.ndarray) -> float:
    y = np.asarray(y_true).astype(np.float64)
    p = np.asarray(prob).astype(np.float64)
    return float(np.mean((p - y) ** 2))


def brier_decomposition(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> dict[str, float]:
    y = np.asarray(y_true).astype(np.float64)
    p = np.asarray(prob).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    base = float(np.mean(y))
    reliability = 0.0
    resolution = 0.0
    for b in range(bins):
        mask = idx == b
        nk = int(mask.sum())
        if nk == 0:
            continue
        ok = float(np.mean(y[mask]))
        pk = float(np.mean(p[mask]))
        reliability += nk * (pk - ok) ** 2
        resolution += nk * (ok - base) ** 2
    n = y.size
    reliability /= n
    resolution /= n
    uncertainty = base * (1.0 - base)
    return {
        "calibration": reliability,
        "refinement": uncertainty - resolution,
        "uncertainty": uncertainty,
    }


def sensitivity_at_specificity(
    y_true: np.ndarray, score: np.ndarray, specificity: float = 0.9
) -> float:
    y = np.asarray(y_true).astype(np.float64)
    s = np.asarray(score).astype(np.float64)
    neg = s[y == 0]
    pos = s[y == 1]
    if neg.size == 0 or pos.size == 0:
        return float("nan")
    threshold = np.quantile(neg, specificity)
    return float(np.mean(pos >= threshold))


def regression_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true).astype(np.float64)
    p = np.asarray(y_pred).astype(np.float64)
    err = p - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    corr = float(np.corrcoef(y, p)[0, 1]) if y.std() > 0 and p.std() > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "corr": corr}
