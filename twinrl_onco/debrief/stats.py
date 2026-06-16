from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import stats

from twinrl_onco.debrief.metrics import auroc


def paired_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    result = stats.ttest_rel(np.asarray(a), np.asarray(b))
    return float(result.statistic), float(result.pvalue)


def bh_fdr(pvalues: np.ndarray, q: float = 0.05) -> dict[str, object]:
    p = np.asarray(pvalues, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresholds = q * (np.arange(1, n + 1) / n)
    passed = ranked <= thresholds
    if not passed.any():
        rejected = np.zeros(n, dtype=bool)
        qmax = 0.0
    else:
        kmax = int(np.max(np.where(passed)[0]))
        cutoff = ranked[kmax]
        rejected = p <= cutoff
        qmax = float(ranked[kmax] * n / (kmax + 1))
    return {"rejected": rejected, "n_rejected": int(rejected.sum()), "qmax": qmax}


def bonferroni(pvalues: np.ndarray, n_tests: int | None = None) -> np.ndarray:
    p = np.asarray(pvalues, dtype=np.float64)
    m = n_tests if n_tests is not None else p.size
    return np.minimum(p * m, 1.0)


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    data = np.asarray(values)
    estimates = np.empty(n_boot)
    for i in range(n_boot):
        sample = data[rng.integers(0, data.shape[0], data.shape[0])]
        estimates[i] = statistic(sample)
    lo = float(np.quantile(estimates, alpha / 2))
    hi = float(np.quantile(estimates, 1 - alpha / 2))
    return lo, hi


def bca_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    data = np.asarray(values)
    n = data.shape[0]
    theta_hat = statistic(data)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = statistic(data[rng.integers(0, n, n)])
    prop = float(np.mean(boot < theta_hat))
    prop = min(max(prop, 1e-6), 1 - 1e-6)
    z0 = stats.norm.ppf(prop)
    jack = np.empty(n)
    for i in range(n):
        jack[i] = statistic(np.delete(data, i))
    jack_mean = jack.mean()
    num = np.sum((jack_mean - jack) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5 + 1e-12)
    acc = num / den
    z_lo = stats.norm.ppf(alpha / 2)
    z_hi = stats.norm.ppf(1 - alpha / 2)
    a_lo = stats.norm.cdf(z0 + (z0 + z_lo) / (1 - acc * (z0 + z_lo)))
    a_hi = stats.norm.cdf(z0 + (z0 + z_hi) / (1 - acc * (z0 + z_hi)))
    lo = float(np.quantile(boot, a_lo))
    hi = float(np.quantile(boot, a_hi))
    return lo, hi


def delong(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true)
    pos = y == 1
    neg = y == 0
    m = int(pos.sum())
    n = int(neg.sum())
    scores = np.vstack(
        [np.asarray(score_a, dtype=np.float64), np.asarray(score_b, dtype=np.float64)]
    )
    auc = np.empty(2)
    v10 = np.empty((2, m))
    v01 = np.empty((2, n))
    for k in range(2):
        x = scores[k, pos]
        ya = scores[k, neg]
        comp = x[:, None] - ya[None, :]
        placement = (comp > 0).astype(np.float64) + 0.5 * (comp == 0).astype(np.float64)
        auc[k] = float(placement.mean())
        v10[k] = placement.mean(axis=1)
        v01[k] = placement.mean(axis=0)
    s10 = np.cov(v10)
    s01 = np.cov(v01)
    cov = s10 / m + s01 / n
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    diff = auc[0] - auc[1]
    if var <= 0:
        z = 0.0
        p = 1.0
    else:
        z = float(diff / np.sqrt(var))
        p = float(2 * stats.norm.sf(abs(z)))
    return {"auc_a": float(auc[0]), "auc_b": float(auc[1]), "z": z, "pvalue": p}


def hosmer_lemeshow(y_true: np.ndarray, prob: np.ndarray, groups: int = 10) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(prob, dtype=np.float64)
    order = np.argsort(p)
    y = y[order]
    p = p[order]
    splits = np.array_split(np.arange(y.size), groups)
    chi = 0.0
    for idx in splits:
        if idx.size == 0:
            continue
        obs = y[idx].sum()
        exp = p[idx].sum()
        nk = idx.size
        denom = exp * (1 - exp / nk) if 0 < exp < nk else 1e-9
        chi += (obs - exp) ** 2 / max(denom, 1e-9)
    dof = max(groups - 2, 1)
    pval = float(stats.chi2.sf(chi, dof))
    return {"chi2": float(chi), "dof": float(dof), "pvalue": pval}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    sd = diff.std(ddof=1)
    if sd == 0:
        return float("inf")
    return float(diff.mean() / sd)


def fine_gray_subdistribution(
    y_event: np.ndarray, score: np.ndarray, competing: np.ndarray
) -> float:
    y = np.asarray(y_event, dtype=np.float64)
    comp = np.asarray(competing, dtype=np.float64)
    weight = np.where(comp == 1, 0.5, 1.0)
    keep = weight > 0
    return auroc(y[keep], np.asarray(score)[keep])
