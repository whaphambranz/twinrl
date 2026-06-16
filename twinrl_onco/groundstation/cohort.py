from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from twinrl_onco.flightplan import DataPlan, ModelPlan

DATASET_SIZES: dict[str, int] = {"GSE39671": 130, "GSE22762": 107, "GSE50006": 111}
DATASET_ENDPOINTS: dict[str, str] = {
    "GSE39671": "TTFT",
    "GSE22762": "OS",
    "GSE50006": "Response",
}
PATHWAY_GENES = 64


@dataclass(frozen=True)
class Cohort:
    gene: torch.Tensor
    flow: torch.Tensor
    clinical: torch.Tensor
    biology: torch.Tensor
    dataset_id: torch.Tensor
    labels: dict[str, torch.Tensor]
    sim_params: dict[str, torch.Tensor]
    pathway_index: torch.Tensor


def dataset_table(plan: DataPlan) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in plan.datasets:
        rows.append(
            {
                "dataset": name,
                "samples": DATASET_SIZES.get(name, 0),
                "platform": "HG-U133 Plus 2.0",
                "endpoint": DATASET_ENDPOINTS.get(name, "Multiple"),
            }
        )
    return rows


def _allocate_sources(plan: DataPlan, rng: np.random.Generator) -> np.ndarray:
    counts = [DATASET_SIZES.get(name, 0) for name in plan.datasets]
    total = sum(counts)
    if total != plan.n_patients:
        scaled = np.full(len(plan.datasets), plan.n_patients // len(plan.datasets))
        scaled[: plan.n_patients % len(plan.datasets)] += 1
        counts = scaled.tolist()
    ids = np.concatenate([np.full(c, i) for i, c in enumerate(counts)])
    rng.shuffle(ids)
    return ids[: plan.n_patients]


def build_cohort(data: DataPlan, model: ModelPlan, seed: int | None = None) -> Cohort:
    rng = np.random.default_rng(data.seed if seed is None else seed)
    n = data.n_patients
    source = _allocate_sources(data, rng)

    ighv_unmut = (rng.random(n) < data.ighv_unmutated_frac).astype(np.float32)
    del17p = (rng.random(n) < data.del17p_frac).astype(np.float32)
    tp53 = (rng.random(n) < data.tp53_frac).astype(np.float32)
    binet = rng.choice(3, size=n, p=(0.41, 0.37, 0.22)).astype(np.int64)
    age = rng.normal(66.0, 9.0, size=n).astype(np.float32)
    age_ge65 = (age >= 65.0).astype(np.float32)

    bdim = data.biology_dim
    biology = rng.normal(0.0, 1.0, size=(n, bdim)).astype(np.float32)
    risk = 0.9 * ighv_unmut + 1.1 * del17p + 1.0 * tp53 + 0.35 * binet
    biology[:, 0] = 0.6 * ighv_unmut + 0.3 * rng.standard_normal(n)
    biology[:, 1] = 0.8 * (del17p + tp53) + 0.3 * rng.standard_normal(n)
    biology[:, 2] = 0.5 * ighv_unmut + 0.25 * rng.standard_normal(n)
    biology[:, 3] = 0.7 * risk / 2.0 + 0.25 * rng.standard_normal(n)
    biology[:, 4] = 0.5 * (del17p + 0.5 * tp53) + 0.25 * rng.standard_normal(n)
    biology[:, 5] = 0.6 * risk / 2.0 + 0.3 * rng.standard_normal(n)
    biology[:, 6] = -0.4 * risk / 2.0 + 0.3 * rng.standard_normal(n)

    bt = torch.from_numpy(biology)

    gene_loadings = torch.from_numpy(
        rng.standard_normal((bdim, model.gene_dim)).astype(np.float32) * 0.5
    )
    flow_loadings = torch.from_numpy(
        rng.standard_normal((bdim, model.flow_dim)).astype(np.float32) * 0.5
    )
    batch_offsets = torch.from_numpy(
        rng.standard_normal((len(data.datasets), model.gene_dim)).astype(np.float32) * 0.4
    )

    gene_clean = torch.tanh(bt @ gene_loadings)
    gene = gene_clean + batch_offsets[torch.from_numpy(source)]
    gene = gene + 0.02 * torch.from_numpy(
        rng.standard_normal((n, model.gene_dim)).astype(np.float32)
    )

    pathway_signal = 0.9 * bt[:, 1:2] + 0.7 * bt[:, 5:6] - 0.5 * bt[:, 0:1]
    gene[:, :PATHWAY_GENES] = gene[:, :PATHWAY_GENES] + pathway_signal

    flow = torch.tanh(bt @ flow_loadings) + 0.02 * torch.from_numpy(
        rng.standard_normal((n, model.flow_dim)).astype(np.float32)
    )

    clinical = torch.zeros((n, model.clinical_dim), dtype=torch.float32)
    clinical[:, 0] = torch.from_numpy(ighv_unmut)
    clinical[:, 1] = torch.from_numpy(del17p)
    clinical[:, 2] = torch.from_numpy(tp53)
    clinical[:, 3] = torch.from_numpy(binet.astype(np.float32)) / 2.0
    clinical[:, 4] = torch.from_numpy((age - 66.0) / 9.0)
    clinical[:, 5] = torch.from_numpy(age_ge65)
    clinical[:, 6] = bt[:, 6]
    if model.clinical_dim > 7:
        clinical[:, 7:] = 0.3 * torch.from_numpy(
            rng.standard_normal((n, model.clinical_dim - 7)).astype(np.float32)
        )

    growth_s = 0.18 * (1.0 + 0.3 * bt[:, 2])
    growth_r = 0.22 * (1.0 + 0.3 * bt[:, 3])
    ec50 = 145.0 * (1.0 + 0.25 * bt[:, 4])
    resist_prop = torch.sigmoid(bt[:, 5]) * 0.04 + 0.004
    mcl1 = torch.sigmoid(bt[:, 1])

    labels = {
        "ighv_unmutated": torch.from_numpy(ighv_unmut),
        "del17p": torch.from_numpy(del17p),
        "tp53": torch.from_numpy(tp53),
        "binet": torch.from_numpy(binet),
        "age": torch.from_numpy(age),
        "age_ge65": torch.from_numpy(age_ge65),
        "risk": torch.from_numpy((risk / 2.0).astype(np.float32)),
    }
    sim_params = {
        "growth_sensitive": growth_s.clone(),
        "growth_resistant": growth_r.clone(),
        "ec50_mg": ec50.clone(),
        "resist_prop": resist_prop.clone(),
        "mcl1": mcl1.clone(),
    }
    return Cohort(
        gene=gene,
        flow=flow,
        clinical=clinical,
        biology=bt,
        dataset_id=torch.from_numpy(source),
        labels=labels,
        sim_params=sim_params,
        pathway_index=torch.arange(PATHWAY_GENES),
    )
