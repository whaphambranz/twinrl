from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import msgspec


class ModelPlan(msgspec.Struct, frozen=True):
    gene_dim: int = 2247
    flow_dim: int = 256
    clinical_dim: int = 32
    fuse_dim: int = 256
    latent_dim: int = 64
    hidden_dim: int = 256
    n_heads: int = 4
    beta: float = 0.1
    action_dim: int = 32
    use_stochastic: bool = True
    use_kl: bool = True
    use_gene_attention: bool = True
    use_multihead_encoder: bool = True
    backbone: str = "gru"


class PolicyPlan(msgspec.Struct, frozen=True):
    phases: tuple[str, ...] = ("induction", "consolidation", "maintenance", "holiday")
    dose_levels_mg: tuple[float, ...] = (0.0, 100.0, 200.0, 400.0)
    combos: tuple[str, ...] = ("mono", "rituximab")
    gamma: float = 0.99
    lam: float = 0.95
    phase_duration: int = 6
    hierarchical: bool = True
    fixed_phase_duration: bool = False
    phase_specific_lowlevel: bool = True


class ImaginationPlan(msgspec.Struct, frozen=True):
    horizon: int = 15
    n_rollouts: int = 50
    use_rollouts: bool = True
    aggregate: bool = True
    progression_horizon: int = 12


class SimPlan(msgspec.Struct, frozen=True):
    growth_sensitive: float = 0.18
    growth_resistant: float = 0.22
    emax: float = 0.95
    ec50_mg: float = 145.0
    hill: float = 1.4
    resistant_kill_ratio: float = 0.35
    gly101val_rate: float = 0.012
    mcl1_rescue: float = 0.08
    rituximab_boost: float = 0.12
    baseline_burden: float = 1.0
    cr_threshold: float = 0.05
    pr_threshold: float = 0.5
    pd_threshold: float = 1.2
    w_resp: float = 1.0
    w_tox: float = 0.3
    w_prog: float = 0.5
    w_qol: float = 0.2
    episode_length: int = 24


class DataPlan(msgspec.Struct, frozen=True):
    datasets: tuple[str, ...] = ("GSE39671", "GSE22762", "GSE50006")
    n_patients: int = 348
    n_folds: int = 5
    combat: bool = True
    ighv_unmutated_frac: float = 0.494
    del17p_frac: float = 0.095
    tp53_frac: float = 0.144
    biology_dim: int = 16
    holdout_dataset: str = ""
    seed: int = 0


class TrainPlan(msgspec.Struct, frozen=True):
    iterations: int = 400
    batch_size: int = 32
    seq_len: int = 15
    world_lr: float = 3e-4
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    grad_clip: float = 100.0
    amp: bool = False
    use_imagination: bool = True
    early_stopping: bool = True
    patience: int = 20
    seed: int = 0
    device: str = "auto"


class Config(msgspec.Struct, frozen=True):
    model: ModelPlan = msgspec.field(default_factory=ModelPlan)
    policy: PolicyPlan = msgspec.field(default_factory=PolicyPlan)
    imagination: ImaginationPlan = msgspec.field(default_factory=ImaginationPlan)
    sim: SimPlan = msgspec.field(default_factory=SimPlan)
    data: DataPlan = msgspec.field(default_factory=DataPlan)
    train: TrainPlan = msgspec.field(default_factory=TrainPlan)
    name: str = "main"


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_with_extends(path: Path, seen: set[Path]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"cyclic extends chain at {resolved}")
    seen.add(resolved)
    with resolved.open("rb") as handle:
        raw = tomllib.load(handle)
    parent_ref = raw.pop("extends", None)
    merged: dict[str, Any] = {}
    if parent_ref is not None:
        parent_path = (resolved.parent / str(parent_ref)).resolve()
        merged = _read_with_extends(parent_path, seen)
    return _deep_merge(merged, raw)


def _coerce(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
    if "," in value:
        return [_coerce(part) for part in value.split(",")]
    return value


def _apply_override(tree: dict[str, Any], dotted: str) -> None:
    key, _, raw = dotted.partition("=")
    if not raw:
        raise ValueError(f"override must be key=value, got {dotted!r}")
    cursor = tree
    parts = key.strip().split(".")
    for part in parts[:-1]:
        nxt = cursor.setdefault(part, {})
        if not isinstance(nxt, dict):
            raise ValueError(f"cannot descend into non-table at {part!r}")
        cursor = nxt
    cursor[parts[-1]] = _coerce(raw)


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    tree = _read_with_extends(Path(path), set())
    for dotted in overrides or []:
        _apply_override(tree, dotted)
    return msgspec.convert(tree, Config)


def to_dict(config: Config) -> dict[str, Any]:
    return msgspec.to_builtins(config)
