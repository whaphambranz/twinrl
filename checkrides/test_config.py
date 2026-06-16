from __future__ import annotations

from pathlib import Path

from twinrl_onco.flightplan import load_config

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "flightplans" / "experiment"


def test_main_inherits_chain():
    cfg = load_config(str(EXP / "main.toml"))
    assert cfg.model.gene_dim == 2247
    assert cfg.model.beta == 0.1
    assert cfg.imagination.horizon == 15
    assert cfg.train.iterations == 400


def test_ablation_overrides_single_flag():
    cfg = load_config(str(EXP / "ablation_no_kl.toml"))
    assert cfg.model.use_kl is False
    assert cfg.model.use_stochastic is True
    assert cfg.model.gene_dim == 2247


def test_cli_override_applies():
    cfg = load_config(str(EXP / "main.toml"), ["model.beta=0.25", "train.iterations=5"])
    assert cfg.model.beta == 0.25
    assert cfg.train.iterations == 5


def test_all_experiment_configs_load():
    for path in EXP.glob("*.toml"):
        cfg = load_config(str(path))
        assert cfg.model.latent_dim > 0
