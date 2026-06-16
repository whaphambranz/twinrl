from __future__ import annotations

import json
from pathlib import Path

import click
import torch

from twinrl_onco.debrief.progression import progression_report
from twinrl_onco.debrief.runner import collect_eval
from twinrl_onco.debrief.therapy import therapy_report
from twinrl_onco.flightdeck.trainer import Trainer
from twinrl_onco.flightplan import Config, load_config
from twinrl_onco.groundstation.cohort import Cohort, build_cohort
from twinrl_onco.groundstation.folds import fold_indices, holdout_split
from twinrl_onco.simenv.environment import behavior_rollout
from twinrl_onco.telemetry import (
    get_logger,
    load_checkpoint,
    save_checkpoint,
    select_device,
    set_seed,
)
from twinrl_onco.tower import build_model, make_environment

LOG = get_logger("panel")


def _build(
    config_path: str, overrides: tuple[str, ...], device_pref: str
) -> tuple[Config, torch.device]:
    config = load_config(config_path, list(overrides))
    device = select_device(device_pref)
    return config, device


def _test_index(config: Config, cohort: Cohort) -> torch.Tensor:
    if config.data.holdout_dataset:
        _, test_idx = holdout_split(cohort, config.data.datasets, config.data.holdout_dataset)
        return test_idx
    return fold_indices(cohort, config.data.n_folds, config.data.seed)[0][1]


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--config", "config_path", required=True, type=str)
@click.option("--set", "overrides", multiple=True)
@click.option("--device", "device_pref", default="auto")
@click.option("--checkpoint", "checkpoint", default="runs/twinrl.pt")
def train(config_path: str, overrides: tuple[str, ...], device_pref: str, checkpoint: str) -> None:
    config, device = _build(config_path, overrides, device_pref)
    trainer = Trainer(config, device)
    result = trainer.fit()
    save_checkpoint(
        {
            "state_dict": trainer.model.state_dict(),
            "config": config_path,
            "seed": config.train.seed,
        },
        checkpoint,
    )
    click.echo(json.dumps({"final_loss": result.final_loss, "val_loss": result.val_loss}, indent=2))


@cli.command()
@click.option("--config", "config_path", required=True, type=str)
@click.option("--set", "overrides", multiple=True)
@click.option("--device", "device_pref", default="auto")
@click.option("--checkpoint", "checkpoint", default=None)
def evaluate(
    config_path: str, overrides: tuple[str, ...], device_pref: str, checkpoint: str | None
) -> None:
    config, device = _build(config_path, overrides, device_pref)
    set_seed(config.train.seed)
    model = build_model(config).to(device)
    if checkpoint and Path(checkpoint).exists():
        model.load_state_dict(load_checkpoint(checkpoint)["state_dict"])
    cohort = build_cohort(config.data, config.model, seed=config.data.seed)
    test_idx = _test_index(config, cohort)
    bundle = collect_eval(model, cohort, test_idx, seed=config.train.seed + 5)
    click.echo(
        json.dumps(
            {"progression": progression_report(bundle), "therapy": therapy_report(bundle)}, indent=2
        )
    )


@cli.command()
@click.option("--config", "config_path", required=True, type=str)
@click.option("--set", "overrides", multiple=True)
@click.option("--device", "device_pref", default="auto")
@click.option("--checkpoint", "checkpoint", default=None)
@click.option("--limit", default=8, type=int)
def forecast(
    config_path: str,
    overrides: tuple[str, ...],
    device_pref: str,
    checkpoint: str | None,
    limit: int,
) -> None:
    config, device = _build(config_path, overrides, device_pref)
    set_seed(config.train.seed)
    model = build_model(config).to(device)
    if checkpoint and Path(checkpoint).exists():
        model.load_state_dict(load_checkpoint(checkpoint)["state_dict"])
    cohort = build_cohort(config.data, config.model, seed=config.data.seed)
    test_idx = _test_index(config, cohort)
    bundle = collect_eval(model, cohort, test_idx, seed=config.train.seed + 5)
    rows = [
        {
            "patient": int(i),
            "p_prog_12m": round(float(bundle.prog_score[12][i]), 4),
            "p_prog_24m": round(float(bundle.prog_score[24][i]), 4),
            "ttp_cycles": float(bundle.ttp[i]),
        }
        for i in range(min(limit, bundle.ttp.shape[0]))
    ]
    click.echo(json.dumps(rows, indent=2))


@cli.command()
@click.option("--config", "config_path", required=True, type=str)
@click.option("--set", "overrides", multiple=True)
@click.option("--device", "device_pref", default="auto")
def simulate(config_path: str, overrides: tuple[str, ...], device_pref: str) -> None:
    config, device = _build(config_path, overrides, device_pref)
    set_seed(config.train.seed)
    model = build_model(config).to(device)
    cohort = build_cohort(config.data, config.model, seed=config.data.seed)
    train_idx, _ = fold_indices(cohort, config.data.n_folds, config.data.seed)[0]
    env = make_environment(model, cohort, train_idx, seed=config.train.seed + 3)
    traj = behavior_rollout(env, config.sim.episode_length, seed=config.train.seed + 4)
    click.echo(
        json.dumps(
            {
                "episode_length": config.sim.episode_length,
                "mean_reward": round(float(traj.reward.mean()), 4),
                "progression_rate": round(float(traj.progression.mean()), 4),
            },
            indent=2,
        )
    )


@cli.command()
@click.option("--config", "config_path", required=True, type=str)
@click.option("--set", "overrides", multiple=True)
@click.option("--checkpoint", "checkpoint", default="runs/twinrl.pt")
@click.option("--out", "out_path", default="runs/twinrl_export.pt")
def export(config_path: str, overrides: tuple[str, ...], checkpoint: str, out_path: str) -> None:
    config = load_config(config_path, list(overrides))
    model = build_model(config)
    if Path(checkpoint).exists():
        model.load_state_dict(load_checkpoint(checkpoint)["state_dict"])
    save_checkpoint({"state_dict": model.state_dict()}, out_path)
    click.echo(json.dumps({"exported": out_path}, indent=2))


if __name__ == "__main__":
    cli()
