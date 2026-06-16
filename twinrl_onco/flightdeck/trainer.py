from __future__ import annotations

from dataclasses import dataclass, field

import torch

from twinrl_onco.cockpit.actorcritic import actor_loss, critic_loss, lambda_returns
from twinrl_onco.flightdeck.optim import EarlyStopper, build_optimizers, clip_grads
from twinrl_onco.flightdeck.replay import ReplayBuffer
from twinrl_onco.flightmodel.elbo import progression_loss, vrssm_objective
from twinrl_onco.flightplan import Config
from twinrl_onco.groundstation.cohort import Cohort, build_cohort
from twinrl_onco.groundstation.folds import fold_indices, split_train_val
from twinrl_onco.imagination.rollout import imagine
from twinrl_onco.simenv.environment import Trajectory, TreatmentEnvironment, behavior_rollout
from twinrl_onco.telemetry import generator_for, get_logger, set_seed
from twinrl_onco.tower import TwinRLOnco, build_model, make_environment


@dataclass
class TrainResult:
    history: list[dict[str, float]] = field(default_factory=list)
    final_loss: float = 0.0
    val_loss: float = 0.0


class Trainer:
    def __init__(self, config: Config, device: torch.device | None = None) -> None:
        self.config = config
        self.device = device or torch.device("cpu")
        self.log = get_logger("flightdeck")
        set_seed(config.train.seed)
        self.model: TwinRLOnco = build_model(config).to(self.device)
        self.opt = build_optimizers(self.model, config.train)
        self.gen = generator_for(config.train.seed + 1)

    def _features(self, action: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        dose_mg, combo_idx = self.model.grid.decode(action)
        return self.model.grid.features(dose_mg, combo_idx, phase)

    def _world_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        feats = self._features(batch["action"], batch["phase"])
        state = self.model.world.observe(
            batch["gene"], batch["flow"], batch["clinical"], feats, self.gen
        )
        obs = {"gene": batch["gene"], "flow": batch["flow"], "clinical": batch["clinical"]}
        terms = self.model.world.per_step_terms(state, obs, batch["reward"], batch["continuation"])
        loss, report = vrssm_objective(terms, self.config.model.beta, self.config.model.use_kl)
        plogit = self.model.world.progression_logits(state)
        prog = progression_loss(plogit, batch["progression"])
        total = loss + prog
        self.opt.world.zero_grad()
        total.backward()
        clip_grads(list(self.model.world.parameters()), self.config.train.grad_clip)
        self.opt.world.step()
        report["world_loss"] = float(total.detach())
        report["prog_loss"] = float(prog.detach())
        return report

    def _policy_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        feats = self._features(batch["action"], batch["phase"])
        with torch.no_grad():
            state = self.model.world.observe(
                batch["gene"], batch["flow"], batch["clinical"], feats, self.gen
            )
        h0 = state.hidden.reshape(-1, state.hidden.shape[-1]).detach()
        z0 = state.latent.reshape(-1, state.latent.shape[-1]).detach()
        rollout = imagine(
            self.model.world,
            self.model.policy,
            h0,
            z0,
            self.config.imagination.horizon,
            self.config.model.use_stochastic,
            self.gen,
        )
        gamma = self.config.policy.gamma
        lam = self.config.policy.lam
        with torch.no_grad():
            vlow_seq = torch.stack(
                [
                    self.model.policy.low_value(rollout.hidden[t], rollout.latent[t])
                    for t in range(rollout.hidden.shape[0])
                ]
            )
            vlow_last = self.model.policy.low_value(rollout.hidden_last, rollout.latent_last)
            vlow_boot = torch.cat([vlow_seq, vlow_last.unsqueeze(0)], dim=0)
            target_low = lambda_returns(rollout.reward, rollout.continuation, vlow_boot, gamma, lam)
            vhigh_seq = torch.stack(
                [
                    self.model.policy.high_value(rollout.hidden[t], rollout.latent[t])
                    for t in range(rollout.hidden.shape[0])
                ]
            )
            vhigh_last = self.model.policy.high_value(rollout.hidden_last, rollout.latent_last)
            vhigh_boot = torch.cat([vhigh_seq, vhigh_last.unsqueeze(0)], dim=0)
            target_high = lambda_returns(
                rollout.reward, rollout.continuation, vhigh_boot, gamma, lam
            )

        vlow_pred = torch.stack(
            [
                self.model.policy.low_value(rollout.hidden[t], rollout.latent[t])
                for t in range(rollout.hidden.shape[0])
            ]
        )
        vhigh_pred = torch.stack(
            [
                self.model.policy.high_value(rollout.hidden[t], rollout.latent[t])
                for t in range(rollout.hidden.shape[0])
            ]
        )
        logp_high, logp_low = [], []
        for t in range(rollout.hidden.shape[0]):
            lh, ll = self.model.policy.log_probs(
                rollout.hidden[t], rollout.latent[t], rollout.phase[t], rollout.action[t]
            )
            logp_high.append(lh)
            logp_low.append(ll)
        lp_high = torch.stack(logp_high)
        lp_low = torch.stack(logp_low)

        adv_low = target_low - vlow_pred.detach()
        adv_high = target_high - vhigh_pred.detach()
        a_loss = actor_loss(lp_low, adv_low)
        if self.config.policy.hierarchical:
            a_loss = a_loss + actor_loss(lp_high, adv_high)
        c_loss = critic_loss(vlow_pred, target_low) + critic_loss(vhigh_pred, target_high)

        self.opt.actor.zero_grad()
        self.opt.critic.zero_grad()
        (a_loss + c_loss).backward()
        clip_grads(
            list(self.model.policy.meta.parameters()) + list(self.model.policy.low.parameters()),
            self.config.train.grad_clip,
        )
        self.opt.actor.step()
        self.opt.critic.step()
        return {
            "actor_loss": float(a_loss.detach()),
            "critic_loss": float(c_loss.detach()),
            "imagined_return": float(rollout.reward.sum(dim=0).mean().detach()),
        }

    def collect(self, env: TreatmentEnvironment, length: int) -> Trajectory:
        world = self.model.world
        policy = self.model.policy
        obs = env.reset()
        h = world.core.reset(env.n, self.device)
        z = torch.zeros(env.n, self.config.model.latent_dim, device=self.device)
        a_emb = torch.zeros(env.n, self.config.model.action_dim, device=self.device)
        genes, flows, clinicals = [], [], []
        actions, phases, rewards, conts, progs = [], [], [], [], []
        with torch.no_grad():
            for _ in range(length):
                h = world.core.step(h, z, a_emb)
                embed = world.encoder(obs["gene"], obs["flow"], obs["clinical"])
                post_mean, post_std = world.posterior(h, embed)
                z = post_mean
                decision = policy.act(h, z, self.gen)
                genes.append(obs["gene"])
                flows.append(obs["flow"])
                clinicals.append(obs["clinical"])
                obs, reward, cont, prog = env.step(decision["dose_mg"], decision["combo_idx"])
                actions.append(decision["action"])
                phases.append(decision["phase"])
                rewards.append(reward)
                conts.append(cont)
                progs.append(prog)
                a_emb = world.encode_actions(decision["features"])
        return Trajectory(
            gene=torch.stack(genes),
            flow=torch.stack(flows),
            clinical=torch.stack(clinicals),
            action=torch.stack(actions),
            phase=torch.stack(phases),
            reward=torch.stack(rewards),
            continuation=torch.stack(conts),
            progression=torch.stack(progs),
        )

    def fit(self, cohort: Cohort | None = None) -> TrainResult:
        cfg = self.config
        set_seed(cfg.train.seed)
        if cohort is None:
            cohort = build_cohort(cfg.data, cfg.model, seed=cfg.data.seed)
        folds = fold_indices(cohort, cfg.data.n_folds, cfg.data.seed)
        train_idx, _ = folds[0]
        tr_idx, va_idx = split_train_val(train_idx, 0.25, cfg.data.seed)
        env = make_environment(self.model, cohort, tr_idx, seed=cfg.train.seed + 7)
        val_env = make_environment(self.model, cohort, va_idx, seed=cfg.train.seed + 11)

        buffer = ReplayBuffer()
        for s in range(2):
            buffer.add(behavior_rollout(env, cfg.sim.episode_length, seed=cfg.train.seed + 20 + s))
        val_traj = behavior_rollout(val_env, cfg.sim.episode_length, seed=cfg.train.seed + 99)

        stopper = EarlyStopper(cfg.train.patience, cfg.train.early_stopping)
        result = TrainResult()
        for step in range(cfg.train.iterations):
            batch = buffer.sample(cfg.train.batch_size, cfg.train.seq_len, self.gen)
            report = self._world_step(batch)
            if cfg.train.use_imagination:
                report.update(self._policy_step(batch))
            result.history.append(report)
            if (step + 1) % 10 == 0:
                env_traj = self.collect(env, cfg.train.seq_len)
                buffer.add(env_traj)
            val_loss = self._validation_loss(val_traj)
            if stopper.update(val_loss):
                self.log.info("early stop at step %d", step)
                result.val_loss = val_loss
                break
            result.val_loss = val_loss
        result.final_loss = result.history[-1]["world_loss"] if result.history else 0.0
        return result

    def _validation_loss(self, traj: Trajectory) -> float:
        feats = self._features(traj.action, traj.phase)
        with torch.no_grad():
            state = self.model.world.observe(traj.gene, traj.flow, traj.clinical, feats, self.gen)
            obs = {"gene": traj.gene, "flow": traj.flow, "clinical": traj.clinical}
            terms = self.model.world.per_step_terms(state, obs, traj.reward, traj.continuation)
            loss, _ = vrssm_objective(terms, self.config.model.beta, self.config.model.use_kl)
        return float(loss)
