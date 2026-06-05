"""
CleanRL-style PPO Training Loop for 6 Nimmt! RL Agent.

Implements Proximal Policy Optimization with:
    - Generalized Advantage Estimation (GAE)
    - Clipped surrogate objective
    - Linear entropy coefficient annealing
    - Action masking throughout
    - TensorBoard logging and periodic checkpointing

The loop is designed for a fixed episode length of 10 steps (one complete
6 Nimmt! game), ensuring clean episode boundaries with no bootstrapping
artifacts from truncated episodes.
"""

import os
import time as _time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.players.b12705048.rl.network import ActorCritic


class PPOTrainer:
    """
    Self-contained PPO trainer that rolls out vectorised environments,
    computes GAE advantages, and updates the Actor-Critic network.

    All hyper-parameters are set at construction time and remain fixed
    (except the entropy coefficient, which decays linearly).
    """

    def __init__(
        self,
        envs,
        *,
        device: torch.device | str = "cpu",
        # ── PPO hyper-parameters ─────────────────────────────────────
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef_start: float = 0.05,
        ent_coef_end: float = 0.001,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        # ── Rollout dimensions ───────────────────────────────────────
        num_steps: int = 10,
        num_envs: int = 16,
        minibatch_size: int = 40,
        # ── Training budget ──────────────────────────────────────────
        total_timesteps: int = 500_000,
        # ── I/O ──────────────────────────────────────────────────────
        checkpoint_dir: str = "",
        checkpoint_every: int = 100,
        log_dir: str = "",
        seed: int = 42,
    ):
        self.envs = envs
        self.device = torch.device(device)

        # Hyper-parameters
        self.lr = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef_start = ent_coef_start
        self.ent_coef_end = ent_coef_end
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs

        self.num_steps = num_steps
        self.num_envs = num_envs
        self.batch_size = num_steps * num_envs
        self.minibatch_size = minibatch_size
        self.total_timesteps = total_timesteps
        self.num_updates = total_timesteps // self.batch_size

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every
        self.log_dir = log_dir
        self.seed = seed

        # Network & optimizer
        self.agent = ActorCritic().to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.lr, eps=1e-5)

        # TensorBoard writer (lazy init)
        self._writer = None

    # ── Public API ───────────────────────────────────────────────────────

    def train(self) -> str:
        """
        Run the full training loop.

        Returns:
            Path to the final checkpoint.
        """
        self._setup_logging()
        self._setup_checkpoints()

        # ── Rollout buffers (CPU) ────────────────────────────────────
        obs_buf = torch.zeros(
            (self.num_steps, self.num_envs, 232), dtype=torch.float32
        )
        act_buf = torch.zeros(
            (self.num_steps, self.num_envs), dtype=torch.long
        )
        logprob_buf = torch.zeros(
            (self.num_steps, self.num_envs), dtype=torch.float32
        )
        reward_buf = torch.zeros(
            (self.num_steps, self.num_envs), dtype=torch.float32
        )
        done_buf = torch.zeros(
            (self.num_steps, self.num_envs), dtype=torch.float32
        )
        value_buf = torch.zeros(
            (self.num_steps, self.num_envs), dtype=torch.float32
        )
        mask_buf = torch.zeros(
            (self.num_steps, self.num_envs, 104), dtype=torch.float32
        )

        # ── Initialise environments ──────────────────────────────────
        next_obs_np, infos = self.envs.reset()
        next_obs = torch.tensor(next_obs_np, dtype=torch.float32)
        next_done = torch.zeros(self.num_envs, dtype=torch.float32)
        next_mask = torch.tensor(
            np.array(infos["action_mask"]), dtype=torch.float32
        )

        global_step = 0
        start_time = _time.time()

        # Accumulators for smoothed logging
        ep_rewards = []
        ep_penalties = []
        ep_ranks = []
        ep_catastrophes = []

        for update in range(1, self.num_updates + 1):
            # ── Entropy annealing (linear) ───────────────────────────
            frac = 1.0 - (update - 1.0) / max(1, self.num_updates)
            ent_coef = self.ent_coef_end + frac * (
                self.ent_coef_start - self.ent_coef_end
            )

            # ── Rollout collection ───────────────────────────────────
            for step in range(self.num_steps):
                global_step += self.num_envs

                obs_buf[step] = next_obs
                done_buf[step] = next_done
                mask_buf[step] = next_mask

                with torch.no_grad():
                    obs_dev = next_obs.to(self.device)
                    mask_dev = next_mask.to(self.device)
                    action, logprob, _, value = self.agent.get_action_and_value(
                        obs_dev, mask_dev
                    )
                    value = value.flatten()

                act_buf[step] = action.cpu()
                logprob_buf[step] = logprob.cpu()
                value_buf[step] = value.cpu()

                # Step all environments
                next_obs_np, rewards, terminations, truncations, infos = (
                    self.envs.step(action.cpu().numpy())
                )

                next_obs = torch.tensor(next_obs_np, dtype=torch.float32)
                reward_buf[step] = torch.tensor(rewards, dtype=torch.float32)
                next_done = torch.tensor(
                    np.logical_or(terminations, truncations),
                    dtype=torch.float32,
                )

                # Update action mask
                if "action_mask" in infos:
                    next_mask = torch.tensor(
                        np.array(infos["action_mask"]), dtype=torch.float32
                    )

                # Collect episode stats from terminated envs
                self._collect_episode_stats(
                    infos, ep_rewards, ep_penalties, ep_ranks, ep_catastrophes
                )

            # ── GAE advantage estimation ─────────────────────────────
            with torch.no_grad():
                next_value = (
                    self.agent.get_value(next_obs.to(self.device))
                    .cpu()
                    .reshape(1, -1)
                )
                advantages = torch.zeros_like(reward_buf)
                lastgaelam = 0.0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - done_buf[t + 1]
                        nextvalues = value_buf[t + 1]
                    delta = (
                        reward_buf[t]
                        + self.gamma * nextvalues * nextnonterminal
                        - value_buf[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + self.gamma
                        * self.gae_lambda
                        * nextnonterminal
                        * lastgaelam
                    )
                returns = advantages + value_buf

            # ── Flatten rollout ──────────────────────────────────────
            b_obs = obs_buf.reshape(-1, 232)
            b_logprobs = logprob_buf.reshape(-1)
            b_actions = act_buf.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = value_buf.reshape(-1)
            b_masks = mask_buf.reshape(-1, 104)

            # ── PPO update ───────────────────────────────────────────
            b_inds = np.arange(self.batch_size)
            clipfracs = []

            for _epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    (_, newlogprob, entropy, newvalue) = (
                        self.agent.get_action_and_value(
                            b_obs[mb_inds].to(self.device),
                            b_masks[mb_inds].to(self.device),
                            b_actions[mb_inds].to(self.device),
                        )
                    )
                    newvalue = newvalue.flatten()

                    logratio = newlogprob - b_logprobs[mb_inds].to(self.device)
                    ratio = logratio.exp()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs.append(
                            ((ratio - 1.0).abs() > self.clip_coef)
                            .float()
                            .mean()
                            .item()
                        )

                    # Advantage normalisation (per-minibatch)
                    mb_adv = b_advantages[mb_inds].to(self.device)
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    # Clipped surrogate policy loss
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(
                        ratio, 1 - self.clip_coef, 1 + self.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss (unclipped MSE)
                    v_loss = 0.5 * (
                        (newvalue - b_returns[mb_inds].to(self.device)) ** 2
                    ).mean()

                    # Entropy bonus
                    entropy_loss = entropy.mean()

                    # Combined loss
                    loss = (
                        pg_loss
                        - ent_coef * entropy_loss
                        + self.vf_coef * v_loss
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.agent.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

            # ── Logging ──────────────────────────────────────────────
            if self._writer is not None:
                self._log_update(
                    update,
                    global_step,
                    pg_loss.item(),
                    v_loss.item(),
                    entropy_loss.item(),
                    approx_kl.item(),
                    np.mean(clipfracs),
                    ent_coef,
                    ep_rewards,
                    ep_penalties,
                    ep_ranks,
                    ep_catastrophes,
                    start_time,
                )
                ep_rewards.clear()
                ep_penalties.clear()
                ep_ranks.clear()
                ep_catastrophes.clear()

            # ── Checkpointing ────────────────────────────────────────
            if update % self.checkpoint_every == 0:
                self._save_checkpoint(update)

            # ── Progress ─────────────────────────────────────────────
            if update % 10 == 0 or update == 1:
                elapsed = _time.time() - start_time
                sps = global_step / max(1, elapsed)
                print(
                    f"[Update {update}/{self.num_updates}] "
                    f"step={global_step} | SPS={sps:.0f} | "
                    f"pg_loss={pg_loss.item():.4f} | "
                    f"v_loss={v_loss.item():.4f} | "
                    f"ent={entropy_loss.item():.4f} | "
                    f"ent_coef={ent_coef:.4f}"
                )

        # Final checkpoint
        final_path = self._save_checkpoint("final")
        if self._writer is not None:
            self._writer.close()
        return final_path

    # ── Internal helpers ─────────────────────────────────────────────────

    def _collect_episode_stats(self, infos, ep_r, ep_p, ep_rank, ep_cat):
        """Extract episode metrics from vectorised info dicts.

        Gymnasium 1.x auto-vectorises sub-env info dicts:
          infos['episode']   → dict of stacked arrays  (r, l, penalty, …)
          infos['_episode']  → bool array marking which envs have episode data
        """
        # ── Gymnasium 1.x format (auto-vectorised) ───────────────────
        if "_episode" in infos:
            done_mask = infos["_episode"]
            ep_dict = infos["episode"]
            for i, done in enumerate(done_mask):
                if not done:
                    continue
                ep_r.append(float(ep_dict["r"][i]))
                ep_p.append(float(ep_dict["penalty"][i]))
                ep_rank.append(float(ep_dict["rank"][i]))
                ep_cat.append(float(ep_dict["catastrophe_rate"][i]))
            return

        # ── Fallback: older gymnasium format ─────────────────────────
        if "_final_info" in infos:
            for i, done_flag in enumerate(infos["_final_info"]):
                if not done_flag:
                    continue
                fi = infos["final_info"][i]
                if fi is None or "episode" not in fi:
                    continue
                ep = fi["episode"]
                ep_r.append(ep["r"])
                ep_p.append(ep["penalty"])
                ep_rank.append(ep["rank"])
                ep_cat.append(ep["catastrophe_rate"])

    def _setup_logging(self):
        if not self.log_dir:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.log_dir, exist_ok=True)
            self._writer = SummaryWriter(self.log_dir)
        except ImportError:
            print("[PPO] tensorboard not found — logging disabled")

    def _setup_checkpoints(self):
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _save_checkpoint(self, tag) -> str:
        if not self.checkpoint_dir:
            return ""
        path = os.path.join(self.checkpoint_dir, f"ppo_{tag}.pt")
        torch.save(self.agent.state_dict(), path)
        print(f"[Checkpoint] Saved → {path}")
        return path

    def _log_update(
        self,
        update,
        global_step,
        pg_loss,
        v_loss,
        entropy_loss,
        approx_kl,
        clipfrac,
        ent_coef,
        ep_rewards,
        ep_penalties,
        ep_ranks,
        ep_catastrophes,
        start_time,
    ):
        w = self._writer
        w.add_scalar("losses/policy_loss", pg_loss, global_step)
        w.add_scalar("losses/value_loss", v_loss, global_step)
        w.add_scalar("losses/entropy", entropy_loss, global_step)
        w.add_scalar("losses/approx_kl", approx_kl, global_step)
        w.add_scalar("losses/clipfrac", clipfrac, global_step)
        w.add_scalar("charts/entropy_coef", ent_coef, global_step)
        w.add_scalar(
            "charts/SPS",
            global_step / max(1, _time.time() - start_time),
            global_step,
        )

        if ep_rewards:
            w.add_scalar(
                "charts/episode_reward", np.mean(ep_rewards), global_step
            )
            w.add_scalar(
                "charts/episode_penalty", np.mean(ep_penalties), global_step
            )
            w.add_scalar(
                "charts/episode_rank", np.mean(ep_ranks), global_step
            )
            w.add_scalar(
                "charts/catastrophe_rate",
                np.mean(ep_catastrophes),
                global_step,
            )
