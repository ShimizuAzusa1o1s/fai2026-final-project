#!/usr/bin/env python3
"""
Entry-point for PPO training of the conservative 6 Nimmt! RL agent.

Usage:
    # Quick smoke test (Minimizer opponents, ~2 min)
    python -m src.players.b12705048.rl.train --total-timesteps 1000 --num-envs 4 --opponent-type minimizer

    # Full training against FlatMCBaseline (~4-6 hours)
    python -m src.players.b12705048.rl.train --total-timesteps 500000 --opponent-type flatmc_baseline

    # Training against Baseline10 (strong curriculum)
    python -m src.players.b12705048.rl.train --total-timesteps 500000 --opponent-type baseline10

Thread-pinning for workstation environments is handled automatically.
"""

# ── Thread-pinning MUST happen before NumPy/Torch imports ────────────────
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import gymnasium as gym

# Ensure project root is importable
sys.path.insert(0, os.getcwd())

from src.players.b12705048.rl.env import SixNimmtEnv
from src.players.b12705048.rl.ppo import PPOTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a conservative PPO agent for 6 Nimmt!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training budget
    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=10,
                    help="Steps per rollout (10 = one complete game)")

    # PPO hyper-parameters
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-start", type=float, default=0.05)
    p.add_argument("--ent-end", type=float, default=0.001)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=40)

    # Opponents
    p.add_argument("--opponent-type", type=str, default="flatmc_baseline",
                    choices=["flatmc_baseline", "baseline10", "minimizer", "mixed"])
    p.add_argument("--opponent-time-limit", type=float, default=0.05,
                    help="Per-action budget for FlatMCBaseline opponents")

    # Reward shaping
    p.add_argument("--reward-alpha", type=float, default=1.0)
    p.add_argument("--reward-gamma", type=float, default=1.5)

    # I/O
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-every", type=int, default=100,
                    help="Save checkpoint every N updates")
    p.add_argument("--run-name", type=str, default=None,
                    help="Name for TensorBoard run (auto-generated if omitted)")
    p.add_argument("--no-tensorboard", action="store_true",
                    help="Disable TensorBoard logging")

    return p.parse_args()


def make_env(
    seed: int,
    opponent_type: str,
    opponent_time_limit: float,
    reward_alpha: float,
    reward_gamma: float,
):
    """Factory closure for gymnasium.vector.SyncVectorEnv."""
    def _init():
        env = SixNimmtEnv(
            opponent_type=opponent_type,
            opponent_time_limit=opponent_time_limit,
            reward_alpha=reward_alpha,
            reward_gamma=reward_gamma,
        )
        env.reset(seed=seed)
        return env
    return _init


def main():
    args = parse_args()

    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Pin torch threads for workstation compatibility
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # Already initialised

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Run naming ───────────────────────────────────────────────────
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = (
            f"ppo_{args.opponent_type}_{args.total_timesteps // 1000}k"
            f"_{timestamp}"
        )

    rl_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
    )
    checkpoint_dir = os.path.join(rl_root, "checkpoints", args.run_name)
    log_dir = "" if args.no_tensorboard else os.path.join(
        rl_root, "runs", args.run_name
    )

    # ── Vectorised environments ──────────────────────────────────────
    print(f"Creating {args.num_envs} environments "
          f"(opponent={args.opponent_type}, "
          f"time_limit={args.opponent_time_limit}s)...")

    envs = gym.vector.SyncVectorEnv([
        make_env(
            seed=args.seed + i,
            opponent_type=args.opponent_type,
            opponent_time_limit=args.opponent_time_limit,
            reward_alpha=args.reward_alpha,
            reward_gamma=args.reward_gamma,
        )
        for i in range(args.num_envs)
    ])

    # ── Trainer ──────────────────────────────────────────────────────
    print(f"Device: {device}")
    print(f"Run name: {args.run_name}")
    print(f"Checkpoints: {checkpoint_dir}")
    if log_dir:
        print(f"TensorBoard: {log_dir}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    batch_size = args.num_envs * args.num_steps
    num_updates = args.total_timesteps // batch_size
    print(f"Batch size: {batch_size} | Num updates: {num_updates}")
    print()

    trainer = PPOTrainer(
        envs,
        device=device,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef_start=args.ent_start,
        ent_coef_end=args.ent_end,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        num_steps=args.num_steps,
        num_envs=args.num_envs,
        minibatch_size=args.minibatch_size,
        total_timesteps=args.total_timesteps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        log_dir=log_dir,
        seed=args.seed,
    )

    final_path = trainer.train()
    envs.close()

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Final model: {final_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
