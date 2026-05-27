"""
Single Deep CFR (SDCFR) Training Script for 6 Nimmt!

Usage::

    # CPU-only (recommended for prototyping):
    python scripts/train_sdcfr.py --device cpu

    # With GPU (respects workstation GPU policy — uses single GPU):
    CUDA_VISIBLE_DEVICES=0 python scripts/train_sdcfr.py --device cuda

    # Quick smoke test:
    python scripts/train_sdcfr.py --iterations 5 --traversals 10 --device cpu

Algorithm:
    For each CFR iteration *t*, for each traversing player *p*:
        1. Deal K random games.
        2. Play through each game round-by-round (External Sampling):
           - At *p*'s decision nodes: evaluate ALL legal actions via rollout.
           - At opponent nodes: sample ONE action from current regret-matched
             strategy.
        3. Compute per-action advantages and store in the advantage buffer.
        4. Train the advantage network on the buffer (weighted MSE, weight = t²).

    The final trained network is saved as ``sdcfr_model.pt``.

GPU Policy Notes:
    - Always use ``CUDA_VISIBLE_DEVICES=<id>`` to pin to a single GPU.
    - GPU context is only active during the training phase (network update),
      NOT during the traversal phase (self-play on CPU).
    - Network is moved to CPU before traversals and back to GPU for training
      to minimise GPU-hours consumed.
    - Buffer files may be large (>2 GB); stored in ``--buffer-dir``
      (default ``/tmp2/b12705048/``).
"""

import os
import sys
import time
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.getcwd())

from src.players.b12705048.core.fast_game import FastGame
from src.players.b12705048.core.features import N_FEATURES
from src.players.b12705048.core.networks import (
    AdvantageNetwork,
    regret_matching_np,
    save_model,
)
from src.players.b12705048.core.reservoir_buffer import ReservoirBuffer


# ── CFR Traversal ──────────────────────────────────────────────────────────


def _sample_action(strategy: np.ndarray, n_valid: int) -> int:
    """Sample an action index from *strategy* restricted to the first
    *n_valid* slots."""
    p = strategy[:n_valid].copy()
    total = p.sum()
    if total > 0:
        p /= total
    else:
        p[:] = 1.0 / n_valid
    return int(np.random.choice(n_valid, p=p))


def _batched_strategies(
    game: FastGame,
    advantage_net: AdvantageNetwork,
) -> tuple[np.ndarray, list[list[int]]]:
    """Compute strategies for all 4 players in one batched forward pass.

    Returns:
        strategies: ``(4, 10)`` array of action probabilities.
        sorted_hands: List of 4 sorted hands.
    """
    batch = np.zeros((4, N_FEATURES), dtype=np.float32)
    sorted_hands: list[list[int]] = []
    masks = np.zeros((4, 10), dtype=np.float32)

    for p in range(4):
        batch[p] = game.get_info_set_features(p)
        sh = sorted(game.hands[p])
        sorted_hands.append(sh)
        masks[p, : len(sh)] = 1.0

    with torch.inference_mode():
        adv_batch = advantage_net(
            torch.from_numpy(batch)
        ).numpy()

    strategies = np.zeros((4, 10), dtype=np.float32)
    for p in range(4):
        strategies[p] = regret_matching_np(adv_batch[p], masks[p])

    return strategies, sorted_hands


def rollout_value(
    game: FastGame,
    player: int,
    advantage_net: AdvantageNetwork,
) -> float:
    """Roll out *game* to completion and return ``-score[player]``.

    Uses batched forward passes (all 4 players in one call) for speed.
    """
    while not game.is_terminal():
        strategies, sorted_hands = _batched_strategies(game, advantage_net)
        actions: dict[int, int] = {}
        for p in range(4):
            n = len(sorted_hands[p])
            if n == 0:
                continue
            idx = _sample_action(strategies[p], n)
            actions[p] = sorted_hands[p][idx]
        if not actions:
            break
        game.resolve_round(actions)

    return -float(game.scores[player])


def cfr_traverse(
    game: FastGame,
    traversing_player: int,
    advantage_net: AdvantageNetwork,
    iteration: int,
    buffer: ReservoirBuffer,
) -> None:
    """External-sampling CFR traversal with round-by-round advantage
    computation and Monte-Carlo rollout evaluation.

    At each round the traversing player's decision node:
      1. Compute strategies for **all** 4 players (batched).
      2. Sample one action per opponent (External Sampling).
      3. Evaluate **every** legal action for the traversing player by
         cloning the game, resolving one round, then rolling out to the end.
      4. Compute per-action advantages and store in *buffer*.
      5. Sample one action for the traversing player and advance the game.
    """
    while not game.is_terminal():
        hand = game.hands[traversing_player]
        if not hand:
            break

        sorted_hand = sorted(hand)
        n_actions = len(sorted_hand)

        # ---- Batched strategy computation for all players ----
        strategies, sorted_hands = _batched_strategies(game, advantage_net)

        # Traversing player's current strategy
        tp_strategy = strategies[traversing_player]

        # ---- Sample opponent actions ----
        opp_actions: dict[int, int] = {}
        for p in range(4):
            if p == traversing_player:
                continue
            n = len(sorted_hands[p])
            if n == 0:
                continue
            idx = _sample_action(strategies[p], n)
            opp_actions[p] = sorted_hands[p][idx]

        # ---- Evaluate each traversing-player action via rollout ----
        action_values = np.zeros(n_actions, dtype=np.float32)

        for a_idx, card in enumerate(sorted_hand):
            game_copy = game.clone()
            all_actions = {traversing_player: card, **opp_actions}
            game_copy.resolve_round(all_actions)
            action_values[a_idx] = rollout_value(
                game_copy, traversing_player, advantage_net,
            )

        # ---- Compute advantages ----
        state_value = float(np.dot(tp_strategy[:n_actions], action_values))
        advantages = np.zeros(10, dtype=np.float32)
        advantages[:n_actions] = action_values - state_value

        # ---- Store features & advantages ----
        features = game.get_info_set_features(traversing_player)
        buffer.add(features, advantages, iteration)

        # ---- Advance game with sampled action ----
        tp_idx = _sample_action(tp_strategy, n_actions)
        chosen_card = sorted_hand[tp_idx]
        all_actions = {traversing_player: chosen_card, **opp_actions}
        game.resolve_round(all_actions)


# ── Network Training ───────────────────────────────────────────────────────


def train_network(
    advantage_net: AdvantageNetwork,
    buffer: ReservoirBuffer,
    optimizer: optim.Optimizer,
    batch_size: int,
    epochs: int,
    device: torch.device,
) -> float:
    """Train *advantage_net* on the buffer for *epochs* epochs.

    Returns the average loss across all batches.
    """
    if buffer.size < min(batch_size, 64):
        return 0.0

    actual_batch = min(batch_size, buffer.size)

    advantage_net.to(device)
    advantage_net.train()
    total_loss = 0.0
    n_batches = 0

    for _ in range(epochs):
        batch = buffer.sample(actual_batch)
        if batch is None:
            continue

        features_np, target_np, iters_np = batch

        # Per-sample weight = iteration² (already biased by sampling, but
        # re-weighting further emphasises recent data).  Normalise so the
        # mean weight ≈ 1 to keep the loss scale stable.
        weights_np = iters_np ** 2
        mean_w = weights_np.mean()
        if mean_w > 0:
            weights_np = weights_np / mean_w

        feat_t = torch.from_numpy(features_np).to(device)
        target_t = torch.from_numpy(target_np).to(device)
        weight_t = torch.from_numpy(weights_np).to(device)

        pred = advantage_net(feat_t)

        # Masked weighted MSE: only penalise valid hand slots.
        # Valid slot count is encoded in feature[14] (hand_size / 10).
        hand_sizes = (feat_t[:, 14] * 10.0).round().long().clamp(1, 10)
        slot_indices = torch.arange(10, device=device).unsqueeze(0)
        mask = (slot_indices < hand_sizes.unsqueeze(1)).float()

        sq_err = (pred - target_t) ** 2 * mask
        loss = (weight_t.unsqueeze(1) * sq_err).sum() / mask.sum().clamp(min=1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    advantage_net.eval()
    # Move back to CPU so traversals don't consume GPU quota
    advantage_net.cpu()
    return total_loss / max(1, n_batches)


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SDCFR Training for 6 Nimmt!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--iterations", type=int, default=500,
                        help="Number of outer CFR iterations.")
    parser.add_argument("--traversals", type=int, default=200,
                        help="Traversals per player per iteration.")
    parser.add_argument("--buffer-size", type=int, default=2_000_000,
                        help="Advantage memory capacity.")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="Training mini-batch size.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs per CFR iteration.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device for training ('cpu' or 'cuda').")
    parser.add_argument("--save-dir", type=str,
                        default="src/players/b12705048/sdcfr",
                        help="Directory for model checkpoints.")
    parser.add_argument("--buffer-dir", type=str,
                        default="/tmp2/b12705048",
                        help="Directory for large buffer files (may exceed 2 GB).")
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="Save checkpoint every N iterations.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint .pt file to resume from.")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device           : {device}")
    print(f"Configuration    : {vars(args)}")

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.buffer_dir, exist_ok=True)

    # ── Initialise ─────────────────────────────────────────────────────
    advantage_net = AdvantageNetwork(input_dim=N_FEATURES)
    advantage_net.eval()  # Start in eval mode (traversal uses inference_mode)
    optimizer = optim.Adam(advantage_net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.iterations,
    )
    buf = ReservoirBuffer(capacity=args.buffer_size, feature_dim=N_FEATURES)
    rng = random.Random(args.seed)

    start_iter = 1

    # ── Resume ─────────────────────────────────────────────────────────
    if args.resume:
        print(f"Resuming from    : {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        advantage_net.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_iter = ckpt["iteration"] + 1

        buf_path = os.path.join(
            args.buffer_dir,
            f"sdcfr_buffer_iter{ckpt['iteration']}.npz",
        )
        if os.path.exists(buf_path):
            buf.load(buf_path)
            print(f"Loaded buffer    : {buf.size:,} entries from {buf_path}")
        advantage_net.eval()

    n_params = sum(p.numel() for p in advantage_net.parameters())
    print(f"Network params   : {n_params:,}")
    print(f"Buffer capacity  : {args.buffer_size:,}")
    print()

    total_start = time.time()

    for iteration in range(start_iter, args.iterations + 1):
        iter_start = time.time()

        # ── Phase 1: Self-Play Traversals (CPU) ────────────────────────
        for traversing_player in range(4):
            for _ in range(args.traversals):
                game = FastGame.deal_random(rng)
                cfr_traverse(
                    game, traversing_player, advantage_net,
                    iteration, buf,
                )

        traverse_time = time.time() - iter_start

        # ── Phase 2: Train Network (optionally GPU) ────────────────────
        train_start = time.time()
        avg_loss = train_network(
            advantage_net, buf, optimizer,
            args.batch_size, args.epochs, device,
        )
        if avg_loss > 0:  # Only step scheduler when training actually happened
            scheduler.step()
        train_time = time.time() - train_start

        # ── Logging ────────────────────────────────────────────────────
        elapsed = time.time() - total_start
        print(
            f"Iter {iteration:4d}/{args.iterations} | "
            f"Buf {buf.size:>8,}/{args.buffer_size:,} | "
            f"Loss {avg_loss:.6f} | "
            f"LR {scheduler.get_last_lr()[0]:.2e} | "
            f"Trav {traverse_time:.1f}s | "
            f"Train {train_time:.1f}s | "
            f"Total {elapsed:.0f}s"
        )

        # ── Checkpointing ─────────────────────────────────────────────
        if iteration % args.checkpoint_every == 0:
            ckpt_path = os.path.join(
                args.save_dir,
                f"sdcfr_model_iter{iteration}.pt",
            )
            torch.save(
                {
                    "iteration": iteration,
                    "model_state_dict": advantage_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                ckpt_path,
            )

            buf_path = os.path.join(
                args.buffer_dir,
                f"sdcfr_buffer_iter{iteration}.npz",
            )
            buf.save(buf_path)

            print(f"  → Checkpoint : {ckpt_path}")
            print(f"  → Buffer     : {buf_path}")

    # ── Final Save ─────────────────────────────────────────────────────
    final_path = os.path.join(args.save_dir, "sdcfr_model.pt")
    save_model(advantage_net, final_path)
    print(f"\nTraining complete! Final model → {final_path}")
    print(f"Total wall-clock : {time.time() - total_start:.0f}s")


if __name__ == "__main__":
    main()
