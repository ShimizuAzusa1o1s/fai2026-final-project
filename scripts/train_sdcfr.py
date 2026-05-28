"""
Deep CFR Training Script for 6 Nimmt!

Algorithm:
    External Sampling Monte Carlo Counterfactual Regret Minimization (MCCFR)
    adapted for a 4-player, non-zero-sum, simultaneous-action game.

    For each CFR iteration *t*, for each traversing player *p*:
        1. Deal K random games.
        2. Play through each game round-by-round (External Sampling):
           - At *p*'s decision nodes: evaluate ALL legal actions via rollout.
           - At opponent nodes: sample ONE action from current regret-matched strategy.
        3. Compute per-action advantages and store in the Advantage Buffer.
        4. Store current strategies in the Strategy Buffer.
        5. Train the Advantage Network on the Advantage Buffer.
        6. Train the Strategy Network on the Strategy Buffer.

    The final trained Strategy Network is saved as ``strategy_model.pt``.

Characteristics:
    - **Reward Reshaping**: Rank-based payouts are used to force a zero-sum environment.
    - **Rollout Policy**: Regret-matched actions from the current Advantage Network.
    - **Performance**: GPU accelerated during network updates, CPU during self-play.

See Also:
    ``sdcfr_player.py`` — Deploys the trained Average Strategy Network.
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
    StrategyNetwork,
    regret_matching_np,
    save_model,
)
from src.players.b12705048.core.reservoir_buffer import ReservoirBuffer


# ---- Phase 1: Utility Functions ───────────────────────────────────────────

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
    """Roll out *game* to completion and return rank-based zero-sum payout.

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

    # ---- Phase 2: Rank-Based Payouts ----
    scores = game.scores
    payouts_table = [1.5, 0.5, -0.5, -1.5]
    player_score = scores[player]
    
    # Calculate rank based on penalty points (fewer is better)
    better_count = sum(1 for s in scores if s < player_score)
    tied_count = sum(1 for s in scores if s == player_score)
    
    avg_payout = sum(payouts_table[better_count : better_count + tied_count]) / tied_count
    return float(avg_payout)


# ---- Phase 3: CFR Traversal ───────────────────────────────────────────────

def cfr_traverse(
    game: FastGame,
    traversing_player: int,
    advantage_net: AdvantageNetwork,
    iteration: int,
    adv_buffer: ReservoirBuffer,
    strat_buffer: ReservoirBuffer,
) -> None:
    """External-sampling CFR traversal with round-by-round advantage
    computation and Monte-Carlo rollout evaluation.
    """
    while not game.is_terminal():
        hand = game.hands[traversing_player]
        if not hand:
            break

        sorted_hand = sorted(hand)
        n_actions = len(sorted_hand)

        # Compute strategies for all 4 players (batched)
        strategies, sorted_hands = _batched_strategies(game, advantage_net)
        tp_strategy = strategies[traversing_player]

        # Store the current strategy into the Strategy Buffer
        features = game.get_info_set_features(traversing_player)
        strat_buffer.add(features, tp_strategy, iteration)

        # Sample opponent actions (External Sampling)
        opp_actions: dict[int, int] = {}
        for p in range(4):
            if p == traversing_player:
                continue
            n = len(sorted_hands[p])
            if n == 0:
                continue
            idx = _sample_action(strategies[p], n)
            opp_actions[p] = sorted_hands[p][idx]

        # Evaluate each traversing-player action against joint opponents' actions
        action_values = np.zeros(n_actions, dtype=np.float32)

        for a_idx, card in enumerate(sorted_hand):
            game_copy = game.clone()
            all_actions = {traversing_player: card, **opp_actions}
            game_copy.resolve_round(all_actions)
            action_values[a_idx] = rollout_value(
                game_copy, traversing_player, advantage_net,
            )

        # Compute advantages and store in Advantage Buffer
        state_value = float(np.dot(tp_strategy[:n_actions], action_values))
        advantages = np.zeros(10, dtype=np.float32)
        advantages[:n_actions] = action_values - state_value
        
        adv_buffer.add(features, advantages, iteration)

        # Advance game with sampled action
        tp_idx = _sample_action(tp_strategy, n_actions)
        chosen_card = sorted_hand[tp_idx]
        all_actions = {traversing_player: chosen_card, **opp_actions}
        game.resolve_round(all_actions)


# ---- Phase 4: Network Training ────────────────────────────────────────────

def train_network(
    net: nn.Module,
    buffer: ReservoirBuffer,
    optimizer: optim.Optimizer,
    batch_size: int,
    epochs: int,
    device: torch.device,
) -> float:
    """Train the network (*net*) on the buffer for *epochs* epochs.

    Returns the average MSE loss across all batches.
    """
    if buffer.size < min(batch_size, 64):
        return 0.0

    actual_batch = min(batch_size, buffer.size)

    net.to(device)
    net.train()
    total_loss = 0.0
    n_batches = 0

    for _ in range(epochs):
        batch = buffer.sample(actual_batch)
        if batch is None:
            continue

        features_np, target_np, iters_np = batch

        # Weight loss by iteration (t) instead of t^2 to balance stability.
        # Since buffer uniformly samples, weighting loss directly achieves proportional updates.
        weights_np = iters_np.copy()
        mean_w = weights_np.mean()
        if mean_w > 0:
            weights_np = weights_np / mean_w

        feat_t = torch.from_numpy(features_np).to(device)
        target_t = torch.from_numpy(target_np).to(device)
        weight_t = torch.from_numpy(weights_np).to(device)

        pred = net(feat_t)

        # Masked weighted MSE: only penalise valid hand slots.
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

    net.eval()
    net.cpu()
    return total_loss / max(1, n_batches)


# ---- Phase 5: Main Loop ───────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deep CFR Training for 6 Nimmt!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--iterations", type=int, default=500,
                        help="Number of outer CFR iterations.")
    parser.add_argument("--traversals", type=int, default=200,
                        help="Traversals per player per iteration.")
    parser.add_argument("--buffer-size", type=int, default=2_000_000,
                        help="Memory capacity for both buffers.")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="Training mini-batch size.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs per CFR iteration.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device for training ('cpu' or 'cuda').")
    parser.add_argument("--save-dir", type=str,
                        default="src/players/b12705048/agents",
                        help="Directory for model checkpoints.")
    parser.add_argument("--buffer-dir", type=str,
                        default="/tmp2/b12705048",
                        help="Directory for large buffer files.")
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="Save checkpoint every N iterations.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint .pt file to resume from.")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device           : {device}")
    print(f"Configuration    : {vars(args)}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.buffer_dir, exist_ok=True)

    advantage_net = AdvantageNetwork(input_dim=N_FEATURES)
    advantage_net.eval()
    adv_optimizer = optim.Adam(advantage_net.parameters(), lr=args.lr)
    adv_scheduler = optim.lr_scheduler.CosineAnnealingLR(adv_optimizer, T_max=args.iterations)

    strategy_net = StrategyNetwork(input_dim=N_FEATURES)
    strategy_net.eval()
    strat_optimizer = optim.Adam(strategy_net.parameters(), lr=args.lr)
    strat_scheduler = optim.lr_scheduler.CosineAnnealingLR(strat_optimizer, T_max=args.iterations)

    adv_buf = ReservoirBuffer(capacity=args.buffer_size, feature_dim=N_FEATURES)
    strat_buf = ReservoirBuffer(capacity=args.buffer_size, feature_dim=N_FEATURES)
    rng = random.Random(args.seed)

    start_iter = 1

    n_params = sum(p.numel() for p in advantage_net.parameters())
    print(f"Network params   : {n_params:,} (x2)")
    print(f"Buffer capacity  : {args.buffer_size:,} (x2)")
    print()

    total_start = time.time()

    for iteration in range(start_iter, args.iterations + 1):
        iter_start = time.time()

        for traversing_player in range(4):
            for _ in range(args.traversals):
                game = FastGame.deal_random(rng)
                cfr_traverse(
                    game, traversing_player, advantage_net,
                    iteration, adv_buf, strat_buf,
                )

        traverse_time = time.time() - iter_start

        train_start = time.time()
        
        # Train Advantage Network
        adv_loss = train_network(
            advantage_net, adv_buf, adv_optimizer,
            args.batch_size, args.epochs, device,
        )
        if adv_loss > 0:
            adv_scheduler.step()
            
        # Train Strategy Network
        strat_loss = train_network(
            strategy_net, strat_buf, strat_optimizer,
            args.batch_size, args.epochs, device,
        )
        if strat_loss > 0:
            strat_scheduler.step()
            
        train_time = time.time() - train_start

        elapsed = time.time() - total_start
        print(
            f"Iter {iteration:4d}/{args.iterations} | "
            f"Buf {adv_buf.size:>8,} | "
            f"AdvLoss {adv_loss:.5f} | StratLoss {strat_loss:.5f} | "
            f"Trav {traverse_time:.1f}s | Train {train_time:.1f}s | "
            f"Total {elapsed:.0f}s"
        )

        if iteration % args.checkpoint_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"dcfr_model_iter{iteration}.pt")
            torch.save({
                "iteration": iteration,
                "adv_state_dict": advantage_net.state_dict(),
                "strat_state_dict": strategy_net.state_dict(),
            }, ckpt_path)

            print(f"  → Checkpoint : {ckpt_path}")

    final_path = os.path.join(args.save_dir, "dcfr_strategy_model.pt")
    save_model(strategy_net, final_path)
    print(f"\nTraining complete! Final Average Strategy model → {final_path}")
    print(f"Total wall-clock : {time.time() - total_start:.0f}s")


if __name__ == "__main__":
    main()
