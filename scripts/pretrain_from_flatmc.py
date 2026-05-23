"""
FlatMCO1 Imitation Pre-Training Script.

This script bootstraps the Deep CFR neural networks (RegretNet and PolicyNet)
by extracting training signal from FlatMCO1's Monte Carlo evaluations.
Running this *before* the first MCCFR iteration ensures the traversal starts
with FlatMCO1-quality opponent models rather than uniform random play,
massively improving the quality of regret signals from iteration 1 onward.

Pipeline Position:
    This is an *optional* Step 0, run once before the iterative loop:
        0. pretrain_from_flatmc.py  ← YOU ARE HERE  (warm-start both networks)
        1. generate_cfr_data.py      (MCCFR traversal with the warm-started RegretNet)
        2. train_cfr.py --regret     (refine RegretNet on counterfactual regrets)
        3. train_cfr.py --policy     (refine PolicyNet on strategy distributions)
        4. Repeat from Step 1

Approach:
    1. Simulate complete 6 Nimmt! games with 4 FlatMCO1 agents playing each
       other (self-play).
    2. At each decision point, intercept FlatMCO1's internal per-candidate
       average penalty estimates (``stats_penalty / stats_visits``).
    3. Derive two training targets from these estimates:
       - **Regret-like target**: ``expected_penalty - penalty[card]`` for each
         card.  Positive values indicate the action was *better* than average
         (i.e., the agent should have played it more).
       - **Policy target**: Softmax(-penalties / temperature) converts raw
         penalty estimates into a soft probability distribution.
    4. Pre-train RegretNet (MSE loss) and PolicyNet (Cross-Entropy loss) on
       these targets.

Estimated Wall Time:
    FlatMCO1 runs at ~0.4s/action in this script (reduced time budget).
    A 10-round, 4-player game has 40 decisions → ~16s/game.
    200 games ≈ ~53 minutes.  1000 games ≈ ~4.4 hours.

Usage:
    python scripts/pretrain_from_flatmc.py --num_games 200
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import argparse
from tqdm import tqdm

# Ensure ./src can be imported
sys.path.append(os.getcwd())
from src.players.b12705048.deep_cfr_net import StateEncoder, RegretNet, PolicyNet, INPUT_DIM, BULLHEADS
from src.players.b12705048.flat_mc_o1 import FlatMCO1


# =============================================================================
# Instrumented FlatMCO1
# =============================================================================

class FlatMCO1Instrumented(FlatMCO1):
    """
    FlatMCO1 subclass that exposes internal per-candidate penalty estimates.

    The standard ``FlatMCO1.action()`` returns only the best card. This
    subclass adds ``action_with_stats()`` which returns the full penalty
    profile for all candidate cards, enabling imitation learning.

    The simulation logic is identical to the parent class but uses a reduced
    time budget (0.4s) since we need to evaluate many decision points across
    many games.
    """

    def action_with_stats(self, hand, history):
        """
        Evaluate all candidate cards and return penalty estimates.

        Args:
            hand (list[int]): Cards currently available to play.
            history (dict): Game state from the engine, containing 'board',
                'scores', 'round', 'history_matrix', 'board_history'.

        Returns:
            tuple: ``(chosen_card, avg_penalties, num_visits)`` where:
                - chosen_card (int): The card FlatMCO1 would play.
                - avg_penalties (np.ndarray): Average simulated penalty for
                  each card in ``hand``, shape ``(len(hand),)``.
                - num_visits (int): Total number of Monte Carlo rollouts.
        """
        start_time = time.perf_counter()

        if isinstance(history, dict):
            board = history.get('board', [])
        else:
            board = history[-1]

        visible_cards = set()
        for row in board:
            visible_cards.update(row)

        if isinstance(history, dict):
            for past_round in history.get('history_matrix', []):
                visible_cards.update(past_round)
            if history.get('board_history'):
                for row in history['board_history'][0]:
                    visible_cards.update(row)

        unseen_cards = list(self.total_cards - visible_cards - set(hand))
        n_turns = len(hand)

        # Trivial case: only one card left, no simulation needed
        if n_turns == 1:
            return hand[0], np.zeros(1), 1

        opp_indices = [i for i in range(4) if i != self.player_idx]

        # Pre-compute board state for vectorized simulation
        orig_row_bullheads = [sum(self.bullhead_lookup_array[c] for c in row) for row in board]
        base_tails = np.array([row[-1] for row in board], dtype=np.int32)
        base_lengths = np.array([len(row) for row in board], dtype=np.int32)
        base_bullheads = np.array(orig_row_bullheads, dtype=np.int32)

        C = len(hand)
        unseen = np.array(unseen_cards, dtype=np.int32)
        n_unseen = len(unseen)

        # For each candidate card c_i, pre-compute the remaining hand after playing c_i
        my_rem = []
        for c in hand:
            rem = [x for x in hand if x != c]
            my_rem.append(rem)
        my_rem = np.array(my_rem, dtype=np.int32)

        hand_array = np.array(hand, dtype=np.int32)[:, None]

        stats_penalty = np.zeros(C, dtype=np.int64)
        stats_visits = 0

        # Batch size: simulate N complete games per iteration
        N = max(100, 15000 // C)

        # Pre-allocate simulation buffers to avoid per-iteration allocation
        tails_buf = np.empty((C, N, 4), dtype=np.int32)
        lengths_buf = np.empty((C, N, 4), dtype=np.int32)
        bullheads_buf = np.empty((C, N, 4), dtype=np.int32)
        penalties_buf = np.empty((C, N, 4), dtype=np.int32)
        plays_buf = np.empty((C, N, 4), dtype=np.int32)
        diff_buf = np.empty((C, N, 4), dtype=np.int32)
        score_buf = np.empty((C, N, 4), dtype=np.int32)

        base_tails_b = np.broadcast_to(base_tails, (C, N, 4))
        base_lengths_b = np.broadcast_to(base_lengths, (C, N, 4))
        base_bullheads_b = np.broadcast_to(base_bullheads, (C, N, 4))

        my_rem_expanded = np.expand_dims(my_rem, axis=1)
        my_rem_broadcasted = np.broadcast_to(my_rem_expanded, (C, N, n_turns - 1))

        # Reduced time budget for data generation (parent default is 0.95s)
        time_budget = min(self.time_limit, 0.4)

        # ---- Vectorized Monte Carlo Simulation ----
        # This loop is identical to FlatMCO1.action() but returns the full
        # penalty profile instead of just argmin.
        while time.perf_counter() - start_time < time_budget - 0.05:
            tails_buf[:] = base_tails_b
            lengths_buf[:] = base_lengths_b
            bullheads_buf[:] = base_bullheads_b
            penalties_buf.fill(0)

            # Sample opponent hands from the unseen pool
            noise = self.rng.random((N, n_unseen))
            req_cards = 3 * n_turns
            if req_cards < n_unseen:
                perm_indices = np.argpartition(noise, req_cards - 1, axis=1)[:, :req_cards]
            else:
                perm_indices = noise.argsort(axis=1)

            opp0_cards = unseen[perm_indices[:, 0:n_turns]]
            opp1_cards = unseen[perm_indices[:, n_turns:2*n_turns]]
            opp2_cards = unseen[perm_indices[:, 2*n_turns:3*n_turns]]

            # Shuffle our remaining cards for rounds 2+
            my_noise = self.rng.random((C, N, n_turns - 1))
            my_perm = my_noise.argsort(axis=2)
            my_cards = np.take_along_axis(my_rem_broadcasted, my_perm, axis=2)

            # Simulate each round of the game
            for t in range(n_turns):
                if t == 0:
                    plays_buf[:, :, self.player_idx] = np.broadcast_to(hand_array, (C, N))
                else:
                    plays_buf[:, :, self.player_idx] = my_cards[:, :, t - 1]

                plays_buf[:, :, opp_indices[0]] = np.broadcast_to(opp0_cards[:, t], (C, N))
                plays_buf[:, :, opp_indices[1]] = np.broadcast_to(opp1_cards[:, t], (C, N))
                plays_buf[:, :, opp_indices[2]] = np.broadcast_to(opp2_cards[:, t], (C, N))

                # Resolve trick: sort by card value, place each card
                order = np.argsort(plays_buf, axis=2)
                sorted_plays = np.take_along_axis(plays_buf, order, axis=2)

                for i in range(4):
                    c = sorted_plays[:, :, i]
                    p = order[:, :, i]

                    c_exp = c[:, :, None]
                    np.subtract(c_exp, tails_buf, out=diff_buf)
                    diff_buf[diff_buf <= 0] = 1000

                    target_row = np.argmin(diff_buf, axis=2)
                    min_diff = np.min(diff_buf, axis=2)
                    invalid_placement = min_diff == 1000

                    if np.any(invalid_placement):
                        # Low Card Rule: pick cheapest row
                        np.multiply(bullheads_buf, 1000, out=score_buf)
                        score_buf += lengths_buf * 10
                        score_buf += np.arange(4, dtype=np.int32)

                        alt_target_row = np.argmin(score_buf, axis=2)
                        final_target_row = np.where(invalid_placement, alt_target_row, target_row)
                    else:
                        final_target_row = target_row

                    final_idx = final_target_row[:, :, None]
                    row_len = np.take_along_axis(lengths_buf, final_idx, axis=2).squeeze(-1)
                    take_row = (row_len == 5) | invalid_placement

                    row_bh = np.take_along_axis(bullheads_buf, final_idx, axis=2).squeeze(-1)
                    penalty_to_add = np.where(take_row, row_bh, 0)

                    # Accumulate penalties per player
                    p_exp = p[:, :, None]
                    curr_pen = np.take_along_axis(penalties_buf, p_exp, axis=2)
                    np.put_along_axis(penalties_buf, p_exp, curr_pen + penalty_to_add[:, :, None], axis=2)

                    # Update board state
                    c_bh = self.bullhead_lookup_array[c]
                    np.put_along_axis(tails_buf, final_idx, c_exp, axis=2)

                    new_len = np.where(take_row, 1, row_len + 1)
                    new_bh_val = np.where(take_row, c_bh, row_bh + c_bh)

                    np.put_along_axis(lengths_buf, final_idx, new_len[:, :, None], axis=2)
                    np.put_along_axis(bullheads_buf, final_idx, new_bh_val[:, :, None], axis=2)

            stats_penalty += penalties_buf[:, :, self.player_idx].sum(axis=1)
            stats_visits += N

        avg_penalties = stats_penalty / max(1, stats_visits)
        best_idx = np.argmin(avg_penalties)
        return hand[best_idx], avg_penalties, stats_visits


# =============================================================================
# Pre-Training Dataset
# =============================================================================

class PretrainDataset(Dataset):
    """
    PyTorch Dataset wrapping (state, regret, strategy, mask) quadruples
    extracted from FlatMCO1 self-play games.
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, regret, policy, mask = self.data[idx]
        return state, regret, policy, mask


# =============================================================================
# Data Generation via FlatMCO1 Self-Play
# =============================================================================

def generate_pretrain_data(num_games):
    """
    Run FlatMCO1 self-play games and extract training data.

    For each decision point in each game, we:
      1. Encode the game state with the full 151-dim StateEncoder.
      2. Extract FlatMCO1's per-candidate penalty estimates.
      3. Derive regret-like and policy targets from the penalty profile.

    Args:
        num_games (int): Number of complete games to simulate.

    Returns:
        list[tuple]: Training quadruples of
            ``(state_tensor, regrets, strategy, legal_mask)``.
    """
    from scripts.generate_cfr_data import resolve_trick

    print(f"Generating {num_games} games of pre-training data via FlatMCO1 self-play...")
    agents = [FlatMCO1Instrumented(i) for i in range(4)]

    training_data = []

    for _ in tqdm(range(num_games)):
        # Deal a fresh game
        deck = list(range(1, 105))
        random.shuffle(deck)

        board = [[deck.pop()] for _ in range(4)]
        row_bullheads = [BULLHEADS[r[0]] for r in board]
        hands = {i: [deck.pop() for _ in range(10)] for i in range(4)}

        # Snapshot the initial board for FlatMCO1's unseen card computation.
        # Must be a deep copy since `board` is mutated by resolve_trick.
        initial_board = [row[:] for row in board]

        scores = [0.0, 0.0, 0.0, 0.0]
        history_matrix = []
        played_cards = set()

        for round_idx in range(10):
            # ---- Phase 1: All agents evaluate simultaneously ----
            actions_and_stats = []
            for i in range(4):
                history_dict = {
                    'board': board,
                    'scores': list(scores),  # Copy to avoid mutation issues
                    'round': round_idx,
                    'history_matrix': history_matrix,
                    'board_history': [initial_board]
                }

                chosen, avg_pens, visits = agents[i].action_with_stats(
                    hands[i], history_dict
                )
                actions_and_stats.append((i, chosen, avg_pens, visits))

            # ---- Phase 2: Convert evaluations into training data ----
            for i in range(4):
                hand = hands[i]
                _, chosen, avg_pens, visits = actions_and_stats[i]

                state_tensor = StateEncoder.encode(
                    hand, board, round_num=round_idx, played_cards=played_cards,
                    scores=scores, history_matrix=history_matrix, player_idx=i
                )
                legal_mask = StateEncoder.get_legal_mask(hand)

                if len(hand) > 1:
                    regrets = torch.zeros(104)
                    strategy = torch.zeros(104)

                    # Convert penalty estimates to a soft probability distribution.
                    # Lower penalties → higher probability.  Temperature controls
                    # how peaked the distribution is (2.0 = moderately soft).
                    temp = 2.0
                    exp_vals = np.exp(-avg_pens / temp)
                    probs = exp_vals / np.sum(exp_vals)

                    # Compute regret-like targets using the CFR convention:
                    #   regret(a) = expected_value - value(a)
                    # where value = penalty (lower is better), so:
                    #   regret(a) = E[penalty] - penalty(a)
                    # Positive regret → action was *better* than average.
                    expected_penalty = np.sum(probs * avg_pens)

                    for idx, c in enumerate(hand):
                        regrets[c - 1] = expected_penalty - avg_pens[idx]
                        strategy[c - 1] = probs[idx]
                else:
                    # Trivial: only one card, deterministic strategy
                    regrets = torch.zeros(104)
                    strategy = torch.zeros(104)
                    strategy[hand[0] - 1] = 1.0

                training_data.append((state_tensor, regrets, strategy, legal_mask))

            # ---- Phase 3: Resolve the trick and update game state ----
            trick_actions = [(actions_and_stats[i][1], i) for i in range(4)]
            resolve_trick(board, row_bullheads, trick_actions, scores, played_cards)

            # Record history and remove played cards from hands
            round_record = [0] * 4
            for card, p_idx in trick_actions:
                round_record[p_idx] = card
                hands[p_idx].remove(card)
            history_matrix.append(round_record)

    return training_data


# =============================================================================
# Network Pre-Training
# =============================================================================

def train_networks_on_flatmc(data, epochs=5, batch_size=64, lr=1e-3):
    """
    Pre-train RegretNet and PolicyNet on FlatMCO1-derived targets.

    Args:
        data (list[tuple]): Training quadruples from ``generate_pretrain_data``.
        epochs (int): Number of training epochs.
        batch_size (int): Mini-batch size for the DataLoader.
        lr (float): Learning rate for the Adam optimizer.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nPre-training on {len(data)} states using device: {device}")

    dataset = PretrainDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    regret_net = RegretNet().to(device)
    policy_net = PolicyNet().to(device)

    regret_opt = optim.Adam(regret_net.parameters(), lr=lr)
    policy_opt = optim.Adam(policy_net.parameters(), lr=lr)

    mse_loss = nn.MSELoss()

    for epoch in range(epochs):
        r_loss_sum = 0.0
        p_loss_sum = 0.0

        for states, regrets, policies, masks in dataloader:
            states = states.to(device)
            regrets = regrets.to(device)
            policies = policies.to(device)
            masks = masks.bool().to(device)

            # ---- Train RegretNet (MSE on legal actions) ----
            regret_opt.zero_grad()
            r_preds = regret_net(states)
            r_loss = mse_loss(r_preds[masks], regrets[masks])
            r_loss.backward()
            regret_opt.step()
            r_loss_sum += r_loss.item()

            # ---- Train PolicyNet (Cross-Entropy on strategy targets) ----
            policy_opt.zero_grad()
            p_logits = policy_net(states, masks)
            log_probs = F.log_softmax(p_logits, dim=-1)
            p_loss = -torch.sum(policies[masks] * log_probs[masks]) / states.size(0)
            p_loss.backward()
            policy_opt.step()
            p_loss_sum += p_loss.item()

        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Regret Loss: {r_loss_sum/len(dataloader):.6f} | "
              f"Policy Loss: {p_loss_sum/len(dataloader):.6f}")

    # Save pre-trained weights
    os.makedirs("src/players/b12705048/weights", exist_ok=True)
    torch.save(regret_net.state_dict(), "src/players/b12705048/weights/regret_net.pt")
    torch.save(policy_net.state_dict(), "src/players/b12705048/weights/policy_net.pt")
    print("Pre-trained weights saved.")


# =============================================================================
# Main Entry Point
# =============================================================================

def generate_and_train(num_games=100):
    """Run the complete pre-training pipeline: generate data then train."""
    data = generate_pretrain_data(num_games)
    train_networks_on_flatmc(data, epochs=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Pre-train Deep CFR networks via FlatMCO1 imitation learning."
    )
    parser.add_argument("--num_games", type=int, default=100,
                        help="Number of FlatMCO1 self-play games to simulate")
    args = parser.parse_args()
    generate_and_train(args.num_games)
