"""
Successive Halving Monte Carlo (1-Ply) Player Module — Neural Determinization Variant.

This module implements the FlatMCNeural agent, which uses the PyTorch
TopologicalOpponentNet to accurately reconstruct opponent hand distributions 
from their behavioral penalty history.

Algorithm:
    1. Parse board state and history into 125-dimensional feature vectors.
    2. Pass features to TopologicalOpponentNet to infer a 3x5 probability distribution
       over the 5 topological gaps for each opponent.
    3. Translate gap probabilities into individual card log-weights.
    4. Allocate simulation budget using Successive Halving (log2(N) stages).
    5. During batch rollout initialization, use a sequential batched Gumbel-Max 
       trick to sample hands without replacement using the neural weights.
    6. Continue with MinMax / Softmax mixed rollout policy to pick specific cards.
    7. Select the final candidate with the lowest average penalty.

Characteristics:
    - O(1) batched SIMD simulation per rollout.
    - Neural determinization accurately resolving hidden state sparsity.
    - Successive Halving for uniform elimination budget allocation.
    - Softmax-based exploration (epsilon portion).

See Also:
    - `flatmc_sh.py`
    - `flatmc_dirichlet.py`
"""

import time
import math
import numpy as np
import torch
import os
import sys

from src.players.b12705048.core.constants import BULLHEAD_LOOKUP
from src.models.opponent_model import TopologicalOpponentNet
from src.models.feature_extractor import (
    build_feature_vector, 
    get_gap_capacities, 
    get_topological_gaps, 
    assign_card_to_bucket
)

class FlatMCNeural:
    """
    Vectorized 1-ply Monte Carlo agent using Successive Halving for budget allocation
    and a Neural Network for opponent hand determinization.

    Attributes:
        player_idx (int): This agent's seat index (0-3).
        time_limit (float): Simulation budget in seconds.
        epsilon (float): Ratio of random rollouts (0.0 to 1.0) mixed into min-max policy.
        tau (float): Temperature parameter for the Softmax rollout distribution.
        total_cards (set[int]): The full card universe {1, ..., 104}.
        batch_size (int): Number of parallel simulations per batch.
        bullhead_lookup (np.ndarray): O(1) bullhead penalty lookup array.
        device (torch.device): The device used for NN inference.
        model (TopologicalOpponentNet): The loaded PyTorch determinization network.
    """

    def __init__(self, player_idx, epsilon=0.2, tau=5.0, time_limit=0.8):
        """
        Initialize the Neural Determinization Monte Carlo player.

        Args:
            player_idx (int): The player's seat index in the game (0-3).
            epsilon (float): Ratio of random rollouts (0.0 to 1.0) mixed into the min-max policy.
            tau (float): Temperature parameter for the Softmax distribution.
            time_limit (float): Simulation budget in seconds.
        """
        self.player_idx = player_idx
        self.time_limit = time_limit
        self.epsilon = epsilon
        self.tau = tau
        self.total_cards = set(range(1, 105))
        self.batch_size = 5000  # Simultaneous simulations per batch
        self.bullhead_lookup = BULLHEAD_LOOKUP
        
        self.device = torch.device('cpu')
        self.model = TopologicalOpponentNet(input_dim=125).to(self.device)
        
        # Resolve path to weights (up 5 levels from src/players/b12705048/agents/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
        model_path = os.path.join(project_root, "models", "topological_net.pth")
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

    def action(self, hand, history):
        start_time = time.perf_counter()

        # ---- Phase 1: State Parsing ----
        if isinstance(history, dict):
            board = history.get('board', [])
            target_round = history.get('round', 0)
        else:
            board = history[-1]
            target_round = 0

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

        stats_penalty = {c: 0.0 for c in hand}
        stats_visits = {c: 0 for c in hand}

        orig_tails = np.array([row[-1] for row in board], dtype=np.int32)
        orig_lengths = np.array([len(row) for row in board], dtype=np.int32)
        orig_rbulls = np.array([sum(self.bullhead_lookup[c] for c in row) for row in board], dtype=np.int32)

        # ---- Phase 1.5: Neural Determinization Setup ----
        if isinstance(history, dict) and 'score_history' in history:
            X = build_feature_vector(history, target_round, self.player_idx, unseen_cards, len(hand))
            sorted_row_ends = get_topological_gaps(board)
            capacities = get_gap_capacities(sorted_row_ends, unseen_cards)
            
            with torch.no_grad():
                x_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(self.device)
                c_t = torch.tensor(capacities, dtype=torch.float32).unsqueeze(0).to(self.device)
                probs = self.model(x_t, gap_capacities=c_t).squeeze(0).numpy() # Shape (3, 5)
        else:
            # Fallback to uniform probabilities if history doesn't contain required keys
            probs = np.full((3, 5), 0.2)
            sorted_row_ends = get_topological_gaps(board)
            capacities = get_gap_capacities(sorted_row_ends, unseen_cards)
            
        log_probs = np.log(probs + 1e-10)
        
        card_log_weights = np.full((3, 105), -1e9, dtype=np.float32)
        for opp in range(3):
            for c in unseen_cards:
                k = assign_card_to_bucket(c, sorted_row_ends)
                card_log_weights[opp, c] = log_probs[opp, k] - np.log(max(1, capacities[k]))

        # ---- Phase 1.8: Compute Safety Scores S(c) ----
        deltas = np.arange(105)[:, None] - orig_tails[None, :]
        valid_mask = deltas > 0
        
        masked_deltas = np.where(valid_mask, deltas, np.inf)
        min_deltas = np.min(masked_deltas, axis=1)
        target_rows = np.argmin(masked_deltas, axis=1)
        
        is_invalid = np.isinf(min_deltas)
        
        target_lengths = orig_lengths[target_rows]
        target_rbulls = orig_rbulls[target_rows]
        
        S = np.zeros(105, dtype=np.float32)
        
        cond1 = (~is_invalid) & (target_lengths < 5)
        S[cond1] = -min_deltas[cond1]
        
        cond2 = (~is_invalid) & (target_lengths == 5)
        S[cond2] = -(10.0 * target_rbulls[cond2])
        
        S[is_invalid] = -100.0
        
        B = np.zeros(105, dtype=np.float32)
        B[1:11] = 2.0
        B[95:105] = 2.0
        S += B
        # -----------------------------------------------

        candidates = list(hand)
        n_stages = max(1, math.ceil(math.log2(len(hand))))
        stage_milestones = [start_time + (i + 1) * (self.time_limit / n_stages) for i in range(n_stages)]

        for stage in range(n_stages):
            milestone = stage_milestones[stage]
            
            while time.perf_counter() < milestone:
                num_cand = len(candidates)
                sims_per = self.batch_size // num_cand
                budget = {c: sims_per for c in candidates}
                actual_batch_size = sum(budget.values())
                if actual_batch_size == 0: 
                    break

                # ---- Phase 2: Batch Initialization & Deal ----
                tails = np.tile(orig_tails, (actual_batch_size, 1))
                lengths = np.tile(orig_lengths, (actual_batch_size, 1))
                rbulls = np.tile(orig_rbulls, (actual_batch_size, 1))
                penalties = np.zeros((actual_batch_size, 4), dtype=np.int32)

                opp_indices = [i for i in range(4) if i != self.player_idx]

                # Neural Determinization: Sequential Gumbel-Max Sampling without replacement
                opp_hands_unsorted = np.zeros((actual_batch_size, 3, n_turns), dtype=np.int32)
                available_mask = np.zeros((actual_batch_size, 105), dtype=bool)
                available_mask[:, unseen_cards] = True
                
                for opp in range(3):
                    U = np.random.uniform(1e-8, 1.0 - 1e-8, size=(actual_batch_size, 105))
                    noisy_w = card_log_weights[opp] - np.log(-np.log(U))
                    
                    noisy_w[~available_mask] = -1e9
                    
                    sort_idx = np.argsort(-noisy_w, axis=1)
                    chosen_cards = sort_idx[:, :n_turns]
                    opp_hands_unsorted[:, opp, :] = chosen_cards
                    
                    np.put_along_axis(available_mask, chosen_cards, False, axis=1)
                
                opp_hands = np.sort(opp_hands_unsorted, axis=2)

                # Min/Max + Softmax Rollout simulation
                choices = (np.random.rand(actual_batch_size, 3, n_turns) > 0.5).astype(np.int32)
                min_counts = np.cumsum(1 - choices, axis=2)
                min_counts_shifted = np.concatenate([np.zeros((actual_batch_size, 3, 1), dtype=np.int32), min_counts[:, :, :-1]], axis=2)
                max_counts = np.cumsum(choices, axis=2)
                max_counts_shifted = np.concatenate([np.zeros((actual_batch_size, 3, 1), dtype=np.int32), max_counts[:, :, :-1]], axis=2)
                
                left_indices = min_counts_shifted
                right_indices = (n_turns - 1) - max_counts_shifted
                selected_indices = np.where(choices == 0, left_indices, right_indices)
                chosen_opp_cards = np.take_along_axis(opp_hands, selected_indices, axis=2)

                eps_mask_opp = np.random.rand(actual_batch_size, 3, 1) < self.epsilon
                
                opp_scores = S[opp_hands_unsorted]
                U_opp = np.random.uniform(1e-8, 1.0 - 1e-8, size=(actual_batch_size, 3, n_turns))
                noisy_opp_scores = (opp_scores / self.tau) - np.log(-np.log(U_opp))
                
                sort_idx_opp = np.argsort(-noisy_opp_scores, axis=2)
                softmax_opp_cards = np.take_along_axis(opp_hands_unsorted, sort_idx_opp, axis=2)
                
                final_opp_cards = np.where(eps_mask_opp, softmax_opp_cards, chosen_opp_cards)

                hands_array = np.zeros((actual_batch_size, 4, n_turns), dtype=np.int32)
                hands_array[:, opp_indices[0], :] = final_opp_cards[:, 0, :]
                hands_array[:, opp_indices[1], :] = final_opp_cards[:, 1, :]
                hands_array[:, opp_indices[2], :] = final_opp_cards[:, 2, :]

                # Assign our candidate cards
                current_start = 0
                for c in candidates:
                    sims_per_cand = budget[c]
                    start_b = current_start
                    end_b = start_b + sims_per_cand
                    current_start = end_b
                    
                    if sims_per_cand == 0:
                        continue

                    my_rest = [x for x in hand if x != c]
                    hands_array[start_b:end_b, self.player_idx, 0] = c
                    
                    if len(my_rest) > 0:
                        my_hands_chunk = np.tile(np.sort(np.array(my_rest, dtype=np.int32)), (sims_per_cand, 1))
                        n_rem = len(my_rest)
                        
                        choices_my = (np.random.rand(sims_per_cand, n_rem) > 0.5).astype(np.int32)
                        min_my = np.cumsum(1 - choices_my, axis=1)
                        min_s_my = np.concatenate([np.zeros((sims_per_cand, 1), dtype=np.int32), min_my[:, :-1]], axis=1)
                        max_my = np.cumsum(choices_my, axis=1)
                        max_s_my = np.concatenate([np.zeros((sims_per_cand, 1), dtype=np.int32), max_my[:, :-1]], axis=1)
                        
                        left_my = min_s_my
                        right_my = (n_rem - 1) - max_s_my
                        sel_my = np.where(choices_my == 0, left_my, right_my)
                        
                        chosen_my = np.take_along_axis(my_hands_chunk, sel_my, axis=1)
                        
                        my_rest_arr = np.array(my_rest, dtype=np.int32)
                        my_hands_unsorted = np.tile(my_rest_arr, (sims_per_cand, 1))
                        
                        eps_mask_my = np.random.rand(sims_per_cand, 1) < self.epsilon
                        
                        my_scores = S[my_hands_unsorted]
                        U_my = np.random.uniform(1e-8, 1.0 - 1e-8, size=(sims_per_cand, n_rem))
                        noisy_my_scores = (my_scores / self.tau) - np.log(-np.log(U_my))
                        
                        my_perm = np.argsort(-noisy_my_scores, axis=1)
                        my_hands_softmax = np.take_along_axis(my_hands_unsorted, my_perm, axis=1)
                        
                        final_my = np.where(eps_mask_my, my_hands_softmax, chosen_my)
                        
                        hands_array[start_b:end_b, self.player_idx, 1:] = final_my

                # ---- Phase 3: SIMD Batch Simulation Loop ----
                for t in range(n_turns):
                    played_cards = hands_array[:, :, t]

                    sort_idx = np.argsort(played_cards, axis=1)
                    sorted_cards = np.take_along_axis(played_cards, sort_idx, axis=1)
                    sorted_players = sort_idx

                    for i in range(4):
                        current_cards = sorted_cards[:, i]
                        current_players = sorted_players[:, i]

                        valid = np.where(current_cards[:, None] > tails, tails, -1)
                        target_rows = np.argmax(valid, axis=1)
                        invalid_mask = np.max(valid, axis=1) == -1

                        scores = rbulls * 1000 + lengths * 10 + np.arange(4)
                        min_rows = np.argmin(scores, axis=1)
                        target_rows = np.where(invalid_mask, min_rows, target_rows)

                        b_idx = np.arange(actual_batch_size)
                        target_lengths = lengths[b_idx, target_rows]
                        target_bullheads = rbulls[b_idx, target_rows]

                        penalty_condition = invalid_mask | (target_lengths == 5)
                        normal_cond = ~penalty_condition
                        card_bulls = self.bullhead_lookup[current_cards]

                        if np.any(penalty_condition):
                            pc = penalty_condition
                            b_pc = b_idx[pc]
                            p_players = current_players[pc]
                            
                            penalties[b_pc, p_players] += target_bullheads[pc]
                            lengths[b_pc, target_rows[pc]] = 1
                            tails[b_pc, target_rows[pc]] = current_cards[pc]
                            rbulls[b_pc, target_rows[pc]] = card_bulls[pc]

                        if np.any(normal_cond):
                            nc = normal_cond
                            b_nc = b_idx[nc]
                            
                            lengths[b_nc, target_rows[nc]] += 1
                            tails[b_nc, target_rows[nc]] = current_cards[nc]
                            rbulls[b_nc, target_rows[nc]] += card_bulls[nc]

                # ---- Phase 4: Stat Aggregation ----
                current_start = 0
                for c in candidates:
                    sims_per_cand = budget[c]
                    start_b = current_start
                    end_b = start_b + sims_per_cand
                    current_start = end_b
                    
                    if sims_per_cand == 0:
                        continue
                        
                    my_pens = penalties[start_b:end_b, self.player_idx]
                    stats_penalty[c] += np.sum(my_pens)
                    stats_visits[c] += sims_per_cand

            # Successive Halving: drop the worst half
            if len(candidates) > 1:
                candidates.sort(key=lambda c: stats_penalty[c] / max(1, stats_visits[c]))
                keep = math.ceil(len(candidates) / 2)
                candidates = candidates[:keep]

        best_card = min(hand, key=lambda k: stats_penalty.get(k, 0.0) / max(1, stats_visits.get(k, 0)))
        return best_card
