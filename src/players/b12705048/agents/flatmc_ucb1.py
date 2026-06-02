"""
MinMax Monte Carlo (1-Ply) Player Module — Vectorized SoA Variant.

Algorithm:
    1. Build SoA arrays for ``batch_size`` independent games.
    2. Randomly assign unseen cards to opponents.
    3. During rollout, instead of playing random cards, all simulated players
       (including our future selves and all opponents) only choose their 
       minimum or maximum available card (50/50 probability).
    4. Simulate all tricks across all games simultaneously using NumPy.

Characteristics:
    - **Depth**: 1-ply (evaluates immediate action).
    - **Rollout Policy**: Min-Max stochastic (simulates edge-playing heuristics).
    - **Time Management**: Repeats batches until wall-clock budget expires.

See Also:
    ``flatmc.py`` — Uniform random rollout baseline.
"""

import time
import math
import numpy as np

from src.players.b12705048.core.constants import BULLHEAD_LOOKUP

class FlatMCUCB1:
    """
    Vectorized 1-ply Monte Carlo agent using UCB1 for dynamic candidate exploration and exploitation.

    Attributes:
        player_idx (int): This agent's seat index (0-3).
        time_limit (float): Simulation budget in seconds.
        total_cards (set[int]): The full card universe {1, ..., 104}.
        batch_size (int): Number of parallel simulations per batch.
        bullhead_lookup (np.ndarray): O(1) bullhead penalty lookup array.
    """

    def __init__(self, player_idx, c_param=5.0, epsilon=0.1, time_limit=0.8):
        """
        Initialize the UCB1 MC player.

        Args:
            player_idx (int): The player's seat index in the game (0-3).
            c_param (float): Exploration constant for UCB1.
            epsilon (float): Ratio of random rollouts (0.0 to 1.0) mixed into the min-max policy.
            time_limit (float): Simulation budget in seconds.
        """
        self.player_idx = player_idx
        self.c_param = c_param
        self.time_limit = time_limit
        self.epsilon = epsilon
        self.total_cards = set(range(1, 105))
        self.batch_size = 5000  # Simultaneous simulations per batch
        self.bullhead_lookup = BULLHEAD_LOOKUP

    def action(self, hand, history):
        """
        Evaluate candidate cards via batched SoA simulation with min-max rollout.

        Args:
            hand (list[int]): Cards currently available to play.
            history (dict | list): Game state from the engine.

        Returns:
            int: The card with the lowest average simulated penalty.
        """
        start_time = time.perf_counter()

        # ---- Phase 1: State Parsing ----
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

        stats_penalty = {c: 0.0 for c in hand}
        stats_visits = {c: 0 for c in hand}

        orig_tails = np.array([row[-1] for row in board], dtype=np.int32)
        orig_lengths = np.array([len(row) for row in board], dtype=np.int32)
        orig_rbulls = np.array([sum(self.bullhead_lookup[c] for c in row) for row in board], dtype=np.int32)

        unseen_mask_base = np.zeros(105, dtype=bool)
        unseen_mask_base[unseen_cards] = True

        while time.perf_counter() - start_time < self.time_limit:
            candidates = hand
            num_cand = len(candidates)
            
            budget = {}
            total_visits = sum(stats_visits.values())
            unvisited = [c for c in candidates if stats_visits[c] == 0]
            
            if unvisited:
                sims_per = self.batch_size // len(unvisited)
                for c in candidates:
                    budget[c] = sims_per if c in unvisited else 0
            else:
                best_cand = None
                best_ucb = float('inf')
                for c in candidates:
                    avg_penalty = stats_penalty[c] / stats_visits[c]
                    ucb_cost = avg_penalty - self.c_param * math.sqrt(math.log(total_visits) / stats_visits[c])
                    if ucb_cost < best_ucb:
                        best_ucb = ucb_cost
                        best_cand = c
                        
                for c in candidates:
                    budget[c] = self.batch_size if c == best_cand else 0
                        
            actual_batch_size = sum(budget.values())
            if actual_batch_size == 0: break

            # ---- Phase 2: Batch Initialization & Deal ----
            tails = np.tile(orig_tails, (actual_batch_size, 1))
            lengths = np.tile(orig_lengths, (actual_batch_size, 1))
            rbulls = np.tile(orig_rbulls, (actual_batch_size, 1))
            penalties = np.zeros((actual_batch_size, 4), dtype=np.int32)

            rand_weights = np.random.rand(actual_batch_size, 105)
            unseen_mask = np.tile(unseen_mask_base, (actual_batch_size, 1))
            rand_weights[~unseen_mask] = -1.0
            perm = np.argsort(-rand_weights, axis=1)

            opp_indices = [i for i in range(4) if i != self.player_idx]

            # Vectorized Min/Max Rollout Generation for Opponents
            opp_hands_unsorted = np.zeros((actual_batch_size, 3, n_turns), dtype=np.int32)
            opp_hands_unsorted[:, 0, :] = perm[:, 0:n_turns]
            opp_hands_unsorted[:, 1, :] = perm[:, n_turns:2*n_turns]
            opp_hands_unsorted[:, 2, :] = perm[:, 2*n_turns:3*n_turns]
            opp_hands = np.sort(opp_hands_unsorted, axis=2)

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
            final_opp_cards = np.where(eps_mask_opp, opp_hands_unsorted, chosen_opp_cards)

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
                    rand_my = np.random.rand(sims_per_cand, n_rem)
                    my_perm = np.argsort(rand_my, axis=1)
                    my_hands_random = np.take_along_axis(my_hands_unsorted, my_perm, axis=1)
                    
                    eps_mask_my = np.random.rand(sims_per_cand, 1) < self.epsilon
                    final_my = np.where(eps_mask_my, my_hands_random, chosen_my)
                    
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

        best_card = min(hand, key=lambda k: stats_penalty[k] / max(1, stats_visits[k]))
        return best_card
