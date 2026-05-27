"""
Flat Monte Carlo (1-Ply) Player Module - O(1) Vectorized.

This module implements a 1-ply Monte Carlo evaluation agent for 6 Nimmt!
using a massive Structure of Arrays (SoA) vectorization across batch_size games.
"""

import time
import random
import numpy as np

class FlatMCo1:
    """
    Vectorized 1-ply Monte Carlo agent for 6 Nimmt!.

    Evaluates each candidate card simultaneously across batched simulations
    by leveraging Numpy boolean masks and SoA architecture to achieve
    orders of magnitude higher simulation throughput with EXACT game rules.
    """

    def __init__(self, player_idx):
        self.player_idx = player_idx
        self.time_limit = 0.90
        self.total_cards = set(range(1, 105))
        self.batch_size = 5000  # Simultaneous simulations per batch

        # Pre-compute bullhead lookup table
        bullheads = [0] * 105
        for card in range(1, 105):
            if card == 55:
                bullheads[card] = 7
            elif card % 11 == 0:
                bullheads[card] = 5
            elif card % 10 == 0:
                bullheads[card] = 3
            elif card % 5 == 0:
                bullheads[card] = 2
            else:
                bullheads[card] = 1
        self.bullhead_lookup = np.array(bullheads, dtype=np.int32)

    def action(self, hand, history):
        start_time = time.perf_counter()

        # Parse history
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
            sims_per_cand = self.batch_size // num_cand
            actual_batch_size = sims_per_cand * num_cand

            if actual_batch_size == 0:
                break

            # Initialize SoA arrays for the batch
            tails = np.tile(orig_tails, (actual_batch_size, 1))
            lengths = np.tile(orig_lengths, (actual_batch_size, 1))
            rbulls = np.tile(orig_rbulls, (actual_batch_size, 1))
            penalties = np.zeros((actual_batch_size, 4), dtype=np.int32)

            # Deal hands
            # Assign random weights to valid unseen cards, take argsort to sample without replacement
            rand_weights = np.random.rand(actual_batch_size, 105)
            # Make seen cards invalid by setting their weight to -1
            unseen_mask = np.tile(unseen_mask_base, (actual_batch_size, 1))
            rand_weights[~unseen_mask] = -1.0

            perm = np.argsort(-rand_weights, axis=1)

            opp_indices = [i for i in range(4) if i != self.player_idx]

            # hands_array shape: (actual_batch_size, 4_players, n_turns)
            hands_array = np.zeros((actual_batch_size, 4, n_turns), dtype=np.int32)
            hands_array[:, opp_indices[0], :] = perm[:, 0:n_turns]
            hands_array[:, opp_indices[1], :] = perm[:, n_turns:2*n_turns]
            hands_array[:, opp_indices[2], :] = perm[:, 2*n_turns:3*n_turns]

            # Assign our candidate cards
            c_idx = 0
            for c in candidates:
                start_b = c_idx * sims_per_cand
                end_b = start_b + sims_per_cand

                my_rest = [x for x in hand if x != c]
                rest_arr = np.array(my_rest, dtype=np.int32)
                my_hands_chunk = np.tile(rest_arr, (sims_per_cand, 1))
                
                if len(my_rest) > 0:
                    rand_my = np.random.rand(sims_per_cand, len(my_rest))
                    my_perm = np.argsort(rand_my, axis=1)
                    my_hands_chunk = np.take_along_axis(my_hands_chunk, my_perm, axis=1)

                hands_array[start_b:end_b, self.player_idx, 0] = c
                if len(my_rest) > 0:
                    hands_array[start_b:end_b, self.player_idx, 1:] = my_hands_chunk

                c_idx += 1

            # Simulate n_turns synchronously across all games
            for t in range(n_turns):
                played_cards = hands_array[:, :, t]

                # Exact rule: sort the 4 cards within each game
                sort_idx = np.argsort(played_cards, axis=1)
                sorted_cards = np.take_along_axis(played_cards, sort_idx, axis=1)
                sorted_players = sort_idx

                # Process sequentially from lowest card to highest
                for i in range(4):
                    current_cards = sorted_cards[:, i]
                    current_players = sorted_players[:, i]

                    # Target row is the row with max tail strictly less than card
                    valid = np.where(current_cards[:, None] > tails, tails, -1)
                    target_rows = np.argmax(valid, axis=1)
                    invalid_mask = np.max(valid, axis=1) == -1

                    # For invalid cards, target the row with min score
                    scores = rbulls * 1000 + lengths * 10 + np.arange(4)
                    min_rows = np.argmin(scores, axis=1)
                    target_rows = np.where(invalid_mask, min_rows, target_rows)

                    # Update logic
                    b_idx = np.arange(actual_batch_size)

                    target_lengths = lengths[b_idx, target_rows]
                    target_bullheads = rbulls[b_idx, target_rows]

                    # Penalty occurs if invalid, or if placing the 6th card (length == 5)
                    penalty_condition = invalid_mask | (target_lengths == 5)
                    normal_cond = ~penalty_condition

                    card_bulls = self.bullhead_lookup[current_cards]

                    if np.any(penalty_condition):
                        pc = penalty_condition
                        b_pc = b_idx[pc]
                        p_players = current_players[pc]
                        
                        # Add penalty
                        penalties[b_pc, p_players] += target_bullheads[pc]
                        
                        # Reset row
                        lengths[b_pc, target_rows[pc]] = 1
                        tails[b_pc, target_rows[pc]] = current_cards[pc]
                        rbulls[b_pc, target_rows[pc]] = card_bulls[pc]

                    if np.any(normal_cond):
                        nc = normal_cond
                        b_nc = b_idx[nc]
                        
                        # Append to row
                        lengths[b_nc, target_rows[nc]] += 1
                        tails[b_nc, target_rows[nc]] = current_cards[nc]
                        rbulls[b_nc, target_rows[nc]] += card_bulls[nc]

            # Aggregate stats
            c_idx = 0
            for c in candidates:
                start_b = c_idx * sims_per_cand
                end_b = start_b + sims_per_cand

                my_pens = penalties[start_b:end_b, self.player_idx]
                stats_penalty[c] += np.sum(my_pens)
                stats_visits[c] += sims_per_cand
                c_idx += 1

        best_card = min(hand, key=lambda k: stats_penalty[k] / max(1, stats_visits[k]))
        return best_card
