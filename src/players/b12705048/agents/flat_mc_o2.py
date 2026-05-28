"""
Flat Monte Carlo (2-Ply) Player Module — Vectorized SoA Variant.

This module implements a high-throughput 2-ply Monte Carlo agent for 6 Nimmt!
using a Structure of Arrays (SoA) architecture and NumPy SIMD batch execution.
It extends the 1-ply approach by evaluating pairs of (current_turn, next_turn)
actions for the agent and finding the best immediate action assuming the
subsequent best response.

Algorithm:
    1. Build SoA arrays for ``batch_size`` independent games.
    2. Enumerate all valid 2-ply action pairs `(c1, c2)` for the agent.
    3. Randomly assign unseen cards to opponents via vectorized argsort.
    4. Simulate all ``n_turns`` tricks simultaneously via SIMD operations.
    5. Aggregate penalties and use minimax logic to select the best `c1`.

Characteristics:
    - **Depth**: 2-ply (evaluates current and next action).
    - **Rollout Policy**: Pure uniform random for all players.
    - **SIMD Batching**: Highly vectorized game simulation for large throughput.
    - **Time Management**: Repeats batches until time expires.

See Also:
    ``flat_mc_o1.py`` — 1-Ply vectorized counterpart.
"""

import time
import random
import numpy as np

class FlatMCo2:
    """
    Vectorized 2-ply Monte Carlo agent for 6 Nimmt!.

    Evaluates pairs of candidate cards `(c1, c2)` simultaneously across
    ``batch_size`` parallel game simulations via NumPy SoA arrays.

    Attributes:
        player_idx (int): This agent's seat index (0–3).
        time_limit (float): Wall-clock budget in seconds per ``action()`` call.
        total_cards (set[int]): The full card universe {1, ..., 104}.
        batch_size (int): Number of simultaneous simulations per batch.
        bullhead_lookup (np.ndarray): O(1) bullhead penalty lookup array.
    """

    def __init__(self, player_idx):
        """
        Initialize the Vectorized 2-Ply Flat Monte Carlo player.

        Args:
            player_idx (int): The player's seat index in the game (0–3).
        """
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
        """
        Evaluate candidate pairs via batched SoA simulation and return
        the card with the lowest expected penalty under 2-ply assumptions.

        Args:
            hand (list[int]): Cards currently available to play.
            history (dict | list): Game state from the engine.

        Returns:
            int: The card value with the lowest average simulated penalty.
        """
        start_time = time.perf_counter()

        n_turns = len(hand)
        if n_turns == 1:
            return hand[0]

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
        
        # 2-Ply Candidates
        candidates = [(c1, c2) for c1 in hand for c2 in hand if c1 != c2]
        num_cand = len(candidates)
        
        # For small hands or near end-game, adjust to avoid zero division
        if num_cand == 0: 
            return hand[0]

        stats_penalty = {pair: 0.0 for pair in candidates}
        stats_visits = {pair: 0 for pair in candidates}

        orig_tails = np.array([row[-1] for row in board], dtype=np.int32)
        orig_lengths = np.array([len(row) for row in board], dtype=np.int32)
        orig_rbulls = np.array([sum(self.bullhead_lookup[c] for c in row) for row in board], dtype=np.int32)

        unseen_mask_base = np.zeros(105, dtype=bool)
        unseen_mask_base[unseen_cards] = True

        while time.perf_counter() - start_time < self.time_limit:
            sims_per_cand = self.batch_size // num_cand
            actual_batch_size = sims_per_cand * num_cand

            if actual_batch_size == 0:
                break

            # ---- Phase 2: Batch Initialization & Deal ----
            # Initialize SoA arrays for the batch
            tails = np.tile(orig_tails, (actual_batch_size, 1))
            lengths = np.tile(orig_lengths, (actual_batch_size, 1))
            rbulls = np.tile(orig_rbulls, (actual_batch_size, 1))
            penalties = np.zeros((actual_batch_size, 4), dtype=np.int32)

            # Deal hands to opponents
            rand_weights = np.random.rand(actual_batch_size, 105)
            unseen_mask = np.tile(unseen_mask_base, (actual_batch_size, 1))
            rand_weights[~unseen_mask] = -1.0

            perm = np.argsort(-rand_weights, axis=1)

            opp_indices = [i for i in range(4) if i != self.player_idx]

            # hands_array shape: (actual_batch_size, 4_players, n_turns)
            hands_array = np.zeros((actual_batch_size, 4, n_turns), dtype=np.int32)
            hands_array[:, opp_indices[0], :] = perm[:, 0:n_turns]
            hands_array[:, opp_indices[1], :] = perm[:, n_turns:2*n_turns]
            hands_array[:, opp_indices[2], :] = perm[:, 2*n_turns:3*n_turns]

            # Assign our candidate pairs
            c_idx = 0
            for (c1, c2) in candidates:
                start_b = c_idx * sims_per_cand
                end_b = start_b + sims_per_cand

                hands_array[start_b:end_b, self.player_idx, 0] = c1
                hands_array[start_b:end_b, self.player_idx, 1] = c2

                my_rest = [x for x in hand if x not in (c1, c2)]
                if len(my_rest) > 0:
                    rest_arr = np.array(my_rest, dtype=np.int32)
                    my_hands_chunk = np.tile(rest_arr, (sims_per_cand, 1))
                    
                    rand_my = np.random.rand(sims_per_cand, len(my_rest))
                    my_perm = np.argsort(rand_my, axis=1)
                    my_hands_chunk = np.take_along_axis(my_hands_chunk, my_perm, axis=1)
                    hands_array[start_b:end_b, self.player_idx, 2:] = my_hands_chunk

                c_idx += 1

            # ---- Phase 3: SIMD Batch Simulation Loop ----
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

            # ---- Phase 4: Stat Aggregation ----
            c_idx = 0
            for pair in candidates:
                start_b = c_idx * sims_per_cand
                end_b = start_b + sims_per_cand

                my_pens = penalties[start_b:end_b, self.player_idx]
                stats_penalty[pair] += np.sum(my_pens)
                stats_visits[pair] += sims_per_cand
                c_idx += 1

        # ---- Phase 5: 2-Ply Minimax Aggregation ----
        # score(c1) = min_{c2} (avg_penalty(c1, c2))
        best_c1 = None
        best_c1_score = float('inf')

        for c1 in hand:
            min_c2_penalty = float('inf')
            for c2 in hand:
                if c1 == c2: continue
                pair = (c1, c2)
                avg_pen = stats_penalty[pair] / max(1, stats_visits[pair])
                if avg_pen < min_c2_penalty:
                    min_c2_penalty = avg_pen
            
            if min_c2_penalty < best_c1_score:
                best_c1_score = min_c2_penalty
                best_c1 = c1

        # Fallback if time budget ran out before 1 batch finished (stats_visits = 0 for all)
        if best_c1 is None:
            best_c1 = hand[0]

        return best_c1
