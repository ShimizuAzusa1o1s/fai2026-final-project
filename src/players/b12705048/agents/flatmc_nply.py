"""
Flat Monte Carlo (N-Ply) Player Module — Vectorized SoA Variant.

This module implements a high-throughput N-ply Monte Carlo agent for 6 Nimmt!
using a Structure of Arrays (SoA) architecture and NumPy SIMD batch execution.
It extends the 1-ply approach by evaluating sequences of N actions
and finding the best immediate action assuming the subsequent best responses.

Algorithm:
    1. Build SoA arrays for ``batch_size`` independent games.
    2. Enumerate all valid N-ply action sequences (permutations) for the agent.
    3. Randomly assign unseen cards to opponents via vectorized argsort.
    4. Simulate all ``n_turns`` tricks simultaneously via SIMD operations.
    5. Aggregate penalties and use minimax logic to select the best first action.

Characteristics:
    - **Depth**: N-ply (evaluates sequences of up to N actions).
    - **Rollout Policy**: Pure uniform random for all players.
    - **SIMD Batching**: Highly vectorized game simulation for large throughput.
    - **Time Management**: Repeats batches until time expires.
"""

import time
import random
import numpy as np
import itertools

class FlatMCNPly:
    """
    Vectorized N-ply Monte Carlo agent for 6 Nimmt!.

    Evaluates sequences of N candidate cards simultaneously across
    ``batch_size`` parallel game simulations via NumPy SoA arrays.

    Attributes:
        player_idx (int): This agent's seat index (0–3).
        n_ply (int): The search depth (number of actions to evaluate as a sequence).
        time_limit (float): Wall-clock budget in seconds per ``action()`` call.
        total_cards (set[int]): The full card universe {1, ..., 104}.
        batch_size (int): Number of simultaneous simulations per batch.
        bullhead_lookup (np.ndarray): O(1) bullhead penalty lookup array.
    """

    def __init__(self, player_idx, n_ply=3):
        """
        Initialize the Vectorized N-Ply Flat Monte Carlo player.

        Args:
            player_idx (int): The player's seat index in the game (0–3).
            n_ply (int): The sequence length (depth) to evaluate (default: 2).
        """
        self.player_idx = player_idx
        self.n_ply = n_ply
        self.time_limit = 0.90
        self.total_cards = set(range(1, 105))
        self.batch_size = 5000  # Base simulations per batch

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
        Evaluate candidate sequences via batched SoA simulation and return
        the card with the lowest expected penalty under N-ply assumptions.

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

        # Determine actual N-ply depth
        actual_n_ply = min(self.n_ply, n_turns)

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
        
        # N-Ply Candidates (permutations of length actual_n_ply)
        candidates = list(itertools.permutations(hand, actual_n_ply))
        num_cand = len(candidates)
        
        if num_cand == 0: 
            return hand[0]

        stats_penalty = {cand: 0.0 for cand in candidates}
        stats_visits = {cand: 0 for cand in candidates}

        orig_tails = np.array([row[-1] for row in board], dtype=np.int32)
        orig_lengths = np.array([len(row) for row in board], dtype=np.int32)
        orig_rbulls = np.array([sum(self.bullhead_lookup[c] for c in row) for row in board], dtype=np.int32)

        unseen_mask_base = np.zeros(105, dtype=bool)
        unseen_mask_base[unseen_cards] = True

        while time.perf_counter() - start_time < self.time_limit:
            # For very large N, we ensure at least 1 simulation per candidate
            sims_per_cand = max(1, self.batch_size // num_cand)
            actual_batch_size = sims_per_cand * num_cand

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

            # Assign our candidate sequences
            c_idx = 0
            for cand_tuple in candidates:
                start_b = c_idx * sims_per_cand
                end_b = start_b + sims_per_cand

                # Assign the deterministic N-ply sequence
                for i, c in enumerate(cand_tuple):
                    hands_array[start_b:end_b, self.player_idx, i] = c

                # Randomize the remaining cards in hand
                my_rest = [x for x in hand if x not in cand_tuple]
                if len(my_rest) > 0:
                    rest_arr = np.array(my_rest, dtype=np.int32)
                    my_hands_chunk = np.tile(rest_arr, (sims_per_cand, 1))
                    
                    rand_my = np.random.rand(sims_per_cand, len(my_rest))
                    my_perm = np.argsort(rand_my, axis=1)
                    my_hands_chunk = np.take_along_axis(my_hands_chunk, my_perm, axis=1)
                    hands_array[start_b:end_b, self.player_idx, actual_n_ply:] = my_hands_chunk

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
            for cand in candidates:
                start_b = c_idx * sims_per_cand
                end_b = start_b + sims_per_cand

                my_pens = penalties[start_b:end_b, self.player_idx]
                stats_penalty[cand] += np.sum(my_pens)
                stats_visits[cand] += sims_per_cand
                c_idx += 1

        # ---- Phase 5: N-Ply Minimax Aggregation ----
        total_visits = sum(stats_visits.values())
        if total_visits == 0:
            return hand[0]

        best_cand = min(
            candidates,
            key=lambda cand: stats_penalty[cand] / max(1, stats_visits[cand])
        )
        return best_cand[0]
