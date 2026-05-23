"""
NumPy-Vectorized Flat Monte Carlo (1-Ply) Player Module.

This module provides a heavily optimized variant of the Flat Monte Carlo
agent (see ``flat_mc.py``) that achieves ~10× higher simulation throughput
by leveraging NumPy array operations to evaluate all candidate cards
simultaneously across batches of N parallel simulations.

Key Optimizations over flat_mc.py:
    1. **Batch Parallelism**: All C candidates × N simulations are evaluated
       in a single set of NumPy operations, eliminating Python-level loops
       over candidates and simulations.
    2. **Pre-allocated Buffers**: Board state arrays (tails, lengths,
       bullheads, penalties) are allocated once and reused via in-place
       writes, minimizing garbage collection pressure.
    3. **Vectorized Trick Resolution**: The 6 Nimmt! placement rules
       (target row finding, 6th-card detection, Low Card Rule) are
       implemented using broadcasting, argmin, and conditional masking
       instead of per-card Python loops.
    4. **Efficient Random Sampling**: Uses ``np.argpartition`` to
       draw opponent hands from the unseen pool in O(n) instead of
       O(n log n) full sort, and ``np.random.default_rng`` for
       faster PRNG state management.

Tensor Shapes (key dimensions):
    - C: number of candidate cards in hand
    - N: batch size (simulations per evaluation loop iteration)
    - T: number of turns remaining (= hand size)
    - Board state arrays are shape (C, N, 4) — one per row per simulation

Constraints:
    - Single-threaded (no multiprocessing/threading) per tournament rules.
    - Time-budgeted: runs as many batch iterations as possible within
      the configured wall-clock limit (default 0.85 seconds).
"""

import time
import numpy as np


class FlatMCO1:
    """
    NumPy-vectorized 1-ply Monte Carlo agent for 6 Nimmt!.

    Evaluates all candidate cards simultaneously across batched random
    simulations, selecting the action with the lowest average penalty.

    Attributes:
        player_idx (int): This agent's seat index (0–3).
        time_limit (float): Wall-clock budget per ``action()`` call (seconds).
        total_cards (set[int]): The full card universe {1, ..., 104}.
        bullhead_lookup_array (np.ndarray): Bullhead penalties indexed by card
            value, shape (105,) with index 0 unused.
        rng (np.random.Generator): NumPy PRNG instance for fast random sampling.
    """

    def __init__(self, player_idx):
        """
        Initialize the vectorized Flat Monte Carlo player.

        Args:
            player_idx (int): The player's seat index in the game (0–3).
        """
        self.player_idx = player_idx
        self.time_limit = 0.85
        self.total_cards = set(range(1, 105))

        # Pre-compute bullhead lookup as a NumPy array for vectorized indexing
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
        self.bullhead_lookup_array = np.array(bullheads, dtype=np.int32)
        self.rng = np.random.default_rng()

    def action(self, hand, history):
        """
        Evaluate all candidate cards via batched vectorized Monte Carlo
        simulations and return the card with the lowest expected penalty.

        Args:
            hand (list[int]): Cards currently available to play.
            history (dict | list): Game state from the engine, used to
                determine the unseen card pool.

        Returns:
            int: The card value with the lowest average simulated penalty.
        """
        start_time = time.perf_counter()

        # ---- 1. State Parsing ----
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
            return hand[0]

        opp_indices = [i for i in range(4) if i != self.player_idx]

        # ---- 2. Initialize Base Board State ----
        orig_row_bullheads = [sum(self.bullhead_lookup_array[c] for c in row) for row in board]
        base_tails = np.array([row[-1] for row in board], dtype=np.int32)
        base_lengths = np.array([len(row) for row in board], dtype=np.int32)
        base_bullheads = np.array(orig_row_bullheads, dtype=np.int32)

        C = len(hand)        # Number of candidate actions
        unseen = np.array(unseen_cards, dtype=np.int32)
        n_unseen = len(unseen)

        # For each candidate card, precompute the remaining hand
        my_rem = []
        for c in hand:
            rem = [x for x in hand if x != c]
            my_rem.append(rem)
        my_rem = np.array(my_rem, dtype=np.int32)  # Shape: (C, T-1)

        hand_array = np.array(hand, dtype=np.int32)[:, None]  # Shape: (C, 1)

        # Cumulative penalty statistics across all batches
        stats_penalty = np.zeros(C, dtype=np.int64)
        stats_visits = 0

        # Dynamically size the batch to balance throughput vs. memory overhead.
        # Larger batches amortize Python overhead but consume more RAM.
        N = max(100, 15000 // C)

        # ---- 3. Pre-allocate Simulation Buffers ----
        # These are reused each batch iteration to avoid allocation overhead.
        tails_buf = np.empty((C, N, 4), dtype=np.int32)
        lengths_buf = np.empty((C, N, 4), dtype=np.int32)
        bullheads_buf = np.empty((C, N, 4), dtype=np.int32)
        penalties_buf = np.empty((C, N, 4), dtype=np.int32)

        plays_buf = np.empty((C, N, 4), dtype=np.int32)
        diff_buf = np.empty((C, N, 4), dtype=np.int32)
        score_buf = np.empty((C, N, 4), dtype=np.int32)

        # Broadcast base state to (C, N, 4) for fast reinitialization
        base_tails_b = np.broadcast_to(base_tails, (C, N, 4))
        base_lengths_b = np.broadcast_to(base_lengths, (C, N, 4))
        base_bullheads_b = np.broadcast_to(base_bullheads, (C, N, 4))

        # Pre-expand remaining-hand array for vectorized permutation sampling
        my_rem_expanded = np.expand_dims(my_rem, axis=1)           # (C, 1, T-1)
        my_rem_broadcasted = np.broadcast_to(my_rem_expanded, (C, N, n_turns - 1))

        # ---- 4. Main Simulation Loop ----
        while time.perf_counter() - start_time < self.time_limit - 0.05:
            # Reinitialize board state buffers from the base snapshot
            tails_buf[:] = base_tails_b
            lengths_buf[:] = base_lengths_b
            bullheads_buf[:] = base_bullheads_b
            penalties_buf.fill(0)

            # Generate random opponent hands from the unseen card pool.
            # Uses argpartition for O(n) partial sorting instead of O(n log n).
            noise = self.rng.random((N, n_unseen))
            req_cards = 3 * n_turns
            if req_cards < n_unseen:
                perm_indices = np.argpartition(noise, req_cards - 1, axis=1)[:, :req_cards]
            else:
                perm_indices = noise.argsort(axis=1)

            opp0_cards = unseen[perm_indices[:, 0:n_turns]]             # (N, T)
            opp1_cards = unseen[perm_indices[:, n_turns:2*n_turns]]     # (N, T)
            opp2_cards = unseen[perm_indices[:, 2*n_turns:3*n_turns]]   # (N, T)

            # Generate random permutations of our remaining cards (per candidate)
            my_noise = self.rng.random((C, N, n_turns - 1))
            my_perm = my_noise.argsort(axis=2)
            my_cards = np.take_along_axis(my_rem_broadcasted, my_perm, axis=2)  # (C, N, T-1)

            # ---- Vectorized Trick Resolution ----
            for t in range(n_turns):
                # Assign each player's card for this turn
                if t == 0:
                    # First turn: our candidate card (same across all N sims)
                    plays_buf[:, :, self.player_idx] = np.broadcast_to(hand_array, (C, N))
                else:
                    # Subsequent turns: draw from our randomized remaining hand
                    plays_buf[:, :, self.player_idx] = my_cards[:, :, t-1]

                plays_buf[:, :, opp_indices[0]] = np.broadcast_to(opp0_cards[:, t], (C, N))
                plays_buf[:, :, opp_indices[1]] = np.broadcast_to(opp1_cards[:, t], (C, N))
                plays_buf[:, :, opp_indices[2]] = np.broadcast_to(opp2_cards[:, t], (C, N))

                # Sort players by card value (6 Nimmt! resolution order)
                order = np.argsort(plays_buf, axis=2)
                sorted_plays = np.take_along_axis(plays_buf, order, axis=2)

                # Process each player in ascending card-value order
                for i in range(4):
                    c = sorted_plays[:, :, i]    # Card values, shape (C, N)
                    p = order[:, :, i]           # Player indices, shape (C, N)

                    c_exp = c[:, :, None]        # Broadcast to (C, N, 1) for row comparison

                    # Compute gap between this card and each row's tail.
                    # Negative/zero gaps mean the card cannot go on that row.
                    np.subtract(c_exp, tails_buf, out=diff_buf)
                    diff_buf[diff_buf <= 0] = 1000  # Sentinel for invalid placements

                    # Target row: the row with the smallest positive gap
                    target_row = np.argmin(diff_buf, axis=2)
                    min_diff = np.min(diff_buf, axis=2)
                    invalid_placement = min_diff == 1000  # Low Card Rule case

                    # Handle Low Card Rule: find the cheapest row to take
                    if np.any(invalid_placement):
                        np.multiply(bullheads_buf, 1000, out=score_buf)
                        score_buf += lengths_buf * 10
                        score_buf += np.arange(4, dtype=np.int32)  # Tiebreak by index

                        alt_target_row = np.argmin(score_buf, axis=2)
                        final_target_row = np.where(invalid_placement, alt_target_row, target_row)
                    else:
                        final_target_row = target_row

                    final_idx = final_target_row[:, :, None]

                    # Determine if placing this card triggers a row take
                    row_len = np.take_along_axis(lengths_buf, final_idx, axis=2).squeeze(-1)
                    take_row = (row_len == 5) | invalid_placement

                    # Compute penalty for taking the row (0 if not taking)
                    row_bh = np.take_along_axis(bullheads_buf, final_idx, axis=2).squeeze(-1)
                    penalty_to_add = np.where(take_row, row_bh, 0)

                    # Accumulate penalty to the correct player index
                    p_exp = p[:, :, None]
                    curr_pen = np.take_along_axis(penalties_buf, p_exp, axis=2)
                    np.put_along_axis(penalties_buf, p_exp, curr_pen + penalty_to_add[:, :, None], axis=2)

                    # Update board state: new tail, new length, new bullheads
                    c_bh = self.bullhead_lookup_array[c]
                    np.put_along_axis(tails_buf, final_idx, c_exp, axis=2)

                    new_len = np.where(take_row, 1, row_len + 1)
                    new_bh_val = np.where(take_row, c_bh, row_bh + c_bh)

                    np.put_along_axis(lengths_buf, final_idx, new_len[:, :, None], axis=2)
                    np.put_along_axis(bullheads_buf, final_idx, new_bh_val[:, :, None], axis=2)

            # Accumulate batch penalties into running statistics
            stats_penalty += penalties_buf[:, :, self.player_idx].sum(axis=1)
            stats_visits += N

        # ---- 5. Action Selection ----
        best_idx = np.argmin(stats_penalty / np.maximum(1, stats_visits))
        return hand[best_idx]
