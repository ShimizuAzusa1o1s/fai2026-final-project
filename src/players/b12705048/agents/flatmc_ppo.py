"""
Flat Monte Carlo (PPO-Guided) Player Module

Algorithm: 
    - SIMD Flat Monte Carlo with PPO-Guided Rollouts.
    - Evaluates sequences of candidate actions by simulating thousands of
      parallel games. Opponent plays are dynamically predicted using the PyTorch MaskablePPO
      policy model instead of uniform random rollouts.

Characteristics:
    - **Depth**: Variable N-ply deterministic search, followed by PPO-guided rollout to the end of the round.
    - **Rollout Policy**: Opponents choose cards based on the trained PPO neural network.
    - **Time Management**: Repeats simulations until the time budget expires.

See Also:
    ``flatmc.py`` — Uniform random rollout baseline.
    ``rl_agent.py`` — Pure PPO inference agent.
"""

import os
import sys
import time
import numpy as np
import itertools

from src.players.b12705048.core.features import extract_features

class FlatMCPPO:
    """
    Agent utilizing SIMD Flat Monte Carlo with PPO Rollouts.

    Attributes:
        player_idx (int): This agent's seat index (0-3).
        n_ply (int): The search depth.
        time_limit (float): Wall-clock budget in seconds.
        total_cards (set[int]): The full card universe {1, ..., 104}.
        batch_size (int): Number of simultaneous simulations per batch.
        bullhead_lookup (np.ndarray): O(1) bullhead penalty lookup array.
        weights (dict | None): The loaded NumPy model weights, or None if not found.
    """

    def __init__(self, player_idx, n_ply=1, model_path="src/players/b12705048/agents/numpy_ppo_weights.npz"):
        """
        Initialize the PPO-Guided Flat Monte Carlo player.

        Args:
            player_idx (int): The player's seat index in the game (0-3).
            n_ply (int): Depth of initial deterministic search sequence.
            model_path (str): Path to the numpy weights file.
        """
        self.player_idx = player_idx
        self.n_ply = n_ply
        self.time_limit = 0.8
        self.total_cards = set(range(1, 105))
        self.batch_size = 2000  # Increased batch size because NumPy inference is fast
        
        self.bullhead_lookup = np.zeros(105, dtype=np.int32)
        for card in range(1, 105):
            if card == 55:
                self.bullhead_lookup[card] = 7
            elif card % 11 == 0:
                self.bullhead_lookup[card] = 5
            elif card % 10 == 0:
                self.bullhead_lookup[card] = 3
            elif card % 5 == 0:
                self.bullhead_lookup[card] = 2
            else:
                self.bullhead_lookup[card] = 1

        # ---- Phase 1: Sub-Agent Initialization ----
        self.weights = None
        if not model_path.endswith('.npz'):
            # Provide fallback to default npz name
            model_path = model_path.replace("stage3_model_final", "numpy_ppo_weights.npz")
            if model_path.endswith('.zip'):
                model_path = model_path.replace('.zip', '.npz')
                
        if os.path.exists(model_path):
            self.weights = np.load(model_path)
            self.fc1_w = self.weights['fc1_w'].T
            self.fc1_b = self.weights['fc1_b']
            self.fc2_w = self.weights['fc2_w'].T
            self.fc2_b = self.weights['fc2_b']
            self.action_w = self.weights['action_w'].T
            self.action_b = self.weights['action_b']
        else:
            print(f"Warning: Numpy model weights not found at {model_path}.")

    def _get_penalties(self, cards, tails, lengths, rbulls):
        """
        Calculates the penalty points for placing cards on given board tails.

        Args:
            cards (np.ndarray): Array of card values being played.
            tails (np.ndarray): Array of the current tail card values for each row.
            lengths (np.ndarray): Array of the current lengths for each row.
            rbulls (np.ndarray): Array of the current accumulated penalties for each row.

        Returns:
            np.ndarray: Calculated penalty points for each card.
        """
        valid = np.where(cards[:, None] > tails[None, :], tails[None, :], -1)
        target_rows = np.argmax(valid, axis=1)
        invalid_mask = np.max(valid, axis=1) == -1
        
        scores = rbulls * 1000 + lengths * 10 + np.arange(4)
        min_row = np.argmin(scores)
        target_rows = np.where(invalid_mask, min_row, target_rows)
        
        pens = np.zeros(len(cards), dtype=np.int32)
        penalty_cond = invalid_mask | (lengths[target_rows] == 5)
        pens[penalty_cond] = rbulls[target_rows[penalty_cond]]
        return pens

    def action(self, hand, history):
        """
        Evaluate candidate cards via batched SoA simulation and return the best card.

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

        actual_n_ply = min(self.n_ply, n_turns)

        # ---- Phase 1: State Parsing ----

        if isinstance(history, dict):
            board = history.get('board', [])
            history_matrix = history.get('history_matrix', [])
            board_history = history.get('board_history', [])
            scores = history.get('scores', [0, 0, 0, 0])
            score_history = history.get('score_history', [])
            round_num = len(history_matrix)
        else:
            board = history[-1]
            history_matrix = []
            board_history = []
            scores = [0, 0, 0, 0]
            score_history = []
            round_num = 0

        visible_cards = set()
        for row in board:
            visible_cards.update(row)
        for past_round in history_matrix:
            visible_cards.update(c for c in past_round if c > 0)
        if board_history:
            for past_b in board_history:
                for row in past_b:
                    visible_cards.update(row)

        unseen_cards = list(self.total_cards - visible_cards - set(hand))
        unseen_cards_arr = np.array(unseen_cards, dtype=np.int32)
        
        unseen_cumcount = np.zeros(106, dtype=np.int32)
        for uc in unseen_cards:
            unseen_cumcount[uc] = 1
        unseen_cumcount = np.cumsum(unseen_cumcount)
        
        candidates = list(itertools.permutations(hand, actual_n_ply))
        num_cand = len(candidates)
        if num_cand == 0: 
            return hand[0]

        stats_penalty = {cand: 0.0 for cand in candidates}
        stats_visits = {cand: 0 for cand in candidates}

        orig_tails = np.array([row[-1] if row else 0 for row in board], dtype=np.int32)
        orig_lengths = np.array([len(row) for row in board], dtype=np.int32)
        orig_rbulls = np.array([sum(self.bullhead_lookup[c] for c in row) for row in board], dtype=np.int32)

        unseen_mask_base = np.zeros(105, dtype=bool)
        unseen_mask_base[unseen_cards] = True

        features_143 = extract_features(
            board=board, hand=hand, unseen=set(unseen_cards),
            scores=scores, player_idx=self.player_idx, round_num=round_num,
            history_matrix=history_matrix, score_history=score_history,
            board_history=board_history
        )
        static_features_np = np.array(features_143[115:143], dtype=np.float32)

        inference_weights = np.ones((4, 105), dtype=np.float32)
        if board_history and history_matrix:
            for k in range(len(history_matrix)):
                past_b = board_history[k]
                p_tails = np.array([r[-1] if r else 0 for r in past_b])
                p_lengths = np.array([len(r) for r in past_b])
                p_rbulls = np.array([sum(self.bullhead_lookup[c] for c in r) for r in past_b])
                
                sim_pens = self._get_penalties(unseen_cards_arr, p_tails, p_lengths, p_rbulls)
                
                for o in range(4):
                    if o == self.player_idx: continue
                    played = history_matrix[k][o]
                    if played <= 0: continue
                    actual_pen = self._get_penalties(np.array([played]), p_tails, p_lengths, p_rbulls)[0]
                    if actual_pen > 0:
                        safe_mask = sim_pens < actual_pen
                        safe_cards = unseen_cards_arr[safe_mask]
                        inference_weights[o, safe_cards] *= 0.05

        # ---- Phase 2: Batch Initialization & Deal ----
        while time.perf_counter() - start_time < self.time_limit:
            sims_per_cand = max(1, self.batch_size // num_cand)
            actual_batch_size = sims_per_cand * num_cand

            tails = np.tile(orig_tails, (actual_batch_size, 1))
            lengths = np.tile(orig_lengths, (actual_batch_size, 1))
            rbulls = np.tile(orig_rbulls, (actual_batch_size, 1))
            penalties = np.zeros((actual_batch_size, 4), dtype=np.int32)

            rand_weights = np.random.rand(actual_batch_size, 4, 105)
            for o in range(4):
                rand_weights[:, o, ~unseen_mask_base] = -1.0
                rand_weights[:, o, :] *= inference_weights[o, :]

            opp_indices = [i for i in range(4) if i != self.player_idx]
            hands_array = np.zeros((actual_batch_size, 4, n_turns), dtype=np.int32)

            for t in range(n_turns):
                for o in opp_indices:
                    chosen = np.argmax(rand_weights[:, o, :], axis=1)
                    hands_array[:, o, t] = chosen
                    for p in range(4):
                        np.put_along_axis(rand_weights[:, p, :], chosen[:, None], -1.0, axis=1)

            c_idx = 0
            for cand_tuple in candidates:
                start_b = c_idx * sims_per_cand
                end_b = start_b + sims_per_cand
                for i, c in enumerate(cand_tuple):
                    hands_array[start_b:end_b, self.player_idx, i] = c
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
                
                # Dynamic NumPy PPO inference
                if self.weights is not None and t >= actual_n_ply:
                    for p_idx_eval in range(4):
                        rem = hands_array[:, p_idx_eval, t:]
                        num_rem = rem.shape[1]
                        if num_rem > 1:
                            sort_idx = np.argsort(tails, axis=1)
                            sorted_tails = np.take_along_axis(tails, sort_idx, axis=1)
                            sorted_lengths = np.take_along_axis(lengths, sort_idx, axis=1)
                            sorted_rbulls = np.take_along_axis(rbulls, sort_idx, axis=1)
                            
                            board_dynamic = np.zeros((actual_batch_size, 15), dtype=np.float32)
                            board_dynamic[:, 0:12:3] = sorted_lengths / 5.0
                            board_dynamic[:, 1:12:3] = sorted_tails / 104.0
                            board_dynamic[:, 2:12:3] = sorted_rbulls / 28.0
                            board_dynamic[:, 12] = np.max(sorted_rbulls, axis=1) / 28.0
                            board_dynamic[:, 13] = np.min(sorted_rbulls, axis=1) / 28.0
                            board_dynamic[:, 14] = num_rem / 10.0
                            
                            static_exp = np.tile(static_features_np, (actual_batch_size, 1))
                            
                            stails = sorted_tails
                            slengths = sorted_lengths
                            srbulls = sorted_rbulls
                            
                            # Build the 100-dim card features (10 slots * 10 dims)
                            c_feat = np.zeros((actual_batch_size, 100), dtype=np.float32)
                            
                            # We only fill the first num_rem slots. The rest are 0.0
                            # To do this efficiently, we iterate over the active slots
                            for slot in range(num_rem):
                                cards = rem[:, slot]
                                base = slot * 10
                                
                                c_feat[:, base + 0] = cards / 104.0
                                c_feat[:, base + 1] = self.bullhead_lookup[cards] / 7.0
                                
                                valid = np.where(cards[:, None] > stails, stails, -1)
                                target_rows = np.argmax(valid, axis=1)
                                invalid_mask = np.max(valid, axis=1) == -1
                                
                                target_tails = stails[np.arange(actual_batch_size), target_rows]
                                dist = cards - target_tails
                                
                                c_feat[:, base + 2] = invalid_mask.astype(np.float32)
                                c_feat[:, base + 3] = np.where(invalid_mask, 1.0, np.minimum(dist, 104.0) / 104.0)
                                
                                gap_counts = unseen_cumcount[np.maximum(0, cards - 1)] - unseen_cumcount[target_tails]
                                c_feat[:, base + 4] = np.where(invalid_mask, 1.0, np.minimum(gap_counts, 100.0) / 100.0)
                                
                                valid_next = valid.copy()
                                valid_next[np.arange(actual_batch_size), target_rows] = -1
                                next_rows = np.argmax(valid_next, axis=1)
                                has_next_mask = np.max(valid_next, axis=1) != -1
                                next_tails = stails[np.arange(actual_batch_size), next_rows]
                                next_dist = cards - next_tails
                                c_feat[:, base + 5] = np.where(invalid_mask | ~has_next_mask, 1.0, np.minimum(next_dist, 104.0) / 104.0)
                                
                                c_feat[:, base + 6] = np.where(invalid_mask, 0.0, slengths[np.arange(actual_batch_size), target_rows] / 5.0)
                                c_feat[:, base + 7] = np.where(invalid_mask, 0.0, srbulls[np.arange(actual_batch_size), target_rows] / 28.0)
                                
                                cheap = np.min(srbulls, axis=1)
                                c_feat[:, base + 8] = np.where(invalid_mask, cheap / 28.0, 0.0)
                                
                                stails_masked = np.where(stails > 0, stails, 1000)
                                min_tails = np.min(stails_masked, axis=1)
                                has_positive = min_tails != 1000
                                c_feat[:, base + 9] = np.where(invalid_mask & has_positive, (min_tails - cards) / 104.0, 0.0)
                                
                            X = np.concatenate([board_dynamic, c_feat, static_exp], axis=1) # (batch_size, 143)
                            
                            # Construct Action Mask (10 slots)
                            act_mask = np.zeros((actual_batch_size, 10), dtype=bool)
                            act_mask[:, :num_rem] = True
                            
                            # Query NumPy
                            if self.weights is not None:
                                h1 = np.tanh(X @ self.fc1_w + self.fc1_b)
                                h2 = np.tanh(h1 @ self.fc2_w + self.fc2_b)
                                logits = h2 @ self.action_w + self.action_b
                                
                                logits[~act_mask] = -np.inf
                                actions = np.argmax(logits, axis=1)
                            else:
                                actions = np.zeros(actual_batch_size, dtype=np.int32)
                                
                            # Reorganize hands_array based on chosen action indices
                            chosen_cards = np.take_along_axis(rem, actions[:, None], axis=1).flatten()
                            old_t = hands_array[:, p_idx_eval, t].copy()
                            hands_array[:, p_idx_eval, t] = chosen_cards
                            np.put_along_axis(hands_array[:, p_idx_eval, t:], actions[:, None], old_t[:, None], axis=1)

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

                    scores_eval = rbulls * 1000 + lengths * 10 + np.arange(4)
                    min_rows = np.argmin(scores_eval, axis=1)
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
            c_idx = 0
            for cand in candidates:
                start_b = c_idx * sims_per_cand
                end_b = start_b + sims_per_cand
                my_pens = penalties[start_b:end_b, self.player_idx]
                stats_penalty[cand] += np.sum(my_pens)
                stats_visits[cand] += sims_per_cand
                c_idx += 1

        total_visits = sum(stats_visits.values())
        if total_visits == 0:
            return hand[0]

        best_cand = min(
            candidates,
            key=lambda cand: stats_penalty[cand] / max(1, stats_visits[cand])
        )
        return best_cand[0]
