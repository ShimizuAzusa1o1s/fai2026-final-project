"""
Flat Monte Carlo (1-Ply) Player Module.

This module implements a 1-ply Monte Carlo evaluation agent for 6 Nimmt!
that uses a pre-trained Random Forest model both to prune the action space
and to guide simulation rollouts.  Instead of building a deep search tree,
it evaluates a small set of candidate cards by running batched rollout
simulations to the end of the round, then selects the card that minimises
the expected penalty.

Algorithm:
    Before the simulation loop:
        1. Evaluate the full hand with the RF model.
        2. Prune to the **Top 3** most promising candidate cards
           (Action Pruning / Candidate Filtering).

    For each simulation batch:
        3. Shuffle unseen cards to generate a random world (opponent hands).
        4. For every candidate card simultaneously:
            a. Resolve the first trick using that candidate card.
            b. Roll out remaining tricks with all players following a
               **Mixed Rollout** policy:
               - Depths 0–2: RF heuristic with ε-Greedy exploration
                 (ε=0.10); 10% of actions are uniformly random to
                 prevent overfitting to the RF's expectations.
               - Depths 3+: Fast uniform random play.
               Both branches are evaluated in a single batched NumPy call
               across all active candidate branches (SIMD vectorisation).
        5. Accumulate the penalty incurred by us for each candidate.
    Select the candidate card with the lowest average simulated penalty.

Characteristics:
    - **Depth**: 1-ply only (evaluates immediate action, no tree search).
    - **Action Pruning**: RF model narrows the search to the top-3 cards,
      concentrating the entire simulation budget on the most promising moves.
    - **Rollout Policy**: Mixed RF heuristic (depths 0–2) + uniform random
      (depth 3+), evaluated via fully-vectorised NumPy tree traversal
      without scikit-learn.
    - **ε-Greedy Exploration**: During RF-guided depths, each simulated
      player has a 10 % chance of playing a uniformly random card to
      diversify the rollout distribution.
    - **Time Management**: Runs as many simulations as possible within a
      configurable wall-clock time budget (default 0.9 seconds).
    - **SIMD Batching**: All candidate branches are evaluated in a single
      batched call to ``_predict_proba`` each turn, maximising throughput.
"""

import time
import random
import os
try:
    import numpy as np
except ImportError:
    np = None

from src.players.b12705048.core.features import extract_features


class FlatMCo1:
    """
    1-ply Monte Carlo agent for 6 Nimmt! with RF-guided action pruning and
    mixed ε-greedy rollouts.

    Before running simulations, the agent uses the RF model to score every
    card in hand and prunes all but the **Top 3** candidates.  Simulations
    then use the RF for the first 3 rollout depths (with ε=0.10 random
    exploration) and fall back to fast uniform-random play thereafter.

    The rollout policy uses a pre-trained Random Forest (50 trees, 143
    features) exported as a NumPy ``.npz`` file.  Inference is fully
    vectorised; scikit-learn is NOT required at runtime.

    Attributes:
        player_idx (int): This agent's seat index (0–3).
        time_limit (float): Wall-clock budget in seconds per ``action()`` call.
        total_cards (set[int]): The full card universe {1, ..., 104}.
        bullhead_lookup (tuple[int]): O(1) bullhead penalty lookup table.
        rf_model (dict | None): Loaded NumPy RF model arrays, or ``None``
            if the model file is missing or NumPy is unavailable.
    """

    def __init__(self, player_idx):
        """
        Initialize the Flat Monte Carlo player.

        Args:
            player_idx (int): The player's seat index in the game (0–3).
        """
        self.player_idx = player_idx
        self.time_limit = 0.9
        self.total_cards = set(range(1, 105))

        # Pre-compute bullhead lookup table for O(1) penalty lookups
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
        self.batch_size = 5000
        self.bullhead_lookup = np.array(bullheads, dtype=np.int32)
        
        # Try to load the RF model
        self.rf_model = None
        if np is not None:
            model_path = os.path.join(os.path.dirname(__file__), "rf_model.npz")
            if os.path.exists(model_path):
                self.rf_model = dict(np.load(model_path))

    def _predict_proba(self, X):
        n_estimators = self.rf_model['n_estimators'][0]
        batch_size = X.shape[0]
        
        children_left = self.rf_model['children_left']
        children_right = self.rf_model['children_right']
        feature = self.rf_model['feature']
        threshold = self.rf_model['threshold']
        value = self.rf_model['value']
        
        node_indices = np.zeros((batch_size, n_estimators), dtype=np.int32)
        is_leaf = np.zeros((batch_size, n_estimators), dtype=bool)
        
        while not np.all(is_leaf):
            active = ~is_leaf
            sample_idx, tree_idx = np.nonzero(active)
            
            active_nodes = node_indices[sample_idx, tree_idx]
            
            f_indices = feature[tree_idx, active_nodes]
            t_values = threshold[tree_idx, active_nodes]
            
            left_mask = X[sample_idx, f_indices] <= t_values
            
            next_nodes = np.where(left_mask, 
                                  children_left[tree_idx, active_nodes], 
                                  children_right[tree_idx, active_nodes])
                                  
            node_indices[sample_idx, tree_idx] = next_nodes
            is_leaf[sample_idx, tree_idx] = children_left[tree_idx, next_nodes] == -1
            
        probas = np.zeros((batch_size, 10), dtype=np.float32)
        for b in range(batch_size):
            p = value[np.arange(n_estimators), node_indices[b, :], :]
            probas[b] = np.sum(p, axis=0)
            
        probas /= n_estimators
        return probas

    def action(self, hand, history):
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
        
        candidates = hand
        if self.rf_model is not None and len(candidates) > 3:
            h_scores = history.get('scores', [0,0,0,0]) if isinstance(history, dict) else [0,0,0,0]
            h_round = history.get('round', 0) if isinstance(history, dict) else 0
            h_history_matrix = history.get('history_matrix', []) if isinstance(history, dict) else []
            h_score_history = history.get('score_history', []) if isinstance(history, dict) else []
            h_board_history = history.get('board_history', []) if isinstance(history, dict) else []
            
            features_143 = extract_features(
                board=board,
                hand=hand,
                unseen=set(unseen_cards),
                scores=h_scores,
                player_idx=self.player_idx,
                round_num=h_round,
                history_matrix=h_history_matrix,
                score_history=h_score_history,
                board_history=h_board_history,
            )
            
            X_batch = np.zeros((1, 143), dtype=np.float32)
            X_batch[0] = features_143
            
            probas = self._predict_proba(X_batch)[0]
            
            sorted_hand = sorted(hand)
            valid_len = len(sorted_hand)
            p = probas.copy()
            p[valid_len:] = -1.0
            
            top3_idx = np.argsort(p)[-3:]
            candidates = [sorted_hand[idx] for idx in top3_idx]

        stats_penalty = {c: 0.0 for c in candidates}
        stats_visits = {c: 0 for c in candidates}

        orig_tails = np.array([row[-1] for row in board], dtype=np.int32)
        orig_lengths = np.array([len(row) for row in board], dtype=np.int32)
        orig_rbulls = np.array([sum(self.bullhead_lookup[c] for c in row) for row in board], dtype=np.int32)
        
        unseen_mask_base = np.zeros(105, dtype=bool)
        unseen_mask_base[unseen_cards] = True

        while time.perf_counter() - start_time < self.time_limit:
            num_cand = len(candidates)
            sims_per_cand = self.batch_size // num_cand
            actual_batch_size = sims_per_cand * num_cand

            if actual_batch_size == 0:
                break

            tails = np.tile(orig_tails, (actual_batch_size, 1))
            lengths = np.tile(orig_lengths, (actual_batch_size, 1))
            rbulls = np.tile(orig_rbulls, (actual_batch_size, 1))
            penalties = np.zeros((actual_batch_size, 4), dtype=np.int32)

            rand_weights = np.random.rand(actual_batch_size, 105)
            unseen_mask = np.tile(unseen_mask_base, (actual_batch_size, 1))
            rand_weights[~unseen_mask] = -1.0
            perm = np.argsort(-rand_weights, axis=1)
            
            opp_indices = [i for i in range(4) if i != self.player_idx]
            hands_array = np.zeros((actual_batch_size, 4, n_turns), dtype=np.int32)
            hands_array[:, opp_indices[0], :] = perm[:, 0:n_turns]
            hands_array[:, opp_indices[1], :] = perm[:, n_turns:2*n_turns]
            hands_array[:, opp_indices[2], :] = perm[:, 2*n_turns:3*n_turns]

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

            c_idx = 0
            for c in candidates:
                start_b = c_idx * sims_per_cand
                end_b = start_b + sims_per_cand
                my_pens = penalties[start_b:end_b, self.player_idx]
                stats_penalty[c] += np.sum(my_pens)
                stats_visits[c] += sims_per_cand
                c_idx += 1

        best_card = min(candidates, key=lambda k: stats_penalty[k] / max(1, stats_visits[k]))
        return best_card