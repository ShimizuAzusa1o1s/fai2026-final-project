"""
UCB-Guided Monte Carlo Agent with Rational Random Forest Rollouts.

This module implements the final, highest-quality agent architecture for
6 Nimmt!, combining Upper Confidence Bound (UCB) action selection at the
root node with a trained Random Forest opponent model during rollouts.

Algorithm:
    1. **Action Pruning** — Evaluate the full hand using the 115-dim RF model
       and retain only the Top 4 candidates, focusing compute on the most
       promising plays.
    2. **UCB Selection** — At each iteration, choose the candidate that
       maximises the UCB1 score: ``-avg_penalty + C * sqrt(ln(N) / n_i)``.
       Unvisited candidates are always selected first.
    3. **Batch Simulation** — Simulate ``batch_size`` independent random
       worlds simultaneously for the selected candidate to amortise Python
       overhead.
    4. **Rational Rollouts** — For the first ``rf_depth`` tricks, opponents
       play according to the RF model (with ε-greedy exploration).  Remaining
       tricks fall back to fast uniform random play.
    5. **Accumulate & Repeat** — Results are folded back into UCB statistics
       and the loop repeats until the time budget is exhausted.

Characteristics:
    - **Action Pruning**: RF model narrows search to Top 4 candidates.
    - **UCB Exploration**: ``C = 10.0`` (configurable); adapts compute
      allocation to empirically bad candidates immediately.
    - **Rational Rollouts**: RF opponents play consistently, reducing outcome
      variance and accelerating UCB convergence.
    - **Batch Size**: 8 worlds per UCB step (configurable); amortises feature
      extraction overhead over multiple simulations.
    - **ε-Greedy**: 10 % random action during RF rollout depths to prevent
      deterministic opponent simulation.

See Also:
    ``rf_flat_mc.py`` — RF-pruned FlatMC without UCB (simpler, faster).
    ``flat_mc.py``    — Pure-Python unbiased baseline.
"""

import time
import math
import random
import os
try:
    import numpy as np
except ImportError:
    np = None

from src.players.b12705048.core.features import extract_features


class UCB_RF_MCo1:
    """
    UCB Monte Carlo agent with Rational Random Forest rollouts.

    Integrates Upper Confidence Bound tree-node selection with a pre-trained
    Random Forest opponent model to produce an agent that is simultaneously
    accurate (RF forces rational play) and precise (UCB focuses compute on
    the best candidates).

    Attributes:
        player_idx (int): This agent's seat index (0–3).
        time_limit (float): Wall-clock budget in seconds per ``action()`` call.
        ucb_c (float): UCB1 exploration constant.
        rf_depth (int): Number of tricks to use RF opponents before falling
            back to pure random play.
        batch_size (int): Number of parallel worlds simulated per UCB step.
        total_cards (set[int]): The full card universe {1, ..., 104}.
        bullhead_lookup (tuple[int]): O(1) bullhead penalty lookup table.
        rf_model (dict | None): Loaded NumPy RF model, or ``None`` if the
            model file is missing or NumPy is unavailable.
    """

    def __init__(self, player_idx, ucb_c=10.0, rf_depth=3, batch_size=64):
        """
        Initialize the UCB + RF Monte Carlo player.

        Args:
            player_idx (int): The player's seat index in the game (0–3).
            ucb_c (float): Exploration constant for UCB1 (default 10.0).
            rf_depth (int): Number of tricks using RF opponents before random
                fallback (default 3).
            batch_size (int): Number of parallel worlds per UCB step
                (default 8).
        """
        self.player_idx = player_idx
        self.time_limit = 0.90
        self.total_cards = set(range(1, 105))

        self.ucb_c = float(ucb_c)
        self.rf_depth = int(rf_depth)
        self.batch_size = int(batch_size)

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
        self.bullhead_lookup = np.array(bullheads, dtype=np.int32)

        # Try to load the RF model
        self.rf_model = None
        if np is not None:
            model_path = os.path.join(os.path.dirname(__file__), "rf_model.npz")
            if os.path.exists(model_path):
                self.rf_model = dict(np.load(model_path))

    def _predict_proba(self, X):
        """
        Evaluate batched feature matrix using the loaded Random Forest model.

        Performs fully vectorised tree traversal using NumPy indexing to
        compute class probabilities for each sample without scikit-learn.

        Args:
            X (np.ndarray): Batched feature matrix of shape ``(batch, 115)``.

        Returns:
            np.ndarray: Probability distribution over the 10 sorted hand
                slots, shape ``(batch, 10)``.  Returns a uniform distribution
                if the model is not loaded.
        """
        if self.rf_model is None:
            batch_size = X.shape[0]
            return np.ones((batch_size, 10), dtype=np.float32) / 10.0

        n_estimators = self.rf_model['n_estimators'][0]
        batch_size = X.shape[0]

        children_left = self.rf_model['children_left']
        children_right = self.rf_model['children_right']
        feature = self.rf_model['feature']
        threshold = self.rf_model['threshold']
        value = self.rf_model['value']

        node_indices = np.zeros((batch_size, n_estimators), dtype=np.int32)
        is_leaf = np.zeros((batch_size, n_estimators), dtype=bool)

        # Traverse all trees simultaneously until every sample reaches a leaf
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
        """
        Select the best card using UCB-guided Monte Carlo with RF rollouts.

        Prunes the action space with the RF model, then iteratively selects
        the most promising candidate via UCB1 and simulates a batch of worlds
        until the time budget is exhausted.

        Args:
            hand (list[int]): Cards currently available to play.
            history (dict | list): Game state from the engine.

        Returns:
            int: The card value with the best UCB-empirical expected penalty.
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

        h_scores = history.get('scores', [0, 0, 0, 0]) if isinstance(history, dict) else [0, 0, 0, 0]
        h_round = history.get('round', 0) if isinstance(history, dict) else 0
        h_history_matrix = history.get('history_matrix', []) if isinstance(history, dict) else []
        h_score_history = history.get('score_history', []) if isinstance(history, dict) else []
        h_board_history = history.get('board_history', []) if isinstance(history, dict) else []

        # ---- 2. Action Pruning: Keep Top 4 ----
        candidates = hand
        if self.rf_model is not None and len(candidates) > 4:
            features_115 = extract_features(
                board=board,
                hand=hand,
                unseen=set(unseen_cards),
                scores=h_scores,
                player_idx=self.player_idx,
                round_num=h_round,
                history_matrix=h_history_matrix,
                score_history=h_score_history,
                board_history=h_board_history,
            )[:115]
            X_batch = np.zeros((1, 115), dtype=np.float32)
            X_batch[0] = features_115
            probas = self._predict_proba(X_batch)[0]

            sorted_hand = sorted(hand)
            valid_len = len(sorted_hand)
            p = probas.copy()
            p[valid_len:] = -1.0

            top4_idx = np.argsort(p)[-4:]
            candidates = [sorted_hand[idx] for idx in top4_idx]

        stats_penalty = {c: 0.0 for c in candidates}
        stats_visits = {c: 0 for c in candidates}
        total_visits = 0

        opp_indices = [i for i in range(4) if i != self.player_idx]
        orig_row_bullheads = [sum(self.bullhead_lookup[c] for c in row) for row in board]

        # ---- 3. UCB Simulation Loop ----
        while time.perf_counter() - start_time < self.time_limit:

            # UCB1 Selection: pick the candidate with the highest UCB score.
            # Unvisited candidates are always selected first.
            best_candidate = None
            best_ucb = float('-inf')

            for c in candidates:
                if stats_visits[c] == 0:
                    best_candidate = c
                    break

                avg_penalty = stats_penalty[c] / stats_visits[c]
                # UCB for minimisation: negate the penalty then add bonus
                ucb_score = -avg_penalty + self.ucb_c * math.sqrt(
                    math.log(total_visits) / stats_visits[c]
                )
                if ucb_score > best_ucb:
                    best_ucb = ucb_score
                    best_candidate = c

            if best_candidate is None:
                best_candidate = candidates[0]

            # ---- Phase 1: Generate batch_size worlds for the selected arm ----
            games = []
            for _ in range(self.batch_size):
                random.shuffle(unseen_cards)

                sim_board = [row[:] for row in board]
                sim_row_bullheads = orig_row_bullheads[:]

                sim_hands = [None] * 4
                sim_hands[opp_indices[0]] = unseen_cards[0:n_turns]
                sim_hands[opp_indices[1]] = unseen_cards[n_turns:2*n_turns]
                sim_hands[opp_indices[2]] = unseen_cards[2*n_turns:3*n_turns]

                my_sim_hand = [c for c in hand if c != best_candidate]
                random.shuffle(my_sim_hand)
                sim_hands[self.player_idx] = my_sim_hand

                g_history_matrix = [r[:] for r in h_history_matrix]
                g_board_history = [[r[:] for r in b] for b in h_board_history]
                g_scores = h_scores[:]
                g_score_history = [s[:] for s in h_score_history]

                penalties = [0.0, 0.0, 0.0, 0.0]

                # Resolve the first trick using our candidate card
                pending_actions = [(best_candidate, self.player_idx)]
                for opp_idx in opp_indices:
                    pending_actions.append((sim_hands[opp_idx].pop(), opp_idx))

                self._resolve_trick(sim_board, sim_row_bullheads, pending_actions, penalties)

                # Update tracking history for RF feature extraction
                round_actions = [0] * 4
                for card, p_idx in pending_actions:
                    round_actions[p_idx] = card
                g_history_matrix.append(round_actions)
                g_board_history.append([r[:] for r in sim_board])
                for p in range(4):
                    g_scores[p] = h_scores[p] + penalties[p]
                g_score_history.append(g_scores[:])

                games.append({
                    'board': sim_board,
                    'row_bullheads': sim_row_bullheads,
                    'hands': sim_hands,
                    'penalties': penalties,
                    'history_matrix': g_history_matrix,
                    'board_history': g_board_history,
                    'score_history': g_score_history
                })

            # ---- Phase 2: Mixed Rollout (RF then random) ----
            for depth in range(n_turns - 1):
                if self.rf_model is not None and depth < self.rf_depth:
                    # RF-guided depths: batch all players across all games
                    X_batch = np.zeros((len(games) * 4, 115), dtype=np.float32)
                    sim_round = h_round + 1 + depth

                    for g_idx, game in enumerate(games):
                        g_unseen = set()
                        for p_idx in range(4):
                            g_unseen.update(game['hands'][p_idx])

                        for i in range(4):
                            idx = g_idx * 4 + i
                            p_unseen = g_unseen - set(game['hands'][i])
                            current_scores = [h_scores[p] + game['penalties'][p] for p in range(4)]

                            features = extract_features(
                                board=game['board'],
                                hand=game['hands'][i],
                                unseen=p_unseen,
                                scores=current_scores,
                                player_idx=i,
                                round_num=sim_round,
                                history_matrix=game['history_matrix'],
                                score_history=game['score_history'],
                                board_history=game['board_history']
                            )[:115]
                            X_batch[idx] = features

                    probas_batch = self._predict_proba(X_batch)

                    for g_idx, game in enumerate(games):
                        pending_actions = []
                        for i in range(4):
                            idx = g_idx * 4 + i
                            valid_cards = game['hands'][i]
                            sorted_hand_sim = sorted(valid_cards)

                            p = probas_batch[idx].copy()
                            valid_len = len(sorted_hand_sim)
                            p[valid_len:] = -1.0

                            # ε-Greedy (ε=0.10): 10 % chance of random card
                            # to diversify rollout distribution.
                            if random.random() < 0.1:
                                card = random.choice(sorted_hand_sim)
                            else:
                                best_idx = int(np.argmax(p))
                                card = sorted_hand_sim[best_idx]

                            game['hands'][i].remove(card)
                            pending_actions.append((card, i))

                        self._resolve_trick(game['board'], game['row_bullheads'], pending_actions, game['penalties'])

                        # Update trick history for subsequent RF feature extraction
                        round_actions = [0] * 4
                        for card, p_idx in pending_actions:
                            round_actions[p_idx] = card
                        game['history_matrix'].append(round_actions)
                        game['board_history'].append([r[:] for r in game['board']])
                        current_scores = [h_scores[p] + game['penalties'][p] for p in range(4)]
                        game['score_history'].append(current_scores)

                else:
                    # Depth >= rf_depth: pure uniform random play.
                    # Convert to NumPy SoA and resolve all remaining tricks instantly.
                    if depth == self.rf_depth:
                        actual_batch_size = len(games)
                        rem_turns = n_turns - 1 - depth
                        
                        if rem_turns > 0:
                            b_tails = np.array([[game['board'][r][-1] for r in range(4)] for game in games], dtype=np.int32)
                            b_lengths = np.array([[len(game['board'][r]) for r in range(4)] for game in games], dtype=np.int32)
                            b_rbulls = np.array([game['row_bullheads'] for game in games], dtype=np.int32)
                            b_penalties = np.zeros((actual_batch_size, 4), dtype=np.int32)
                            
                            hands_array = np.zeros((actual_batch_size, 4, rem_turns), dtype=np.int32)
                            for g_idx, game in enumerate(games):
                                for p_idx in range(4):
                                    rem_hand = game['hands'][p_idx][:]
                                    random.shuffle(rem_hand)
                                    hands_array[g_idx, p_idx, :] = rem_hand
                                    
                            for t in range(rem_turns):
                                played_cards = hands_array[:, :, t]
                                sort_idx = np.argsort(played_cards, axis=1)
                                sorted_cards = np.take_along_axis(played_cards, sort_idx, axis=1)
                                sorted_players = sort_idx
                                
                                for i in range(4):
                                    current_cards = sorted_cards[:, i]
                                    current_players = sorted_players[:, i]

                                    valid = np.where(current_cards[:, None] > b_tails, b_tails, -1)
                                    target_rows = np.argmax(valid, axis=1)
                                    invalid_mask = np.max(valid, axis=1) == -1

                                    scores = b_rbulls * 1000 + b_lengths * 10 + np.arange(4)
                                    min_rows = np.argmin(scores, axis=1)
                                    target_rows = np.where(invalid_mask, min_rows, target_rows)

                                    b_idx = np.arange(actual_batch_size)
                                    target_lengths = b_lengths[b_idx, target_rows]
                                    target_bullheads = b_rbulls[b_idx, target_rows]

                                    penalty_condition = invalid_mask | (target_lengths == 5)
                                    normal_cond = ~penalty_condition
                                    card_bulls = self.bullhead_lookup[current_cards]

                                    if np.any(penalty_condition):
                                        pc = penalty_condition
                                        b_pc = b_idx[pc]
                                        p_players = current_players[pc]
                                        b_penalties[b_pc, p_players] += target_bullheads[pc]
                                        b_lengths[b_pc, target_rows[pc]] = 1
                                        b_tails[b_pc, target_rows[pc]] = current_cards[pc]
                                        b_rbulls[b_pc, target_rows[pc]] = card_bulls[pc]

                                    if np.any(normal_cond):
                                        nc = normal_cond
                                        b_nc = b_idx[nc]
                                        b_lengths[b_nc, target_rows[nc]] += 1
                                        b_tails[b_nc, target_rows[nc]] = current_cards[nc]
                                        b_rbulls[b_nc, target_rows[nc]] += card_bulls[nc]
                            
                            for g_idx, game in enumerate(games):
                                for p_idx in range(4):
                                    game['penalties'][p_idx] += int(b_penalties[g_idx, p_idx])
                        break

            # ---- Phase 3: Accumulate Results ----
            for game in games:
                stats_penalty[best_candidate] += game['penalties'][self.player_idx]
                stats_visits[best_candidate] += 1
                total_visits += 1

        # ---- 4. Action Selection ----
        # Return the candidate with the best empirical average penalty.
        best_card = min(
            candidates,
            key=lambda k: stats_penalty[k] / max(1, stats_visits[k])
        )
        return best_card

    def _resolve_trick(self, board, row_bullheads, pending_actions, penalties):
        """
        Resolve a single trick according to 6 Nimmt! placement rules.

        Cards are sorted by value (lowest first) and placed sequentially.
        Modifies ``board``, ``row_bullheads``, and ``penalties`` in-place.

        Args:
            board (list[list[int]]): Current board rows.
            row_bullheads (list[int]): Running bullhead totals per row.
            pending_actions (list[tuple[int, int]]): (card, player_idx) pairs.
            penalties (list[float]): Per-player accumulated penalties.
        """
        pending_actions.sort(key=lambda x: x[0])

        for card, player_idx in pending_actions:
            target_row = -1
            max_val = -1

            # Find the row whose tail is the largest value below this card
            for r in range(4):
                val = board[r][-1]
                if val < card and val > max_val:
                    max_val = val
                    target_row = r

            if target_row != -1:
                if len(board[target_row]) == 5:
                    # 6th-card rule: player takes the entire row
                    penalties[player_idx] += row_bullheads[target_row]
                    board[target_row] = [card]
                    row_bullheads[target_row] = self.bullhead_lookup[card]
                else:
                    board[target_row].append(card)
                    row_bullheads[target_row] += self.bullhead_lookup[card]
            else:
                # Low Card Rule: take the row with the lowest penalty
                # Tiebreak: lowest bullheads → shortest row → smallest index
                min_score = 100000
                target_row = -1
                for r in range(4):
                    score = row_bullheads[r] * 1000 + len(board[r]) * 10 + r
                    if score < min_score:
                        min_score = score
                        target_row = r

                penalties[player_idx] += row_bullheads[target_row]
                board[target_row] = [card]
                row_bullheads[target_row] = self.bullhead_lookup[card]
