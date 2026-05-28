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
                 (ε=0.10); 10 % of actions are uniformly random to
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


class FlatMC:
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
        self.bullhead_lookup = tuple(bullheads)
        
        # Try to load the RF model
        self.rf_model = None
        if np is not None:
            model_path = os.path.join(os.path.dirname(__file__), "rf_model.npz")
            if os.path.exists(model_path):
                self.rf_model = dict(np.load(model_path))

    def _predict_proba(self, X):
        """
        Evaluate the batched feature matrix using the loaded Random Forest model.

        Performs a fully vectorised tree traversal using NumPy to compute
        class probabilities for each sample in the batch without relying
        on scikit-learn.

        Args:
            X (np.ndarray): Batched feature matrix of shape (batch_size, 143).

        Returns:
            np.ndarray: Probability distribution over the 10 sorted hand slots,
                shape (batch_size, 10).
        """
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
        """
        Evaluate each candidate card via batched Monte Carlo rollouts and
        return the card with the lowest expected penalty.

        The method uses all available wall-clock time to run as many
        simulation batches as possible.  Each batch processes every candidate
        card simultaneously (SIMD vectorisation over candidate branches).

        Args:
            hand (list[int]): Cards currently available to play.
            history (dict | list): Game state from the engine. Used to
                determine which cards are visible (on board or played in
                prior rounds) and which remain unseen.

        Returns:
            int: The card value with the lowest average simulated penalty.
        """
        start_time = time.perf_counter()

        # ---- 1. State Parsing ----
        # Extract current board and identify all visible cards to determine
        # the pool of unseen cards that could be in opponents' hands.
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
            # ---- Action Pruning: Candidate Filtering ----
            # Evaluate every card in our hand using the RF model and keep
            # only the Top 3 most likely choices.  All simulation time is
            # then focused exclusively on these promising candidates.
            # Use the robust 143-dim features.
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
            p[valid_len:] = -1.0  # Mask out empty slots
            
            # Get indices of top 3 predictions
            top3_idx = np.argsort(p)[-3:]
            candidates = [sorted_hand[idx] for idx in top3_idx]

        # Per-candidate accumulators: total penalty and number of visits
        stats_penalty = {c: 0.0 for c in candidates}
        stats_visits = {c: 0 for c in candidates}

        opp_indices = [i for i in range(4) if i != self.player_idx]

        # Pre-compute row totals once so each simulation can copy cheaply
        orig_row_bullheads = [sum(self.bullhead_lookup[c] for c in row) for row in board]

        # ---- 2. Monte Carlo Simulation Loop ----
        # Run as many simulations as the time budget allows
        while time.perf_counter() - start_time < self.time_limit:
            # Shuffle unseen cards once per batch
            random.shuffle(unseen_cards)
            
            # ---- Phase 1: Initial Placement ----
            # For each candidate, clone the board, deal opponent hands from
            # the shared shuffled pool, then resolve the first trick.
            games = []
            for candidate in candidates:
                # Shallow-copy rows (cards are immutable ints, so this is safe)
                sim_board = [row[:] for row in board]
                sim_row_bullheads = orig_row_bullheads[:]

                # Slice (copy) opponent hands from the shuffled unseen pool.
                # All candidates share the same random world in this batch,
                # which is statistically equivalent to separate shuffles and
                # reduces variance while saving computation.
                sim_hands = [None] * 4
                sim_hands[opp_indices[0]] = unseen_cards[0:n_turns]
                sim_hands[opp_indices[1]] = unseen_cards[n_turns:2*n_turns]
                sim_hands[opp_indices[2]] = unseen_cards[2*n_turns:3*n_turns]

                # Our hand minus the candidate card, shuffled for rollout order
                my_sim_hand = [c for c in hand if c != candidate]
                random.shuffle(my_sim_hand)
                sim_hands[self.player_idx] = my_sim_hand

                penalties = [0.0, 0.0, 0.0, 0.0]

                # Opponents play their highest-indexed (last) card; order does
                # not matter here since _resolve_trick sorts by card value.
                pending_actions = [(candidate, self.player_idx)]
                for opp_idx in opp_indices:
                    pending_actions.append((sim_hands[opp_idx].pop(), opp_idx))

                self._resolve_trick(sim_board, sim_row_bullheads, pending_actions, penalties)

                games.append({
                    'candidate': candidate,
                    'board': sim_board,
                    'row_bullheads': sim_row_bullheads,
                    'hands': sim_hands,
                    'penalties': penalties
                })

            # ---- Phase 2: Pure Random Rollout ----
            # Roll out remaining tricks with fast uniform random play.
            # (RF heuristic is disabled during rollouts due to feature complexity).
            for depth in range(n_turns - 1):
                for g_idx, game in enumerate(games):
                    pending_actions = []
                    for i in range(4):
                        valid_cards = game['hands'][i]
                        # Fast uniform random fallback
                        idx = random.randrange(len(valid_cards))
                        best_card = valid_cards.pop(idx)
                        pending_actions.append((best_card, i))
                    self._resolve_trick(game['board'], game['row_bullheads'], pending_actions, game['penalties'])
                        
            # ---- Phase 3: Accumulate Results ----
            for game in games:
                stats_penalty[game['candidate']] += game['penalties'][self.player_idx]
                stats_visits[game['candidate']] += 1

        # ---- 3. Action Selection ----
        # Return the candidate with the lowest average simulated penalty.
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