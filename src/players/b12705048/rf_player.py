"""
Pure Random Forest Player Module.

This module is primarily a **benchmarking tool** used to evaluate the
standalone strength of the Random Forest heuristic, independent of the
Monte Carlo search layer in ``flat_mc.py``.

The agent uses the same pre-trained ``.npz`` model file (50 trees, 115
features) and the same inference code as ``FlatMC``, so their heuristic
quality is directly comparable.  Because there is no search, each
``action()`` call is essentially instant.

Feature Layout (115 dimensions):
    [  0– 11]  Board: length, top card, penalty for each of 4 rows
               (sorted ascending by top card).
    [ 12]      Board max row bullheads.
    [ 13]      Board min row bullheads.
    [ 14]      Turn number (hand size).
    [ 15–114]  Card Features: 10 slots (sorted hand), each with 10 values:
               1. Card Value
               2. Card Bullheads
               3. Is Under-Board (1 or 0)
               4. Distance to Target Row
               5. Unseen Cards in Gap
               6. Distance to Next Closest Row
               7. Target Row Length
               8. Target Row Bullheads
               9. Cheapest Available Row Bullheads
               10. Difference to Lowest Tail
"""

import os
import numpy as np


class RFPlayer:
    """
    Pure Random Forest agent for 6 Nimmt! (no Monte Carlo search).

    This agent does NOT perform any simulations or tree search.  Instead,
    it uses the trained Random Forest (exported to NumPy) to instantly
    predict the best card to play given the current board state and hand.

    The model predicts a probability distribution over the 10 sorted hand
    slots and plays the card at the slot with the highest probability.

    Attributes:
        player_idx (int): This agent's seat index (0–3).
        bullhead_lookup (tuple[int]): O(1) bullhead penalty lookup table.
        rf_model (dict | None): Loaded NumPy RF model arrays, or ``None``
            if the model file is missing.
    """

    def __init__(self, player_idx):
        """
        Initialize the RF Player.

        Args:
            player_idx (int): The player's seat index in the game (0–3).
        """
        self.player_idx = player_idx
        self.bullhead_lookup = self._init_bullheads()

        # Try to load the pre-trained NumPy Random Forest model
        self.rf_model = None
        model_path = os.path.join(os.path.dirname(__file__), "rf_model.npz")
        if os.path.exists(model_path):
            self.rf_model = dict(np.load(model_path))
        else:
            print(f"WARNING: RF Model not found at {model_path}. Player will play randomly.")

    def _init_bullheads(self):
        """
        Build the bullhead penalty lookup table.

        Returns:
            tuple[int]: 105-element tuple where index ``c`` gives the
                bullhead penalty for card ``c``.
        """
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
        return tuple(bullheads)

    def _extract_features(self, board, hand, unseen_cards):
        """
        Extract a 115-dimensional feature vector from a game state.

        This layout must stay in exact sync with ``extract_features`` in
        ``train_rf_model.py`` and ``_extract_features_multi`` in
        ``flat_mc.py``.

        Args:
            board (list[list[int]]): Current board rows.
            hand (list[int]): Current player hand.
            unseen_cards (list[int]): Cards unplayed and not in hand.

        Returns:
            np.ndarray: Feature matrix of shape ``(1, 115)``.
        """
        sorted_board = sorted(board, key=lambda row: row[-1] if row else 0)

        board_features = []
        min_bullheads = 1000
        max_bullheads = -1

        row_lengths = []
        row_tops = []
        row_bullheads = []

        for row in sorted_board:
            if row:
                length = len(row)
                top = row[-1]
                b_heads = sum(self.bullhead_lookup[c] for c in row)
            else:
                length = 0
                top = 0
                b_heads = 0

            row_lengths.append(length)
            row_tops.append(top)
            row_bullheads.append(b_heads)

            board_features.extend([length, top, b_heads])
            if b_heads < min_bullheads:
                min_bullheads = b_heads
            if b_heads > max_bullheads:
                max_bullheads = b_heads

        if min_bullheads == 1000:
            min_bullheads = 0
        if max_bullheads == -1:
            max_bullheads = 0

        board_features.extend([max_bullheads, min_bullheads])
        turn_number = len(hand)
        board_features.append(turn_number)

        card_features = []
        sorted_hand = sorted(hand)
        unseen_set = set(unseen_cards)

        for i in range(10):
            if i < len(sorted_hand):
                card = sorted_hand[i]
                c_bullheads = self.bullhead_lookup[card]

                # Find the target row: the one whose tail is the largest value
                # strictly below this card.
                target_row_idx = -1
                max_val = -1
                for r in range(4):
                    val = row_tops[r]
                    if val < card and val > max_val:
                        max_val = val
                        target_row_idx = r

                if target_row_idx != -1:
                    is_under_board = 0
                    target_tail = row_tops[target_row_idx]
                    dist_to_target = card - target_tail

                    # Count unseen cards between the target row tail and this card;
                    # a higher count means higher interception risk.
                    unseen_in_gap = sum(1 for uc in unseen_set if target_tail < uc < card)

                    # Second-closest row (fallback interception target)
                    next_closest_row_idx = -1
                    max_val2 = -1
                    for r in range(4):
                        val = row_tops[r]
                        if val < card and val > max_val2 and r != target_row_idx:
                            max_val2 = val
                            next_closest_row_idx = r

                    if next_closest_row_idx != -1:
                        dist_to_next = card - row_tops[next_closest_row_idx]
                    else:
                        dist_to_next = 1000  # Sentinel: no second row found

                    t_length = row_lengths[target_row_idx]
                    t_bulls = row_bullheads[target_row_idx]
                    cheap_avail = 0
                    diff_to_lowest = 0
                else:
                    # Card is under-board: it will trigger the Low Card Rule.
                    is_under_board = 1
                    dist_to_target = 1000
                    unseen_in_gap = 1000
                    dist_to_next = 1000
                    t_length = 0
                    t_bulls = 0

                    # Find the cheapest row to take (lowest penalty)
                    cheap_avail = 1000
                    for r in range(4):
                        if row_bullheads[r] < cheap_avail:
                            cheap_avail = row_bullheads[r]
                    if cheap_avail == 1000:
                        cheap_avail = 0

                    # How far below the lowest row tail is this card?
                    min_tail = min([top for top in row_tops if top > 0] + [1000])
                    if min_tail != 1000:
                        diff_to_lowest = min_tail - card
                    else:
                        diff_to_lowest = 0

                card_features.extend([
                    card, c_bullheads, is_under_board, dist_to_target,
                    unseen_in_gap, dist_to_next, t_length, t_bulls,
                    cheap_avail, diff_to_lowest
                ])
            else:
                # Pad empty hand slots with zeros
                card_features.extend([0] * 10)

        X = np.zeros((1, 115), dtype=np.float32)
        X[0, :15] = board_features
        X[0, 15:115] = card_features
        return X

    def _predict_proba(self, X):
        """
        Evaluate the feature matrix using the loaded Random Forest model.

        Performs a fully vectorised tree traversal using NumPy to compute
        class probabilities without relying on scikit-learn.

        Args:
            X (np.ndarray): Feature matrix of shape ``(1, 115)``.

        Returns:
            np.ndarray: Probability distribution over the 10 sorted hand slots,
                shape ``(1, 10)``.  Returns a uniform distribution if the
                model is not loaded.
        """
        if self.rf_model is None:
            return np.ones((1, 10), dtype=np.float32) / 10.0

        n_estimators = self.rf_model['n_estimators'][0]

        children_left = self.rf_model['children_left']
        children_right = self.rf_model['children_right']
        feature = self.rf_model['feature']
        threshold = self.rf_model['threshold']
        value = self.rf_model['value']

        node_indices = np.zeros((1, n_estimators), dtype=np.int32)
        is_leaf = np.zeros((1, n_estimators), dtype=bool)

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

        # Average leaf probabilities across all trees
        p = value[np.arange(n_estimators), node_indices[0, :], :]
        probas = np.sum(p, axis=0) / n_estimators
        return probas.reshape(1, 10)

    def action(self, hand, history):
        """
        Select the card predicted as most likely by the RF model.

        Extracts features from the current game state, queries the RF
        model, and returns the card in the sorted hand corresponding to
        the highest-probability slot.

        Args:
            hand (list[int]): Cards currently available to play.
            history (dict | list): Game state from the engine.  Used to
                determine which cards are visible (on board or played in
                prior rounds) and which remain unseen.

        Returns:
            int: The card value selected by the RF model.
        """
        # ---- State Parsing ----
        # Reconstruct the set of visible cards from the engine history to
        # derive the pool of unseen cards used by the feature extractor.
        if isinstance(history, dict):
            board = history.get('board', [])
            visible_cards = set()
            for row in board:
                visible_cards.update(row)
            for past_round in history.get('history_matrix', []):
                visible_cards.update(past_round)
            for board_hist in history.get('board_history', []):
                for row in board_hist:
                    visible_cards.update(row)
        else:
            board = history[-1]
            visible_cards = set()
            for row in board:
                visible_cards.update(row)

        unseen_cards = list(set(range(1, 105)) - visible_cards - set(hand))

        # ---- Inference ----
        features = self._extract_features(board, hand, unseen_cards)
        probas = self._predict_proba(features)

        sorted_hand = sorted(hand)
        p = probas[0].copy()

        # Mask out padding slots beyond the actual hand size
        valid_len = len(sorted_hand)
        p[valid_len:] = -1.0

        best_idx = int(np.argmax(p))
        best_card = sorted_hand[best_idx]

        return best_card
