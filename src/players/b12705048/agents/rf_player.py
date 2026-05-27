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
    Uses the Deep CFR normalized feature format defined in ``features.py``.
"""

import os
import numpy as np
from src.players.b12705048.core.features import extract_features


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

        # Try to load the pre-trained NumPy Random Forest model
        self.rf_model = None
        model_path = os.path.join(os.path.dirname(__file__), "rf_model.npz")
        if os.path.exists(model_path):
            self.rf_model = dict(np.load(model_path))
        else:
            print(f"WARNING: RF Model not found at {model_path}. Player will play randomly.")



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

        h_scores = history.get('scores', [0,0,0,0]) if isinstance(history, dict) else [0,0,0,0]
        h_round = history.get('round', 0) if isinstance(history, dict) else 0
        h_history_matrix = history.get('history_matrix', []) if isinstance(history, dict) else []
        h_score_history = history.get('score_history', []) if isinstance(history, dict) else []
        h_board_history = history.get('board_history', []) if isinstance(history, dict) else []

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
        )
        
        X = np.zeros((1, 115), dtype=np.float32)
        X[0] = features_115

        probas = self._predict_proba(X)

        sorted_hand = sorted(hand)
        p = probas[0].copy()

        # Mask out padding slots beyond the actual hand size
        valid_len = len(sorted_hand)
        p[valid_len:] = -1.0

        best_idx = int(np.argmax(p))
        best_card = sorted_hand[best_idx]

        return best_card
