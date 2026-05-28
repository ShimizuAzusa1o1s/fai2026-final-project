import time
import os
import numpy as np

from src.players.b12705048.core.features import extract_features

class LGBMRankerAgent:
    """
    Learning to Rank Agent using LightGBM (LambdaMART).
    
    This agent uses a pre-trained LightGBM ranker exported to NumPy arrays
    to score candidate cards in its hand based on 53-dimensional pointwise
    features (state + candidate card). It then deterministically plays the
    card with the highest ranking score.
    """
    
    def __init__(self, player_idx):
        self.player_idx = player_idx
        self.time_limit = 0.9  # Not heavily used since inference is fast
        self.total_cards = set(range(1, 105))
        
        self.lgbm_model = None
        model_path = os.path.join(os.path.dirname(__file__), "lgbm_model.npz")
        if os.path.exists(model_path):
            self.lgbm_model = dict(np.load(model_path))
        else:
            print(f"Warning: Model not found at {model_path}")

    def _predict(self, X):
        """
        Vectorised tree traversal for LightGBM model.
        Args:
            X: np.ndarray of shape (batch_size, 53)
        Returns:
            np.ndarray of shape (batch_size,) containing ranking scores.
        """
        if self.lgbm_model is None:
            return np.zeros(X.shape[0], dtype=np.float32)

        n_estimators = self.lgbm_model['n_estimators'][0]
        batch_size = X.shape[0]
        
        children_left = self.lgbm_model['children_left']
        children_right = self.lgbm_model['children_right']
        feature = self.lgbm_model['feature']
        threshold = self.lgbm_model['threshold']
        value = self.lgbm_model['value']
        
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
            
        scores = np.zeros(batch_size, dtype=np.float32)
        for b in range(batch_size):
            scores[b] = np.sum(value[np.arange(n_estimators), node_indices[b, :]])
            
        return scores

    def action(self, hand, history):
        if len(hand) == 1:
            return hand[0]
            
        if isinstance(history, dict):
            board = history.get('board', [])
            scores = history.get('scores', [0,0,0,0])
            round_num = history.get('round', 0)
            history_matrix = history.get('history_matrix', [])
            score_history = history.get('score_history', [])
            board_history = history.get('board_history', [])
        else:
            board = history[-1]
            scores = [0,0,0,0]
            round_num = 0
            history_matrix = []
            score_history = []
            board_history = []

        visible_cards = set()
        for row in board:
            visible_cards.update(row)
        for past_round in history_matrix:
            visible_cards.update(past_round)
        if board_history:
            for row in board_history[0]:
                visible_cards.update(row)

        unseen_cards = list(self.total_cards - visible_cards - set(hand))
        
        # 1. Extract 143-dim feature (which includes the state and all cards)
        features_143 = extract_features(
            board=board,
            hand=hand,
            unseen=set(unseen_cards),
            scores=scores,
            player_idx=self.player_idx,
            round_num=round_num,
            history_matrix=history_matrix,
            score_history=score_history,
            board_history=board_history,
        )
        
        state_features = np.concatenate([features_143[0:15], features_143[115:143]])
        
        sorted_hand = sorted(hand)
        n_hand = len(sorted_hand)
        
        X_batch = np.zeros((n_hand, 53), dtype=np.float32)
        for slot in range(n_hand):
            card_features = features_143[15 + slot*10 : 15 + slot*10 + 10]
            X_batch[slot] = np.concatenate([state_features, card_features])
            
        scores_batch = self._predict(X_batch)
        
        # Higher relevance score is better
        best_idx = np.argmax(scores_batch)
        return sorted_hand[best_idx]
