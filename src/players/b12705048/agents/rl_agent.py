import os
import numpy as np
from sb3_contrib import MaskablePPO

from src.players.b12705048.core.features import extract_features, compute_unseen_cards

class RLAgent:
    """
    Agent wrapper for the trained MaskablePPO RL Model.
    Can be used in standard 6 Nimmt! tournaments.
    """
    def __init__(self, player_idx, model_path="models/ppo_agent/stage3_model_final"):
        self.player_idx = player_idx
        
        self.model = None
        if os.path.exists(f"{model_path}.zip"):
            self.model = MaskablePPO.load(model_path)
        else:
            print(f"Warning: RL model not found at {model_path}. Using fallback strategy.")
            
    def action(self, hand, history):
        if self.model is None:
            return min(hand)
            
        if isinstance(history, dict):
            board = history.get('board', [])
            scores = history.get('scores', [0]*4)
            round_num = history.get('round', 0)
            history_matrix = history.get('history_matrix', [])
            board_history = history.get('board_history', [])
            score_history = history.get('score_history', [])
        else:
            board = history[-1]
            scores = [0]*4
            round_num = len(hand)
            history_matrix = []
            board_history = []
            score_history = []

        unseen = compute_unseen_cards(
            hand=hand,
            board=board,
            history_matrix=history_matrix,
            board_history=board_history
        )
        
        obs = extract_features(
            board=board,
            hand=hand,
            unseen=unseen,
            scores=scores,
            player_idx=self.player_idx,
            round_num=round_num,
            history_matrix=history_matrix,
            score_history=score_history,
            board_history=board_history
        )
        
        sorted_hand = sorted(hand)
        n_hand = len(sorted_hand)
        mask = np.zeros(10, dtype=bool)
        if n_hand > 0:
            mask[:n_hand] = True
            
        action, _states = self.model.predict(obs, action_masks=mask, deterministic=True)
        
        action = int(action)
        if action >= n_hand:
            action = 0
            
        return sorted_hand[action]
