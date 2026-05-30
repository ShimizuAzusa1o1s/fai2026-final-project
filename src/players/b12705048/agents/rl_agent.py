"""
RL Agent Wrapper for 6 Nimmt!

Algorithm:
    - Wraps a trained MaskablePPO Reinforcement Learning model to act as a 
      standard player in the game engine.

Characteristics:
    - **Depth**: 1-ply (evaluates immediate action via policy network).
    - **Rollout Policy**: Deterministic policy prediction from the PPO model.
    - **Time Management**: O(1) inference time (no simulation budget required).

See Also:
    ``scripts/train_ppo.py`` — Training script that generates the model.
    ``scripts/rl_env.py`` — The Gymnasium environment used for training.
"""
import os
import numpy as np
import torch
torch.set_num_threads(1)
from sb3_contrib import MaskablePPO

from src.players.b12705048.core.features import extract_features, compute_unseen_cards

class RLAgent:
    """
    Agent wrapper for the trained MaskablePPO RL Model.

    Can be used in standard 6 Nimmt! tournaments by providing the standard `action` method.

    Attributes:
        player_idx (int): This agent's seat index (0-3).
        model (MaskablePPO | None): The loaded RL model, or None if not found.
    """
    def __init__(self, player_idx, model_path=None):
        """
        Initialize the RLAgent.

        Args:
            player_idx (int): The player's seat index in the game (0-3).
            model_path (str | None): Path to the trained MaskablePPO model zip file (without .zip).
                                     If None, defaults to 'stage3_model_final' in the same directory.
        """
        self.player_idx = player_idx
        
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "stage3_model_final")

        self.model = None
        if os.path.exists(f"{model_path}.zip"):
            self.model = MaskablePPO.load(model_path)
        else:
            print(f"Warning: RL model not found at {model_path}. Using fallback strategy.")
            
    def action(self, hand, history):
        """
        Select the best card to play using the trained RL policy.

        Args:
            hand (list[int]): Cards currently available to play.
            history (dict | list): Game state from the engine.

        Returns:
            int: The card value selected by the policy.
        """
        # ---- Phase 1: State Parsing ----
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
        
        # ---- Phase 2: Action Masking & Prediction ----
        sorted_hand = sorted(hand)
        n_hand = len(sorted_hand)
        mask = np.zeros(10, dtype=bool)
        if n_hand > 0:
            mask[:n_hand] = True
            
        action, _states = self.model.predict(obs, action_masks=mask, deterministic=True)
        
        # ---- Phase 3: Action Resolution ----
        action = int(action)
        if action >= n_hand:
            action = 0
            
        return sorted_hand[action]
