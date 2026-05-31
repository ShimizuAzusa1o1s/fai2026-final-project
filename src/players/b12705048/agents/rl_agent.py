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
from sb3_contrib import RecurrentPPO


class RLAgent:
    """
    Agent wrapper for the trained MaskablePPO RL Model.

    Can be used in standard 6 Nimmt! tournaments by providing the standard `action` method.

    Attributes:
        player_idx (int): This agent's seat index (0-3).
        model (RecurrentPPO | None): The loaded RL model, or None if not found.
        lstm_states: The hidden states for the LSTM.
        episode_starts (np.ndarray): Array indicating if a new episode has started.
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
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)
        
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "models", "rl_model_167_stage3")

        self.model = None
        if os.path.exists(f"{model_path}.zip"):
            self.model = RecurrentPPO.load(model_path)
            
            # Dynamically determine feature extractor based on model input space
            obs_dim = self.model.observation_space.shape[0]
            if obs_dim == 143:
                from src.players.b12705048.core.features_143 import extract_features, compute_unseen_cards
            elif obs_dim == 167:
                from src.players.b12705048.core.features_167 import extract_features, compute_unseen_cards
            else:
                raise ValueError(f"Unsupported model feature dimension: {obs_dim}")
                
            self.extract_features = extract_features
            self.compute_unseen_cards = compute_unseen_cards
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
            
        if len(hand) == 10:
            self.lstm_states = None
            self.episode_starts = np.ones((1,), dtype=bool)
            
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

        unseen = self.compute_unseen_cards(
            hand=hand,
            board=board,
            history_matrix=history_matrix,
            board_history=board_history
        )
        
        obs = self.extract_features(
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
        
        # ---- Phase 2: Action Prediction ----
        sorted_hand = sorted(hand)
        n_hand = len(sorted_hand)
            
        action, lstm_states = self.model.predict(
            obs,
            state=self.lstm_states,
            episode_start=self.episode_starts,
            deterministic=True
        )
        self.lstm_states = lstm_states
        self.episode_starts[:] = False
        
        # ---- Phase 3: Action Resolution ----
        action = int(action)
        if action >= n_hand:
            action = 0
            
        return sorted_hand[action]

    def evaluate_batch(self, obs_matrix: np.ndarray) -> np.ndarray:
        """
        Evaluate a batch of states using the PPO Critic (Value Network).
        
        Args:
            obs_matrix (np.ndarray): Batch of N-dimensional feature vectors
                (N=143 or N=167 depending on the loaded model).
            
        Returns:
            np.ndarray: Batch of estimated state values (expected relative penalties).
        """
        if self.model is None or obs_matrix.shape[0] == 0:
            return np.zeros(obs_matrix.shape[0], dtype=np.float32)
            
        import torch
        obs_tensor = torch.as_tensor(obs_matrix, dtype=torch.float32, device=self.model.device)
        with torch.no_grad():
            values = self.model.policy.predict_values(obs_tensor)
            
        return values.cpu().numpy().flatten()
