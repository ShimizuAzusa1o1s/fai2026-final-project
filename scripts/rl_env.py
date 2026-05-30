"""
Gymnasium Environment for 6 Nimmt! RL Training

Algorithm:
    - Wraps the custom 6 Nimmt! `Engine` as a single-agent Gymnasium environment.
    - Simulates the 3 opponent turns internally during `step()` to advance the game.

Characteristics:
    - **Depth**: Not applicable (Environment Definition).
    - **Reward Policy**: Zero-sum relative trick reward (opponent average penalty - my penalty).
    - **State Extraction**: Generates the standard 143-dimensional normalized feature vector.

See Also:
    ``src/engine.py`` — The core game engine.
    ``scripts/train_ppo.py`` — The PPO training script.
"""
import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.engine import Engine
from src.players.b12705048.core.features import extract_features, compute_unseen_cards
from src.players.b12705048.agents.flatmc import FlatMC
from src.players.b12705048.agents.greedy import Minimizer

class RLDummyPlayer:
    """
    A dummy player used by the SixNimmtEnv to inject RL agent actions into the Engine.
    
    Attributes:
        player_idx (int): This agent's seat index (0-3).
        played_card (int | None): The card to be returned when queried by the Engine.
    """
    def __init__(self, player_idx):
        """
        Initialize the RLDummyPlayer.

        Args:
            player_idx (int): The player's seat index in the game (0-3).
        """
        self.player_idx = player_idx
        self.played_card = None

    def action(self, hand, history):
        return self.played_card

class SixNimmtEnv(gym.Env):
    """
    Gymnasium wrapper for 6 Nimmt! to train an RL agent against 3 fixed opponents.

    Attributes:
        action_space (spaces.Discrete): The discrete action space representing the 10 hand slots.
        observation_space (spaces.Box): The 143-dimensional normalized feature vector.
        opponent_type (str): The type of opponents to spawn ("minimizer" or "flatmc").
        opponent_time_limit (float): The time budget for FlatMC opponents.
        opponent_model_path (str | None): Path to the RLAgent model if using older selves.
        my_idx (int): The seat index of the RL agent (fixed at 0).
    """
    def __init__(self, opponent_type="minimizer", opponent_time_limit=0.01, opponent_model_path=None):
        """
        Initialize the SixNimmt environment.

        Args:
            opponent_type (str): Opponent type to spawn ("minimizer", "flatmc", or "rl_agent").
            opponent_time_limit (float): Time budget for FlatMC opponents in seconds.
            opponent_model_path (str | None): Model path for RLAgent opponents.
        """
        super().__init__()
        
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(143,), dtype=np.float32)
        
        self.opponent_type = opponent_type
        self.opponent_time_limit = opponent_time_limit
        self.opponent_model_path = opponent_model_path
        self.my_idx = 0
        
        self.engine = None
        self.opponents = []
        self.current_hand = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        cfg = {
            "n_cards": 104,
            "n_players": 4,
            "n_rounds": 10,
            "seed": seed,
            "verbose": False
        }
        
        self.opponents = []
        for i in range(1, 4):
            if self.opponent_type == "flatmc":
                opp = FlatMC(player_idx=i)
                opp.time_limit = self.opponent_time_limit
                # Reduce batch size for short time budgets to prevent massive overshoots
                if self.opponent_time_limit <= 0.05:
                    opp.batch_size = 500
            elif self.opponent_type == "minimizer":
                opp = Minimizer(player_idx=i)
            elif self.opponent_type == "rl_agent":
                from src.players.b12705048.agents.rl_agent import RLAgent
                opp = RLAgent(player_idx=i, model_path=self.opponent_model_path)
            else:
                raise ValueError(f"Unknown opponent type: {self.opponent_type}")
            self.opponents.append(opp)
            
        self.rl_player = RLDummyPlayer(player_idx=self.my_idx)
        players = [self.rl_player] + self.opponents
        self.engine = Engine(cfg, players)
        
        self.current_hand = sorted(self.engine.hands[self.my_idx])
        
        return self._get_obs(), {}

    def _get_obs(self):
        unseen = compute_unseen_cards(
            hand=self.current_hand,
            board=self.engine.board,
            history_matrix=self.engine.history_matrix,
            board_history=self.engine.board_history
        )
        
        features = extract_features(
            board=self.engine.board,
            hand=self.current_hand,
            unseen=unseen,
            scores=self.engine.scores,
            player_idx=self.my_idx,
            round_num=self.engine.round,
            history_matrix=self.engine.history_matrix,
            score_history=self.engine.score_history,
            board_history=self.engine.board_history
        )
        return features

    def valid_action_mask(self):
        mask = np.zeros(10, dtype=bool)
        if len(self.current_hand) > 0:
            mask[:len(self.current_hand)] = True
        return mask

    def step(self, action):
        """
        Execute one step of the environment given the agent's action.

        Args:
            action (int): The chosen index from the remaining hand.

        Returns:
            tuple: (new_obs, reward, done, truncated, info)
        """
        # ---- Phase 1: Action Resolution ----
        if action >= len(self.current_hand):
            raise ValueError(f"Invalid action {action} for hand size {len(self.current_hand)}")
            
        played_card = self.current_hand[action]
        self.rl_player.played_card = played_card
        
        # Snapshot scores before trick to calculate relative reward
        prev_scores = list(self.engine.scores)
        
        # Run one round (the engine will query opponents normally and use self.rl_player.played_card for RL agent)
        self.engine.play_round()
        self.engine.round += 1
        
        # Re-sort hand (play_round removes the played card via engine.hands modification)
        self.current_hand = sorted(self.engine.hands[self.my_idx])
        
        # Calculate Relative Trick Reward
        current_scores = self.engine.scores
        my_penalty = current_scores[self.my_idx] - prev_scores[self.my_idx]
        
        opp_indices = [1, 2, 3]
        opp_penalty = sum(current_scores[i] - prev_scores[i] for i in opp_indices) / 3.0
        
        reward = opp_penalty - my_penalty
        
        done = len(self.current_hand) == 0
        truncated = False
        info = {}
        
        if done:
            info["my_final_score"] = current_scores[self.my_idx]
            info["avg_opp_score"] = sum(current_scores[i] for i in opp_indices) / 3.0
            
        return self._get_obs(), reward, done, truncated, info
