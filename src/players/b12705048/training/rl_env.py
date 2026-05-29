import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.engine import Engine
from src.players.b12705048.core.features import extract_features, compute_unseen_cards
from src.players.b12705048.agents.flatmc import FlatMC
from src.players.b12705048.agents.greedy import Minimizer

class RLDummyPlayer:
    def __init__(self, player_idx):
        self.player_idx = player_idx
        self.played_card = None

    def action(self, hand, history):
        return self.played_card

class SixNimmtEnv(gym.Env):
    """
    Gymnasium wrapper for 6 Nimmt! to train an RL agent against 3 fixed opponents.
    """
    def __init__(self, opponent_type="minimizer", opponent_time_limit=0.01):
        super().__init__()
        
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(143,), dtype=np.float32)
        
        self.opponent_type = opponent_type
        self.opponent_time_limit = opponent_time_limit
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
            elif self.opponent_type == "minimizer":
                opp = Minimizer(player_idx=i)
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
