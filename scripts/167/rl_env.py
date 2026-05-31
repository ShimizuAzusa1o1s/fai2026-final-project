"""
Gymnasium Environment for 6 Nimmt! RL Training

Algorithm:
    - Wraps the custom 6 Nimmt! `Engine` as a single-agent Gymnasium environment.
    - Simulates the 3 opponent turns internally during `step()` to advance the game.

Characteristics:
    - **Depth**: Not applicable (Environment Definition).
    - **Reward Policy**: Zero-sum relative trick reward (opponent average penalty - my penalty).
    - **State Extraction**: Generates the standard 167-dimensional normalized feature vector.

See Also:
    ``src/engine.py`` — The core game engine.
    ``scripts/train_ppo.py`` — The PPO training script.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.engine import Engine
from src.players.b12705048.core.features_167 import extract_features, compute_unseen_cards
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
        if self.played_card is not None:
            return self.played_card
        return int(np.random.choice(hand))

class SixNimmtEnv(gym.Env):
    """
    Gymnasium wrapper for 6 Nimmt! to train an RL agent against 3 fixed opponents.

    Attributes:
        action_space (spaces.Discrete): The discrete action space representing the 10 hand slots.
        observation_space (spaces.Box): The 167-dimensional normalized feature vector.
        opponent_type (str): The type of opponents to spawn ("minimizer" or "flatmc").
        opponent_time_limit (float): The time budget for FlatMC opponents.
        opponent_model_path (str | None): Path to the RLAgent model if using older selves.
        spawn_trick (int): How many tricks to fast-forward before the agent takes control.
        reward_shaping_weight (float): Multiplier for heatmap-driven penalty shaping.
        my_idx (int): The seat index of the RL agent (fixed at 0).
    """
    _model_cache = {}

    def _get_rl_agent(self, player_idx, model_path):
        """
        Retrieves a cached RLAgent or loads a new one, ensuring memory efficiency.
        
        Args:
            player_idx (int): The seat index for the agent (1-3).
            model_path (str): The path to the trained RL model to load.
            
        Returns:
            RLAgent: A shallow copy of the cached RLAgent with the specified player_idx.
        """
        import os
        from src.players.b12705048.agents.rl_agent import RLAgent
        import copy
        
        # Dynamically reload latest model if modified time changed
        if "latest" in model_path:
            full_path = f"{model_path}.zip"
            if os.path.exists(full_path):
                mtime = os.path.getmtime(full_path)
                if getattr(self, "_latest_mtime", 0) < mtime:
                    if model_path in self._model_cache:
                        del self._model_cache[model_path]
                    self._latest_mtime = mtime

        if model_path not in self._model_cache:
            # Load and cache RLAgent once per process to save memory
            self._model_cache[model_path] = RLAgent(player_idx=0, model_path=model_path)
        
        # Shallow copy to assign the correct player_idx
        agent = copy.copy(self._model_cache[model_path])
        agent.player_idx = player_idx
        return agent

    def __init__(self, opponent_type="minimizer", opponent_time_limit=0.01, opponent_model_path=None, spawn_trick=0, reward_shaping_weight=0.0):
        """
        Initialize the SixNimmt environment.

        Args:
            opponent_type (str): Opponent type to spawn ("minimizer", "flatmc", or "rl_agent").
            opponent_time_limit (float): Time budget for FlatMC opponents in seconds.
            opponent_model_path (str | None): Model path for RLAgent opponents.
            spawn_trick (int): How many tricks to fast-forward before the agent takes control.
            reward_shaping_weight (float): Multiplier for heatmap-driven penalty shaping.
        """
        super().__init__()
        
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(167,), dtype=np.float32)
        
        self.opponent_type = opponent_type
        self.opponent_time_limit = opponent_time_limit
        self.opponent_model_path = opponent_model_path
        self.spawn_trick = spawn_trick
        self.reward_shaping_weight = reward_shaping_weight
        self.my_idx = 0
        
        self.engine = None
        self.opponents = []
        self.current_hand = []
        self.last_obs = None
        
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
            current_opp_type = self.opponent_type
            if current_opp_type == "mixed":
                p = np.random.rand()
                if p < 0.2:
                    current_opp_type = "minimizer"
                elif p < 0.5:
                    current_opp_type = "flatmc"
                else:
                    current_opp_type = "rl_agent"

            if current_opp_type == "flatmc":
                opp = FlatMC(player_idx=i)
                opp.time_limit = self.opponent_time_limit
                # Reduce batch size for short time budgets to prevent massive overshoots
                if self.opponent_time_limit <= 0.05:
                    opp.batch_size = 500
            elif current_opp_type == "minimizer":
                opp = Minimizer(player_idx=i)
            elif current_opp_type == "rl_agent":
                if isinstance(self.opponent_model_path, list):
                    path = np.random.choice(self.opponent_model_path)
                else:
                    path = self.opponent_model_path
                opp = self._get_rl_agent(i, path)
            else:
                raise ValueError(f"Unknown opponent type: {current_opp_type}")
            self.opponents.append(opp)
            
        self.rl_player = RLDummyPlayer(player_idx=self.my_idx)
        players = [self.rl_player] + self.opponents
        self.engine = Engine(cfg, players)
        
        # Fast-forward for spawn_trick
        for _ in range(self.spawn_trick):
            self.rl_player.played_card = None
            self.engine.play_round()
            self.engine.round += 1
        
        self.current_hand = sorted(self.engine.hands[self.my_idx])
        
        self.last_obs = self._get_obs()
        return self.last_obs, {}

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
            tuple[np.ndarray, float, bool, bool, dict]: Observation, reward, done flag, truncation flag, and info dict.
        """
        # ---- Phase 1: Action Resolution ----
        if action >= len(self.current_hand):
            raise ValueError(f"Invalid action {action} for hand size {len(self.current_hand)}")
            
        played_card = self.current_hand[action]
        self.rl_player.played_card = played_card
        
        # ---- Phase 2: Heatmap Reward Shaping ----
        if self.reward_shaping_weight > 0 and self.last_obs is not None:
            base_idx = 15 + action * 12
            gap_bh_norm = self.last_obs[base_idx + 10]
            gap_density = self.last_obs[base_idx + 11]
            shaping_penalty = gap_bh_norm * gap_density * 5.0
        else:
            shaping_penalty = 0.0

        # ---- Phase 3: Engine Simulation Step ----
        # Snapshot scores before trick to calculate relative reward
        prev_scores = list(self.engine.scores)
        
        # Run one round (the engine will query opponents normally and use self.rl_player.played_card for RL agent)
        self.engine.play_round()
        self.rl_player.played_card = None
        self.engine.round += 1
        
        # Re-sort hand (play_round removes the played card via engine.hands modification)
        self.current_hand = sorted(self.engine.hands[self.my_idx])
        
        # ---- Phase 4: Calculate Relative Trick Reward ----
        current_scores = self.engine.scores
        my_penalty = current_scores[self.my_idx] - prev_scores[self.my_idx]
        
        opp_indices = [1, 2, 3]
        opp_penalty = sum(current_scores[i] - prev_scores[i] for i in opp_indices) / 3.0
        
        reward = opp_penalty - my_penalty - (shaping_penalty * self.reward_shaping_weight)
        
        done = len(self.current_hand) == 0
        truncated = False
        info = {}
        
        self.last_obs = self._get_obs()
        
        if done:
            info["my_final_score"] = current_scores[self.my_idx]
            info["avg_opp_score"] = sum(current_scores[i] for i in opp_indices) / 3.0
            
        return self.last_obs, reward, done, truncated, info
