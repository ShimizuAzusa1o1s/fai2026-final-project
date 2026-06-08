import os
import sys
import copy
import numpy as np

# Ensure src can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.engine import Engine
from src.players.b12705048.agents.flatmc_cpp import FlatMCCPP
from src.players.b12705048.models.opp_net.feature_extractor import build_feature_vector

class DummyPlayer:
    def action(self, hand, history):
        pass

class MuZeroEnv:
    """
    OpenAI Gym-style wrapper for 6 Nimmt!
    MuZero controls Player 0. Players 1-3 are controlled by FlatMC bots.
    """
    def __init__(self, time_limit=0.1):
        self.cfg = {
            "n_players": 4,
            "n_rounds": 10,
            "timeout": None, # Disable timeouts for training
            "verbose": False
        }
        self.muzero_idx = 0
        self.time_limit = time_limit
        
    def reset(self):
        # Create 3 FlatMC opponents using Level 3 config
        players = [DummyPlayer()]
        for i in range(1, 4):
            players.append(FlatMCCPP(player_idx=i, time_limit=self.time_limit, epsilon=0.2, tau=1.0, model_level=3, use_neural_determinization=True))
            
        self.engine = Engine(self.cfg, players)
        self.engine.board_history.append([row.copy() for row in self.engine.board])
        return self._get_observation(), self.engine.hands[self.muzero_idx].copy()
        
    def _get_observation(self):
        """
        Build 125-dim history feature + 105-dim one-hot hand = 230-dim observation
        """
        my_hand = self.engine.hands[self.muzero_idx]
        
        total_cards = set(range(1, 105))
        visible = set()
        for row in self.engine.board:
            visible.update(row)
        visible.update(my_hand)
        for row in self.engine.history_matrix:
            visible.update(row)
            
        unseen = list(total_cards - visible)
        
        history_dict = {
            'history_matrix': self.engine.history_matrix,
            'board_history': self.engine.board_history,
            'score_history': self.engine.score_history
        }
        
        X = build_feature_vector(history_dict, self.engine.round, self.muzero_idx, unseen, len(my_hand))
        
        # One-hot encode the hand
        hand_feat = np.zeros(105, dtype=np.float32)
        hand_feat[my_hand] = 1.0
        
        obs = np.concatenate([X, hand_feat])
        return obs

    def step(self, action):
        """
        action: int (card value to play)
        """
        my_hand = self.engine.hands[self.muzero_idx]
        if action not in my_hand:
            # Illegal move. Return massive penalty and done.
            return self._get_observation(), -100.0, True, {"legal_actions": []}
            
        current_played_cards = []
        round_actions = [0] * 4
        round_flags = [False] * 4
        
        history_state = {
            "board": copy.deepcopy(self.engine.board),
            "scores": list(self.engine.scores),
            "round": self.engine.round,
            "history_matrix": [r.copy() for r in self.engine.history_matrix],
            "board_history": [b.copy() for b in self.engine.board_history],
            "score_history": [s.copy() for s in self.engine.score_history],
        }
        
        # Get actions for all players
        for p_idx in range(4):
            if p_idx == self.muzero_idx:
                played_card = action
            else:
                played_card = self.engine.players[p_idx].action(self.engine.hands[p_idx].copy(), copy.deepcopy(history_state))
                
            self.engine.hands[p_idx].remove(played_card)
            current_played_cards.append((played_card, p_idx))
            round_actions[p_idx] = played_card
            
        self.engine.history_matrix.append(round_actions)
        self.engine.flags_matrix.append(round_flags)
        
        # Resolve turn
        current_played_cards.sort(key=lambda x: x[0])
        
        old_score = self.engine.scores[self.muzero_idx]
        for card, p_idx in current_played_cards:
            self.engine.process_card_placement(card, p_idx)
            
        self.engine.score_history.append(list(self.engine.scores))
        
        new_score = self.engine.scores[self.muzero_idx]
        reward = -(new_score - old_score) # Penalty incurred this turn (negative reward)
        
        self.engine.round += 1
        done = self.engine.round >= self.engine.n_rounds
        
        # Snapshot board history for the NEXT round (or terminal state)
        self.engine.board_history.append([row.copy() for row in self.engine.board])
        
        legal_actions = [] if done else self.engine.hands[self.muzero_idx].copy()
        
        return self._get_observation(), float(reward), done, {"legal_actions": legal_actions}
