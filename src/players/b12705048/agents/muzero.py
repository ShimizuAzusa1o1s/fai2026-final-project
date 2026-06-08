import os
import sys
import torch
import numpy as np

from src.players.b12705048.agents.muzero_mcts import LatentMCTS
from src.players.b12705048.models.muzero_net.model import MuZeroNet
from src.players.b12705048.models.opp_net.feature_extractor import build_feature_vector

class MuZeroAgent:
    """
    The MuZero Agent interface for 6 Nimmt! server and evaluation.
    """
    def __init__(self, player_idx, weights_path=None, num_simulations=50, temperature=0.0):
        self.player_idx = player_idx
        self.num_simulations = num_simulations
        self.temperature = temperature
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MuZeroNet(obs_dim=230, hidden_dim=128, action_dim=105).to(self.device)
        self.model.eval()
        
        if weights_path and os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            
    def action(self, hand, history):
        """
        Evaluate the position and select a card using Latent MCTS.
        """
        # Parse history
        if isinstance(history, dict):
            board = history.get('board', [])
            target_round = history.get('round', 0)
        else:
            board = history[-1]
            target_round = 0

        visible_cards = set()
        for row in board:
            visible_cards.update(row)
            
        if isinstance(history, dict):
            for past_round in history.get('history_matrix', []):
                visible_cards.update(past_round)
            if history.get('board_history'):
                for row in history['board_history'][0]:
                    visible_cards.update(row)
                    
        total_cards = set(range(1, 105))
        unseen_cards = list(total_cards - visible_cards - set(hand))
        
        if isinstance(history, dict) and 'score_history' in history:
            X = build_feature_vector(
                history, target_round, self.player_idx,
                unseen_cards, len(hand)
            )
        else:
            X = np.zeros(125, dtype=np.float32)

        hand_feat = np.zeros(105, dtype=np.float32)
        hand_feat[hand] = 1.0
        
        obs = np.concatenate([X, hand_feat])
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Run Latent MCTS
        mcts = LatentMCTS(self.model, num_simulations=self.num_simulations)
        root = mcts.run(obs_tensor, hand)
        
        policy = mcts.get_action_policy(root, temperature=self.temperature)
        
        # Select action
        actions = list(policy.keys())
        probs = list(policy.values())
        
        if len(actions) == 0:
            return hand[0]
            
        chosen_action = int(np.random.choice(actions, p=probs))
        
        return chosen_action
