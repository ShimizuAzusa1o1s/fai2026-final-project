import os
import torch
import time
import random
import numpy as np

from .fast_env import FastState
from .state_encoding import Encoding, get_state_dim, N_CARDS
from .model import TinyAlphaZeroNet

class AlphaZeroPlayer:
    def __init__(self, player_idx, model_path=None, n_playouts=50, time_limit=0.9, device=None):
        self.player_idx = player_idx
        self.n_playouts = n_playouts
        self.time_limit = time_limit
        self.state_dim = get_state_dim()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model = TinyAlphaZeroNet(self.state_dim, N_CARDS).to(self.device)
        self.model.device = self.device # type: ignore
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.total_cards = set(range(1, 105))

    def _determinize_hands(self, my_hand, history):
        board = history.get('board', [])
        
        visible_cards = set()
        for row in board:
            visible_cards.update(row)
            
        for past_round in history.get('history_matrix', []):
            visible_cards.update(past_round)
            
        if history.get('board_history'):
            for row in history['board_history'][0]:
                visible_cards.update(row)
        
        unseen_cards = list(self.total_cards - visible_cards - set(my_hand))
        random.shuffle(unseen_cards)
        
        h = len(my_hand)
        
        # Determine opponent hands randomly
        hands = []
        for i in range(4):
            if i == self.player_idx:
                hands.append(my_hand.copy())
            else:
                hands.append(unseen_cards[:h])
                unseen_cards = unseen_cards[h:]
                
        return hands

    def _playout(self, root_state, c_puct=1.5):
        # Perform iterative playouts using the neural model
        pass

    def get_action_probs(self, state_history, current_hand, temperature=1e-3, use_mcts=True):
        if use_mcts and self.n_playouts > 0:
            from .mcts import MCTS_PUCT
            # Build fast state
            board = state_history.get('board', [])
            scores = state_history.get('scores', [0] * 4)
            round_num = state_history.get('round', 0)
            
            # Determinize other hands
            hands = self._determinize_hands(current_hand, state_history)
            
            fast_state = FastState(board, scores, hands, round_num)
            
            def mcts_encoder(fast_state, p_idx):
                # We mock a small history dict to reuse Encoding.encode_state
                # We can't perfectly reconstruct history_matrix for future nodes,
                # but we can pass what we have plus the new round's state.
                mock_history = dict(state_history)
                mock_history['board'] = fast_state.board
                mock_history['scores'] = fast_state.scores
                mock_history['round'] = fast_state.round
                return Encoding.encode_state(mock_history, fast_state.hands[p_idx], p_idx)
            
            mcts = MCTS_PUCT(self.model.predict, c_puct=1.5, n_playout=self.n_playouts, time_limit=self.time_limit)
            best_a, target_probs = mcts.get_action(fast_state, None, mcts_encoder, self.player_idx, temperature=temperature)
            return best_a, target_probs
            
        else:
            # We will use the model's forward pass to evaluate all available options.
            state_vec, mask = Encoding.encode_state(state_history, current_hand, self.player_idx)
            
            # Direct policy fallback if search is skipped or too slow
            p, v = self.model.predict(state_vec, mask)
            
            # If we have very little time or budget, we just act greedly on policy prior:
            best_a = np.argmax(p) + 1 # Convert 0-indexed back to 1-104
            
            target_probs = p.copy()
            
            return best_a, target_probs

    def action(self, hand, history):
        start_time = time.perf_counter()
        
        # We simply use the model prior + PUCT to stay safe
        time_left = self.time_limit - (time.perf_counter() - start_time)
        
        best_action, target_probs = self.get_action_probs(history, hand, temperature=0.0, use_mcts=True)
        
        elapsed = time.perf_counter() - start_time
        # print(f"AlphaZero Player {self.player_idx} acted in {elapsed:.3f}s - chose {best_action}")
        
        return best_action
