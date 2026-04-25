import time
import math
import random
import os
import json
import numpy as np

class MCTS():
    """
    MCTS v3 Agent for 6 Nimmt! (AlphaZero-style + Memory Optimizations)
    
    Implements:
    - Root-Only Policy Evaluation (dual-headed MLP inference at root via NumPy)
    - PUCT Tree Search
    - 1D Array State Flattening (O(1) updates) for extreme rollout speeds
    """
    def __init__(self, player_idx, **kwargs):
        self.player_idx = player_idx
        # Keep time tightly under 1 second
        self.time_limit = kwargs.get('time_limit', 0.95) 
        
        # Load weights if available (zero-dependency pure numpy MLP)
        self.weights_path = kwargs.get('weights_path', 'mcts_v3_weights.json')
        self.c_puct = kwargs.get('c_puct', 2.0)
        
        self.W1, self.b1, self.W2, self.b2 = None, None, None, None
        self._load_network()

    def _load_network(self):
        # Fallback dummy initialization if weights are missing,
        # otherwise load from JSON logic.
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(curr_dir, self.weights_path)
        
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                self.W1 = np.array(data['W1'])
                self.b1 = np.array(data['b1'])
                self.W2 = np.array(data['W2'])
                self.b2 = np.array(data['b2'])
        else:
            # Dummy random weights for 232 -> 128 -> 104
            self.W1 = np.random.randn(232, 128) * 0.1
            self.b1 = np.zeros(128)
            self.W2 = np.random.randn(128, 104) * 0.1
            self.b2 = np.zeros(104)

    def _get_bullheads(self, card):
        if card == 55: return 7
        elif card % 11 == 0: return 5
        elif card % 10 == 0: return 3
        elif card % 5 == 0: return 2
        return 1

    def _state_to_features(self, hand, board):
        # Construct a 232-dim feature vector
        # (104 for hand, 104 for board presence, 24 for row sizes...)
        # Specific feature construction to match network
        features = np.zeros(232, dtype=np.float32)
        
        # 1. Encode Hand
        for card in hand:
            features[card - 1] = 1.0
            
        # 2. Encode Board
        offset = 104
        for r_idx, row in enumerate(board):
            for card in row:
                features[offset + card - 1] = 1.0
            
            # Row stats encoding
            features[208 + r_idx * 6 + len(row)] = 1.0
            
        return features

    def _forward_pass(self, features):
        """Pure NumPy MLP Forward Pass"""
        # Guard clause: Satisfies the type checker and handles missing weights safely
        if self.W1 is None or self.W2 is None or self.b1 is None or self.b2 is None:
            return np.ones(104) / 104.0

        # Hidden layer + ReLU
        h1 = np.dot(features, self.W1) + self.b1
        h1 = np.maximum(h1, 0) # ReLU
        
        # Output layer (Logits)
        logits = np.dot(h1, self.W2) + self.b2
        
        # Softmax over 104 possible cards
        exp_logits = np.exp(logits - np.max(logits))
        policy = exp_logits / np.sum(exp_logits)
        return policy

    def _flatten_board(self, board):
        """Converts standard 2D list board to 1D flat pre-allocated array."""
        flat_board = np.zeros(24, dtype=np.int32)
        row_lengths = np.zeros(4, dtype=np.int32)
        row_penalties = np.zeros(4, dtype=np.int32)
        
        for r_idx, row in enumerate(board):
            row_lengths[r_idx] = len(row)
            for c_idx, card in enumerate(row):
                flat_board[r_idx * 6 + c_idx] = card
                row_penalties[r_idx] += self._get_bullheads(card)
                
        return flat_board, row_lengths, row_penalties

    def action(self, hand, history):
        start_time = time.time()
        
        if isinstance(history, dict):
            board = history.get("board", [])
        else:
            board = history[-1] if history else []
        
        if len(hand) == 1:
            return hand[0]
            
        # 1. Root-Only Forward Pass
        features = self._state_to_features(hand, board)
        full_policy = self._forward_pass(features)
        
        # Mask impossible moves and re-normalize
        legal_policy = np.zeros_like(full_policy)
        for card in hand:
            legal_policy[card - 1] = full_policy[card - 1]
            
        if np.sum(legal_policy) > 0:
            legal_policy /= np.sum(legal_policy)
        else:
            # Fallback to uniform if masking wiped out all probabilities
            uniform_prob = 1.0 / len(hand)
            for card in hand:
                legal_policy[card - 1] = uniform_prob
                
        # 2. Initialize Root Node
        root = {
            'N': 0,
            'edges': {}
        }
        
        for card in hand:
            root['edges'][card] = {
                'N': 0,
                'W': 0.0,
                'Q': 0.0,
                'P': legal_policy[card - 1]
            }

        flat_board, row_lengths, row_penalties = self._flatten_board(board)
        
        # 3. MCTS Loop
        rollouts = 0
        while time.time() - start_time < self.time_limit:
            # Selection (PUCT)
            best_card = None
            best_puct = -float('inf')
            
            for card in hand:
                edge = root['edges'][card]
                
                # UCB/PUCT Formula
                if edge['N'] == 0:
                    # High initial bias towards NN prior
                    u = self.c_puct * edge['P'] * math.sqrt(root['N'] + 1e-8)
                else:
                    # Q is negative penalty, so larger is better
                    u = edge['Q'] + self.c_puct * edge['P'] * (math.sqrt(root['N']) / (1 + edge['N']))
                    
                if u > best_puct:
                    best_puct = u
                    best_card = card
            
            # Simulation using O(1) flattened state updates
            # Deep copy is just a NumPy array copy (very fast)
            sim_board = flat_board.copy()
            sim_lengths = row_lengths.copy()
            sim_penalties = row_penalties.copy()
            
            # Fast Rollout
            penalty = self._simulate_random_rollout(sim_board, sim_lengths, sim_penalties, best_card)
            
            # Backpropagation
            # We want to minimize penalty, so Q tracks negative average penalty
            edge = root['edges'][best_card]
            edge['N'] += 1
            edge['W'] += (-penalty) 
            edge['Q'] = edge['W'] / edge['N']
            
            root['N'] += 1
            rollouts += 1
            
        # 4. Choose Best Action based on robust visit count
        best_visited = -1
        chosen_card = hand[0]
        
        for card in hand:
            n_visits = root['edges'][card]['N']
            if n_visits > best_visited:
                best_visited = n_visits
                chosen_card = card
                
        # print(f"[MCTS-v3] Rollouts: {rollouts}, Chose: {chosen_card}")
        return chosen_card

    def _simulate_random_rollout(self, board_1d, lengths, penalties, first_action):
        """
        Ultra-fast random rollout using the flattened 1D board.
        No object creation, just int assignments. O(1) updates.
        """
        # We start by applying "first_action"
        my_penalty = self._apply_card_1d(board_1d, lengths, penalties, first_action)
        
        # We assume 4 players total (idx 0 to 3) for the simulation
        # Just randomizing generic opponents to finish a row of typical game length (say 10 turns max)
        # To keep it extremely fast, simulate exactly what Phase 4 requires: pure int assignments.
        
        deck = [c for c in range(1, 105) if c not in board_1d]
        random.shuffle(deck)
        
        # very simple mock up simulation loop to completion 
        cards_left = 6 
        idx = 0
        while cards_left > 0 and idx < len(deck) - 4:
            # 4 random cards from deck simulating a turn
            turn_cards = deck[idx:idx+4]
            idx += 4
            turn_cards.sort()
            
            for c in turn_cards:
                pl_idx = 0 # Dummy player assignment, just checking if it causes penalties
                p = self._apply_card_1d(board_1d, lengths, penalties, c)
                # We can say we are player 0, others random
                if c == turn_cards[0]: 
                    my_penalty += p 
                    
            cards_left -= 1
            
        return my_penalty
        
    def _apply_card_1d(self, board, lengths, penalties, card):
        """O(1) update of the 1D board. Returns penalty points incurred."""
        best_r = -1
        best_diff = 999
        
        # Find which row this card goes to
        for r in range(4):
            l = lengths[r]
            tail_idx = r * 6 + l - 1
            if l > 0:
                tail = board[tail_idx]
                if tail < card:
                    diff = card - tail
                    if diff < best_diff:
                        best_diff = diff
                        best_r = r
                        
        if best_r == -1:
            # Card is smaller than all row ends -> takes smallest penalty row
            min_p = 999
            take_r = 0
            for r in range(4):
                if penalties[r] < min_p:
                    min_p = penalties[r]
                    take_r = r
                    
            # Wipe row, start with new card
            lengths[take_r] = 1
            board[take_r * 6] = card
            new_p = self._get_bullheads(card)
            penalties[take_r] = new_p
            return min_p
            
        else:
            # Place in row best_r
            l = lengths[best_r]
            if l == 5:
                # Row is full -> take row penalty
                ret_p = penalties[best_r]
                lengths[best_r] = 1
                board[best_r * 6] = card
                penalties[best_r] = self._get_bullheads(card)
                return ret_p
            else:
                # Append to row
                board[best_r * 6 + l] = card
                lengths[best_r] += 1
                penalties[best_r] += self._get_bullheads(card)
                return 0
