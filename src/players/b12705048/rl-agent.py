import os
import json

class RLPlayer():
    def __init__(self, player_idx):
        self.player_idx = player_idx
        self.total_cards = set(range(1, 105))
        self.seen_cards = set()
        
        # Load the trained weights from the JSON file
        self.W1, self.b1, self.W2, self.b2 = self._load_weights()

    def _load_weights(self):
        """Safely loads weights relative to this script's location."""
        # This ensures the file is found even if run_tournament.py is executed from the root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(current_dir, 'dqn_weights.json')
        
        try:
            with open(weights_path, 'r') as f:
                data = json.load(f)
            return data['W1'], data['b1'], data['W2'], data['b2']
        except Exception as e:
            # Fallback: If JSON is missing, return empty lists so we don't crash the engine
            print(f"Warning: Could not load weights from {weights_path}: {e}")
            return [], [], [], []

    def _forward_pass(self, state):
        """Pure Python, zero-dependency Neural Network inference."""
        # If weights failed to load, return zeros
        if not self.W1:
            return [0.0] * 104

        # 1. Hidden Layer with ReLU Activation
        hidden = []
        for i in range(len(self.W1)):
            dot_product = sum(state[j] * self.W1[i][j] for j in range(len(state))) + self.b1[i]
            hidden.append(max(0.0, dot_product)) # ReLU

        # 2. Output Layer (Linear)
        output = []
        for i in range(len(self.W2)):
            dot_product = sum(hidden[j] * self.W2[i][j] for j in range(len(hidden))) + self.b2[i]
            output.append(dot_product)

        return output

    def _encode_state(self, hand, board):
        """Converts the game state into the 232-length feature vector."""
        # 1. Board Features (4 rows x 6 slots = 24 inputs)
        # We pad the rows with 0s if they have less than 6 cards
        board_features = []
        for row in board:
            row_padded = row + [0] * (6 - len(row))
            board_features.extend(row_padded)
            
        # 2. Hand Features (104 binary inputs: 1 if in hand, 0 otherwise)
        hand_features = [1.0 if card in hand else 0.0 for card in range(1, 105)]
        
        # 3. Seen Cards Features (104 binary inputs: 1 if dead/seen, 0 otherwise)
        seen_features = [1.0 if card in self.seen_cards else 0.0 for card in range(1, 105)]
        
        # Combine everything into a 232-length flat list
        state_vector = board_features + hand_features + seen_features
        return state_vector

    def action(self, hand, history):
        # 1. Parse Board & Update Memory
        board = history.get('board', []) if isinstance(history, dict) else history[-1]
        
        self.seen_cards.update(hand)
        for row in board:
            self.seen_cards.update(row)
            
        # 2. Encode State
        state_vector = self._encode_state(hand, board)
        
        # 3. Brain Inference (Takes < 5ms)
        q_values = self._forward_pass(state_vector)
        
        # 4. Action Masking (Critical Step)
        # The network outputs 104 values, but we can only play cards we actually hold.
        # We find the valid card with the lowest expected penalty (or highest reward,
        # depending on how you framed your PyTorch loss function). 
        # Assuming Q-values represent expected NEGATIVE bullheads (e.g., -5 is worse than -1):
        # We want to maximize the Q-value.
        # Note: If your network predicts actual bullhead penalties (positive numbers), 
        # change `max` to `min` below!
        
        best_card = None
        best_q = float('-inf') 
        
        for card in hand:
            # Q-values list is 0-indexed, so card 1 is at index 0
            card_q_value = q_values[card - 1] 
            
            if card_q_value > best_q:
                best_q = card_q_value
                best_card = card
                
        # Fallback just in case
        if best_card is None:
            best_card = min(hand)
            
        return best_card