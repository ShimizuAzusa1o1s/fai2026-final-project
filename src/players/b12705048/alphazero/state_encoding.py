"""
Game State Encoding for Neural Network Input
=============================================

Converts 6 Nimmt! game states into fixed-length vectors suitable for neural network
inference. This is crucial for the AlphaZero architecture.

The encoder creates a single feature vector from:
  - Board state (current rows of cards)
  - Player's hand (cards available to play)
  - Game history (cards played in previous rounds)
  - Score information (cumulative penalties)
  - Round number (game progress)
  - Bullhead counts per row (penalty features)

All features are normalized to reasonable ranges for neural network input.
"""

import numpy as np
from .fast_env import get_bullheads

# Game constants for 6 Nimmt!
N_CARDS = 104      # Total number of unique cards
N_ROWS = 4         # Number of rows on the board
MAX_ROW_LEN = 5    # Maximum cards per row before forced placement
N_PLAYERS = 4      # Number of players
# Note: STATE_DIM is computed at module init, see get_state_dim()
# Added 4 features for cumulative bullheads per row
STATE_DIM = N_CARDS * 3 + N_ROWS * MAX_ROW_LEN + N_PLAYERS * 1 + 1 + N_ROWS


class Encoding:
    """
    Static class for encoding game states to neural network input vectors.
    
    Converts variable-length game history into a fixed-size feature vector
    that can be fed to the TinyAlphaZeroNet for policy and value prediction.
    """
    
    @staticmethod
    def encode_state(history, current_hand, player_idx):
        """
        Encode a complete game state into a fixed-length feature vector.
        
        Creates a single 237-dimensional vector representation of the game state
        from the perspective of a given player. Also returns a legal action mask.
        
        Feature breakdown (237 total):
          - Board features (20): 4 rows x 5 max positions, values normalized [0,1]
          - Hand features (104): One-hot encoding of cards in hand
          - Visible cards (104): One-hot of all cards ever seen or on board
          - Score features (4): Normalized cumulative scores for all players
          - Round feature (1): Current round normalized to [0,1]
          - Bullhead features (4): Cumulative bullhead points per row normalized [0,1]
        
        Args:
            history (dict): Game state containing:
                - 'board': Current board (list of 4 rows)
                - 'scores': Current scores (list of 4 ints)
                - 'round': Current round number (0-9)
                - 'history_matrix': Cards played in past rounds
                - 'board_history': Board state history
            current_hand (list): Cards in current player's hand (1-104)
            player_idx (int): Index of the player we're encoding for (0-3)
        
        Returns:
            tuple: (state_vec, mask)
                   state_vec: 237-d numpy float32 vector for network input
                   mask: 104-d binary vector, 1.0 for cards in hand (legal moves)
        """
        # BOARD REPRESENTATION (20 features)
        # Flattened 4 rows x 5 max positions, normalized by max card value
        board = history.get('board', [])
        board_vec = np.zeros(N_ROWS * MAX_ROW_LEN, dtype=np.float32)
        for r, row in enumerate(board):
            for c, card in enumerate(row):
                if c < MAX_ROW_LEN and r < N_ROWS:
                    # Normalize card values to [0, 1] range
                    board_vec[r * MAX_ROW_LEN + c] = card / 104.0
            
        # HAND REPRESENTATION (104 features)
        # Bag-of-words encoding: 1.0 if card in hand, 0.0 otherwise
        hand_vec = np.zeros(N_CARDS, dtype=np.float32)
        for card in current_hand:
            hand_vec[card - 1] = 1.0  # Convert 1-indexed card to 0-indexed vector
            
        # SCORE REPRESENTATION (4 features)
        # Organize scores from player perspective
        # Position 0: current player's score
        # Positions 1-3: other players' scores
        scores = history.get('scores', [0] * N_PLAYERS)
        my_score = scores[player_idx]
        score_vec = np.zeros(N_PLAYERS, dtype=np.float32)
        score_vec[0] = my_score / 100.0  # Normalize by approximate max penalty
        o_idx = 1
        for i, s in enumerate(scores):
            if i != player_idx:
                score_vec[o_idx] = s / 100.0
                o_idx += 1
                
        # VISIBLE CARDS REPRESENTATION (104 features)
        # Track which cards have been played or are on board
        # This helps the network infer opponent hands
        visible_vec = np.zeros(N_CARDS, dtype=np.float32)
        
        # Add current board cards
        for row in board:
            for c in row:
                visible_vec[c - 1] = 1.0
                
        # Add cards from past rounds
        for past_round in history.get('history_matrix', []):
            for c in past_round:
                visible_vec[c - 1] = 1.0
                
        # Add cards from initial board setup
        if history.get('board_history'):
            for row in history.get('board_history')[0]:
                for c in row:
                    visible_vec[c - 1] = 1.0
                    
        # Add player's hand (cards visible to this player)
        for c in current_hand:
            visible_vec[c - 1] = 1.0
            
        # ROUND NUMBER REPRESENTATION (1 feature)
        # Current round normalized to [0, 1] range
        # 6 Nimmt! has 10 rounds: 0-9
        round_idx = history.get('round', 0)
        round_vec = np.array([round_idx / 10.0], dtype=np.float32)
        
        # BULLHEAD FEATURES (4 features)
        # Cumulative bullhead points per row, normalized by max possible (7*5=35 per row)
        bullhead_vec = np.zeros(N_ROWS, dtype=np.float32)
        for r, row in enumerate(board):
            row_bullheads = sum(get_bullheads(card) for card in row)
            # Normalize: max 5 cards * 7 bullheads = 35 per row
            bullhead_vec[r] = min(row_bullheads / 35.0, 1.0)
        
        # CONCATENATE ALL FEATURES
        state_vec = np.concatenate([
            board_vec,       # 20 features
            hand_vec,        # 104 features
            visible_vec,     # 104 features
            score_vec,       # 4 features
            round_vec,       # 1 feature
            bullhead_vec     # 4 features
        ])                   # Total: 237 features
        
        # LEGAL ACTION MASK
        # Binary mask indicating which cards can be legally played
        mask = np.zeros(N_CARDS, dtype=np.float32)
        for card in current_hand:
            mask[card - 1] = 1.0  # 1.0 = legal, 0.0 = illegal
            
        return state_vec, mask


def get_state_dim():
    """
    Get the feature vector dimension for the encoded state.
    
    Computes the total size of the state vector:
      - Board: 4 rows x 5 max cards = 20
      - Hand: 104 cards = 104
      - Visible: 104 cards = 104
      - Scores: 4 players = 4
      - Round: 1 value = 1
      - Bullheads: 4 rows = 4
      - Total: 237
    
    Returns:
        int: State vector dimension (237)
    """
    # 20 + 104 + 104 + 4 + 1 + 4 = 237
    return 20 + N_CARDS + N_CARDS + N_PLAYERS + 1 + N_ROWS
