"""
Game State Encoding for Neural Network Input (Enhanced)
========================================================

Converts 6 Nimmt! game states into fixed-length vectors suitable for neural network
inference. This is crucial for the AlphaZero architecture.

Enhanced encoding adds strategic features beyond the base representation:
  - Row lengths (how close to forced take)
  - Row end cards (explicit endpoints for placement logic)
  - Hand-to-board relationship features (placement distribution, low card danger)
  - Hand statistics (min, max, mean, std)

All features are normalized to reasonable ranges for neural network input.
"""

import numpy as np
from .fast_env import get_bullheads

# Game constants for 6 Nimmt!
N_CARDS = 104      # Total number of unique cards
N_ROWS = 4         # Number of rows on the board
MAX_ROW_LEN = 5    # Maximum cards per row before forced placement
N_PLAYERS = 4      # Number of players

# Feature dimensions breakdown:
#   Board:          4 rows × 5 positions = 20
#   Hand:           104 (one-hot)
#   Visible cards:  104 (one-hot)
#   Scores:         4
#   Round:          1
#   Bullheads/row:  4
#   Row lengths:    4  (NEW)
#   Row end cards:  4  (NEW)
#   Hand stats:     4  (NEW: min, max, mean, std)
#   Low card count: 1  (NEW)
#   Placement dist: 4  (NEW: fraction of hand cards targeting each row)
# Total: 237 + 17 = 254
STATE_DIM = N_CARDS * 3 + N_ROWS * MAX_ROW_LEN + N_PLAYERS * 1 + 1 + N_ROWS + N_ROWS + N_ROWS + 4 + 1 + N_ROWS


class Encoding:
    """
    Static class for encoding game states to neural network input vectors.
    
    Converts variable-length game history into a fixed-size feature vector
    that can be fed to the AlphaZeroNet for policy and value prediction.
    """
    
    @staticmethod
    def encode_state(history, current_hand, player_idx):
        """
        Encode a complete game state into a fixed-length feature vector.
        
        Creates a 254-dimensional vector representation of the game state
        from the perspective of a given player. Also returns a legal action mask.
        
        Feature breakdown (254 total):
          - Board features (20): 4 rows × 5 max positions, values normalized [0,1]
          - Hand features (104): One-hot encoding of cards in hand
          - Visible cards (104): One-hot of all cards ever seen or on board
          - Score features (4): Normalized cumulative scores for all players
          - Round feature (1): Current round normalized to [0,1]
          - Bullhead features (4): Cumulative bullhead points per row normalized [0,1]
          - Row length features (4): How full each row is [0,1]
          - Row end features (4): Last card value in each row normalized [0,1]
          - Hand statistics (4): min, max, mean, std of hand cards normalized
          - Low card fraction (1): Fraction of hand cards below all row ends
          - Placement distribution (4): Fraction of hand cards targeting each row
        
        Args:
            history (dict): Game state containing board, scores, round, history
            current_hand (list): Cards in current player's hand (1-104)
            player_idx (int): Index of the player we're encoding for (0-3)
        
        Returns:
            tuple: (state_vec, mask)
                   state_vec: 254-d numpy float32 vector for network input
                   mask: 104-d binary vector, 1.0 for cards in hand (legal moves)
        """
        # BOARD REPRESENTATION (20 features)
        # Flattened 4 rows × 5 max positions, normalized by max card value
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
        round_idx = history.get('round', 0)
        round_vec = np.array([round_idx / 10.0], dtype=np.float32)
        
        # BULLHEAD FEATURES (4 features)
        # Cumulative bullhead points per row, normalized
        bullhead_vec = np.zeros(N_ROWS, dtype=np.float32)
        for r, row in enumerate(board):
            row_bullheads = sum(get_bullheads(card) for card in row)
            bullhead_vec[r] = min(row_bullheads / 35.0, 1.0)
        
        # ========== NEW STRATEGIC FEATURES ==========
        
        # ROW LENGTH FEATURES (4 features)
        # How full each row is — critical for predicting 6th-card forced takes
        # 0.0 = 1 card (just started), 1.0 = 5 cards (next card forces take)
        row_len_vec = np.zeros(N_ROWS, dtype=np.float32)
        for r, row in enumerate(board):
            row_len_vec[r] = len(row) / MAX_ROW_LEN
        
        # ROW END FEATURES (4 features)
        # Explicit row endpoint values (most important for placement logic)
        row_end_vec = np.zeros(N_ROWS, dtype=np.float32)
        for r, row in enumerate(board):
            if row:
                row_end_vec[r] = row[-1] / N_CARDS
        
        # HAND STATISTICS (4 features)
        # Aggregate hand properties: min, max, mean, std (all normalized)
        hand_stats_vec = np.zeros(4, dtype=np.float32)
        if current_hand:
            hand_arr = np.array(current_hand, dtype=np.float32)
            hand_stats_vec[0] = hand_arr.min() / N_CARDS
            hand_stats_vec[1] = hand_arr.max() / N_CARDS
            hand_stats_vec[2] = hand_arr.mean() / N_CARDS
            hand_stats_vec[3] = hand_arr.std() / N_CARDS if len(current_hand) > 1 else 0.0
        
        # LOW CARD FRACTION (1 feature)
        # Fraction of hand cards below ALL row ends → triggers Low Card Rule
        low_card_vec = np.zeros(1, dtype=np.float32)
        if current_hand and board:
            row_ends = [row[-1] for row in board if row]
            if row_ends:
                min_end = min(row_ends)
                low_count = sum(1 for c in current_hand if c < min_end)
                low_card_vec[0] = low_count / len(current_hand)
        
        # PLACEMENT DISTRIBUTION (4 features)
        # For each row, what fraction of hand cards would be placed on it
        # This captures "how crowded is each row from my perspective"
        placement_vec = np.zeros(N_ROWS, dtype=np.float32)
        if current_hand and board:
            row_ends = [row[-1] for row in board if row]
            row_ends_sorted = sorted(enumerate(row_ends), key=lambda x: x[1])
            
            for card in current_hand:
                # Find which row this card would go to (highest end below card)
                best_row = -1
                for r_idx, end in row_ends_sorted:
                    if card > end:
                        best_row = r_idx
                # If best_row found, increment that row's count
                if best_row >= 0:
                    placement_vec[best_row] += 1.0
            
            if len(current_hand) > 0:
                placement_vec /= len(current_hand)
        
        # CONCATENATE ALL FEATURES
        state_vec = np.concatenate([
            board_vec,       # 20 features
            hand_vec,        # 104 features
            visible_vec,     # 104 features
            score_vec,       # 4 features
            round_vec,       # 1 feature
            bullhead_vec,    # 4 features
            row_len_vec,     # 4 features  (NEW)
            row_end_vec,     # 4 features  (NEW)
            hand_stats_vec,  # 4 features  (NEW)
            low_card_vec,    # 1 feature   (NEW)
            placement_vec,   # 4 features  (NEW)
        ])                   # Total: 254 features
        
        # LEGAL ACTION MASK
        mask = np.zeros(N_CARDS, dtype=np.float32)
        for card in current_hand:
            mask[card - 1] = 1.0
            
        return state_vec, mask


def get_state_dim():
    """
    Get the feature vector dimension for the encoded state.
    
    Returns:
        int: State vector dimension (254)
    """
    # 20 + 104 + 104 + 4 + 1 + 4 + 4 + 4 + 4 + 1 + 4 = 254
    return 20 + N_CARDS + N_CARDS + N_PLAYERS + 1 + N_ROWS + N_ROWS + N_ROWS + 4 + 1 + N_ROWS
