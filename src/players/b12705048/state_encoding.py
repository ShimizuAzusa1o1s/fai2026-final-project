import numpy as np

# Encoding parameters
N_CARDS = 104
N_ROWS = 4
MAX_ROW_LEN = 5
N_PLAYERS = 4
STATE_DIM = N_CARDS * 3 + N_ROWS * MAX_ROW_LEN + N_PLAYERS * 1 + 1 # Rough size

class Encoding:
    @staticmethod
    def encode_state(history, current_hand, player_idx):
        """
        Creates a fixed-length numpy array suitable for PyTorch MLP.
        - current_hand: list of card numbers (1 to 104)
        - history: The same history dictionary passed to action()
        
        Outputs:
        - state_vec: A 1D float32 numpy array representing the game state.
        - mask: A 104-d binary array where mask[card-1] == 1 if card is legal to play.
        """
        # Board representation
        # Flattened board: 4 rows up to 5 cards = 20 elements (value / 104.0)
        board = history.get('board', [])
        board_vec = np.zeros(N_ROWS * MAX_ROW_LEN, dtype=np.float32)
        for r, row in enumerate(board):
            for c, card in enumerate(row):
                if c < MAX_ROW_LEN and r < N_ROWS:
                    board_vec[r * MAX_ROW_LEN + c] = card / 104.0
            
        # Hand representation: 104-d bag of words
        hand_vec = np.zeros(N_CARDS, dtype=np.float32)
        for card in current_hand:
            hand_vec[card - 1] = 1.0
            
        # Scores representation: Normalized scores
        # We put our score first, then others' scores relative to us
        scores = history.get('scores', [0] * N_PLAYERS)
        my_score = scores[player_idx]
        score_vec = np.zeros(N_PLAYERS, dtype=np.float32)
        score_vec[0] = my_score / 100.0 # Approximate normalization
        o_idx = 1
        for i, s in enumerate(scores):
            if i != player_idx:
                score_vec[o_idx] = s / 100.0
                o_idx += 1
                
        # Visible cards representation: 104-d
        # All cards seen played or on board
        visible_vec = np.zeros(N_CARDS, dtype=np.float32)
        
        # Add current board
        for row in board:
            for c in row:
                visible_vec[c - 1] = 1.0
                
        # Add past rounds
        for past_round in history.get('history_matrix', []):
            for c in past_round:
                visible_vec[c - 1] = 1.0
                
        # Add initial board cards
        if history.get('board_history'):
            for row in history.get('board_history')[0]:
                for c in row:
                    visible_vec[c - 1] = 1.0
                    
        # Add hand
        for c in current_hand:
            visible_vec[c - 1] = 1.0
            
        # Round number normalized (1-10 -> 0.1-1.0)
        round_idx = history.get('round', 0)
        round_vec = np.array([round_idx / 10.0], dtype=np.float32)
        
        state_vec = np.concatenate([
            board_vec,
            hand_vec,
            visible_vec,
            score_vec,
            round_vec
        ])
        
        # Legal mask
        mask = np.zeros(N_CARDS, dtype=np.float32)
        for card in current_hand:
            mask[card - 1] = 1.0
            
        return state_vec, mask

def get_state_dim():
    # 20 + 104 + 104 + 4 + 1 = 233
    return 20 + N_CARDS + N_CARDS + N_PLAYERS + 1
