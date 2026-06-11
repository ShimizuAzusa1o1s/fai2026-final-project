import numpy as np
from src.players.b12705048.core.constants import BULLHEAD_LOOKUP

def build_student_feature_vector(history, target_round, player_idx, my_hand):
    """
    Builds the V2 334-dimensional input vector for the Student Distillation Network.
    """
    board = history['board'] if 'board' in history else history['board_history'][target_round]
    
    # 1. Complete Card State (312 dims)
    # ---------------------------------
    my_hand_mask = np.zeros(104, dtype=np.float32)
    for c in my_hand:
        my_hand_mask[c - 1] = 1.0

    board_mask = np.zeros(104, dtype=np.float32)
    for row in board:
        for c in row:
            board_mask[c - 1] = 1.0

    discard_mask = np.zeros(104, dtype=np.float32)
    if 'history_matrix' in history:
        for r in range(target_round):
            if r < len(history['history_matrix']):
                for p_idx in range(4):
                    c = history['history_matrix'][r][p_idx]
                    discard_mask[c - 1] = 1.0

    # 2. Topologically Sorted Board State (12 dims)
    # ---------------------------------------------
    # Sort the 4 rows by their tail value in ascending order
    sorted_board = sorted(board, key=lambda row: row[-1])
    row_ends = [row[-1] / 104.0 for row in sorted_board]
    lengths = [len(row) / 5.0 for row in sorted_board]
    bullheads = [sum(BULLHEAD_LOOKUP[c] for c in row) / 25.0 for row in sorted_board]
    
    board_features = np.array(row_ends + lengths + bullheads, dtype=np.float32)

    # 3. Opponent Behavioral State (9 dims)
    # --------------------------------------
    opp_features = []
    opp_indices = [i for i in range(4) if i != player_idx]
    
    # Penalty calculation
    prev_penalties = {i: False for i in range(4)}
    if target_round > 0 and 'score_history' in history and len(history['score_history']) > target_round:
        scores_before = history['score_history'][target_round - 1]
        scores_after = history['score_history'][target_round]
        for i in range(4):
            prev_penalties[i] = (scores_after[i] - scores_before[i] > 0)
            
    current_scores = history['score_history'][target_round] if 'score_history' in history and len(history['score_history']) > target_round else [0, 0, 0, 0]
    
    for opp_idx in opp_indices:
        norm_score = current_scores[opp_idx] / 100.0
        
        last_card = 0.0
        if target_round > 0 and 'history_matrix' in history and len(history['history_matrix']) >= target_round:
            last_card = history['history_matrix'][target_round - 1][opp_idx] / 104.0
            
        penalty_flag = 1.0 if prev_penalties[opp_idx] else 0.0
        
        opp_features.extend([norm_score, last_card, penalty_flag])
        
    opp_features = np.array(opp_features, dtype=np.float32)

    # 4. Game Context (1 dim)
    # -----------------------
    round_feature = np.array([target_round / 10.0], dtype=np.float32)

    # Combine all (312 + 12 + 9 + 1 = 334 dims)
    features = np.concatenate([
        my_hand_mask,
        board_mask,
        discard_mask,
        board_features,
        opp_features,
        round_feature
    ])
    
    return features
