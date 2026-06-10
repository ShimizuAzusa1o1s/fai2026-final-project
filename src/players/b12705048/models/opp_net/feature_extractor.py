import numpy as np
from src.players.b12705048.core.constants import BULLHEAD_LOOKUP

def get_topological_gaps(board):
    """
    Legacy method for legacy targets
    """
    row_ends = np.array([row[-1] for row in board])
    return np.sort(row_ends)

def assign_card_to_bucket(card, sorted_row_ends):
    """
    Legacy method for legacy targets
    """
    if card < sorted_row_ends[0]: return 0
    elif card < sorted_row_ends[1]: return 1
    elif card < sorted_row_ends[2]: return 2
    elif card < sorted_row_ends[3]: return 3
    else: return 4

def get_gap_capacities(sorted_row_ends, unseen_cards):
    """
    Calculates the number of available unplayed cards that fit into each of the 5 buckets.
    """
    capacities = np.zeros(5, dtype=np.int32)
    for card in unseen_cards:
        bucket = assign_card_to_bucket(card, sorted_row_ends)
        capacities[bucket] += 1
    return capacities

def build_feature_vector_v1(history, target_round, player_idx, unseen_cards, current_hand_size):
    """
    Legacy 125-dim feature extractor.
    """
    board = history['board_history'][target_round]
    row_ends = [row[-1] for row in board]
    lengths = [len(row) for row in board]
    bullheads = [sum(BULLHEAD_LOOKUP[c] for c in row) for row in board]
    board_features = row_ends + lengths + bullheads 
    
    card_mask = np.zeros(104, dtype=np.float32)
    for c in unseen_cards:
        card_mask[c - 1] = 1.0 
        
    opp_features = []
    opp_indices = [i for i in range(4) if i != player_idx]
    
    prev_penalties = {i: False for i in range(4)}
    if target_round > 0 and 'score_history' in history and len(history['score_history']) > target_round:
        scores_before = history['score_history'][target_round - 1]
        scores_after = history['score_history'][target_round]
        for i in range(4):
            prev_penalties[i] = (scores_after[i] - scores_before[i] > 0)
    
    current_scores = history['score_history'][target_round - 1] if target_round > 0 else [0, 0, 0, 0]
    
    for opp_idx in opp_indices:
        opp_features.extend([
            current_hand_size,
            current_scores[opp_idx],
            float(prev_penalties[opp_idx])
        ])
        
    features = np.concatenate([
        np.array(board_features, dtype=np.float32),
        card_mask,
        np.array(opp_features, dtype=np.float32)
    ])
    return features

def build_feature_vector_v2(history, target_round, player_idx, my_hand):
    """
    Builds the V2 334-dimensional input vector for the neural network.
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
    row_ends = [row[-1] for row in sorted_board]
    lengths = [len(row) for row in sorted_board]
    bullheads = [sum(BULLHEAD_LOOKUP[c] for c in row) for row in sorted_board]
    
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

# Alias default to V2
build_feature_vector = build_feature_vector_v2

def build_target_matrix(board, opp_hands):
    """
    Legacy method for legacy targets
    """
    sorted_row_ends = get_topological_gaps(board)
    targets = np.zeros((3, 5), dtype=np.float32)
    
    for i, hand in enumerate(opp_hands):
        if len(hand) == 0:
            continue
        for card in hand:
            bucket = assign_card_to_bucket(card, sorted_row_ends)
            targets[i, bucket] += 1
        targets[i] /= len(hand)
        
    return targets
