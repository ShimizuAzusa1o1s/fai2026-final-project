import numpy as np

def extract_penalty_events(history, target_round):
    """
    Reconstructs the round to identify penalty triggers.
    Returns a dictionary mapping player_idx to True if they triggered a penalty.
    """
    if target_round == 0:
        return {i: False for i in range(4)}
        
    scores_before = np.array(history['score_history'][target_round - 1])
    scores_after = np.array(history['score_history'][target_round])
    score_deltas = scores_after - scores_before
    
    if np.all(score_deltas == 0):
        return {i: False for i in range(4)}
    
    return {i: (score_deltas[i] > 0) for i in range(4)}

def build_value_feature_vector(history, target_round, player_idx, my_hand, unseen_cards, current_hand_size):
    """
    Builds the 232-dimensional input vector for the Value Network.
    Features:
    - Board (12)
    - Player State (3)
    - Opponent States (9)
    - Player Hand Mask (104)
    - Unseen Card Mask (104)
    """
    board = history['board_history'][target_round]
    
    # 1. Public Board State (12 dims)
    row_ends = [row[-1] for row in board]
    lengths = [len(row) for row in board]
    from src.players.b12705048.core.constants import BULLHEAD_LOOKUP
    bullheads = [sum(BULLHEAD_LOOKUP[c] for c in row) for row in board]
    board_features = row_ends + lengths + bullheads # 12 dims
    
    prev_penalties = extract_penalty_events(history, target_round - 1) if target_round > 0 else {i: False for i in range(4)}
    current_scores = history['score_history'][target_round - 1] if target_round > 0 else [0, 0, 0, 0]

    # 2. Player State (3 dims)
    player_features = [
        current_hand_size,
        current_scores[player_idx],
        float(prev_penalties[player_idx])
    ]

    # 3. Opponent State (9 dims)
    opp_features = []
    opp_indices = [i for i in range(4) if i != player_idx]
    for opp_idx in opp_indices:
        opp_features.extend([
            current_hand_size,
            current_scores[opp_idx],
            float(prev_penalties[opp_idx])
        ])

    # 4. Player Hand Mask (104 dims)
    hand_mask = np.zeros(104, dtype=np.float32)
    for c in my_hand:
        hand_mask[c - 1] = 1.0

    # 5. Unseen Card Mask (104 dims)
    unseen_mask = np.zeros(104, dtype=np.float32)
    for c in unseen_cards:
        unseen_mask[c - 1] = 1.0
        
    features = np.concatenate([
        np.array(board_features, dtype=np.float32),
        np.array(player_features, dtype=np.float32),
        np.array(opp_features, dtype=np.float32),
        hand_mask,
        unseen_mask
    ])
    
    assert len(features) == 232, f"Expected 232 features, got {len(features)}"
    return features

def build_value_target(history, target_round, player_idx):
    """
    The target is the TOTAL additional penalty the player accumulates from target_round to the end.
    = final_score - current_score
    """
    final_score = history['score_history'][-1][player_idx]
    current_score = history['score_history'][target_round - 1][player_idx] if target_round > 0 else 0
    
    # We predict the extra penalty they will take.
    return np.array([float(final_score - current_score)], dtype=np.float32)
