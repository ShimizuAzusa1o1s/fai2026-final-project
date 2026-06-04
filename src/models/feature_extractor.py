import numpy as np

def get_topological_gaps(board):
    """
    Returns the 4 row ends sorted in ascending order.
    These define the boundaries of the 5 topological gaps.
    """
    row_ends = np.array([row[-1] for row in board])
    return np.sort(row_ends)

def assign_card_to_bucket(card, sorted_row_ends):
    """
    Assigns a single card to one of the 5 topological buckets (0-4).
    """
    # Bucket 0: cards < R1
    if card < sorted_row_ends[0]:
        return 0
    # Bucket 1: between R1 and R2
    elif card < sorted_row_ends[1]:
        return 1
    # Bucket 2: between R2 and R3
    elif card < sorted_row_ends[2]:
        return 2
    # Bucket 3: between R3 and R4
    elif card < sorted_row_ends[3]:
        return 3
    # Bucket 4: cards > R4
    else:
        return 4

def get_gap_capacities(sorted_row_ends, unseen_cards):
    """
    Calculates the number of available unplayed cards that fit into each of the 5 buckets.
    """
    capacities = np.zeros(5, dtype=np.int32)
    for card in unseen_cards:
        bucket = assign_card_to_bucket(card, sorted_row_ends)
        capacities[bucket] += 1
    return capacities

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
    
    # If no one took a penalty, we can skip reconstruction
    if np.all(score_deltas == 0):
        return {i: False for i in range(4)}
    
    # If there are score deltas, we know they took penalties, so we just use score_deltas
    # as the behavioral flag directly. Anyone with delta > 0 took a penalty!
    return {i: (score_deltas[i] > 0) for i in range(4)}

def build_feature_vector(history, target_round, player_idx, unseen_cards, current_hand_size):
    """
    Builds the 125-dimensional input vector for the neural network.
    """
    board = history['board_history'][target_round]
    
    # 1. Public Board State (12 dims)
    row_ends = [row[-1] for row in board]
    lengths = [len(row) for row in board]
    bullheads = [sum(c % 10 == 5 or c % 10 == 0 or c % 11 == 0 for c in row) for row in board] # Approximation, actual rules used in engine
    # Let's just use exact bullhead counts. Wait, bullhead_lookup is in constants.
    
    # We can just import BULLHEAD_LOOKUP
    from src.players.b12705048.core.constants import BULLHEAD_LOOKUP
    bullheads = [sum(BULLHEAD_LOOKUP[c] for c in row) for row in board]
    
    board_features = row_ends + lengths + bullheads # 12 dims
    
    # 2. Card Availability Mask (104 dims)
    card_mask = np.zeros(104, dtype=np.float32)
    for c in unseen_cards:
        card_mask[c - 1] = 1.0 # 0-indexed internally
        
    # 3. Opponent State (9 dims: 3 opponents * 3 features)
    opp_features = []
    opp_indices = [i for i in range(4) if i != player_idx]
    
    # Penalty events from the *previous* round
    prev_penalties = extract_penalty_events(history, target_round - 1) if target_round > 0 else {i: False for i in range(4)}
    
    current_scores = history['score_history'][target_round - 1] if target_round > 0 else [0, 0, 0, 0]
    
    for opp_idx in opp_indices:
        opp_features.extend([
            current_hand_size,              # Hand size remaining
            current_scores[opp_idx],        # Cumulative penalty
            float(prev_penalties[opp_idx])  # Behavioral flag
        ])
        
    features = np.concatenate([
        np.array(board_features, dtype=np.float32),
        card_mask,
        np.array(opp_features, dtype=np.float32)
    ])
    
    return features # 125 dims

def build_target_matrix(board, opp_hands):
    """
    Builds the 3x5 target probability matrix given the opponents' actual hands.
    opp_hands is a list of 3 lists (the hands of the 3 opponents).
    """
    sorted_row_ends = get_topological_gaps(board)
    targets = np.zeros((3, 5), dtype=np.float32)
    
    for i, hand in enumerate(opp_hands):
        if len(hand) == 0:
            continue
        for card in hand:
            bucket = assign_card_to_bucket(card, sorted_row_ends)
            targets[i, bucket] += 1
        # Normalize to probability
        targets[i] /= len(hand)
        
    return targets
