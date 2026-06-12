import numpy as np
from src.players.b12705048.core.constants import BULLHEAD_LOOKUP
from src.players.b12705048.core.utils import get_topological_gaps, assign_card_to_bucket


def build_opp_feature_vector(history, target_round, player_idx, unseen_cards, current_hand_size):
    """
    125-dim feature extractor for Opponent Network.
    """
    board = history['board_history'][target_round]
    row_ends = [float(row[-1]) for row in board]
    lengths = [float(len(row)) for row in board]
    bullheads = [float(sum(BULLHEAD_LOOKUP[c] for c in row)) for row in board]
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
            float(current_hand_size),
            float(current_scores[opp_idx]),
            float(prev_penalties[opp_idx])
        ])
        
    features = np.concatenate([
        np.array(board_features, dtype=np.float32),
        card_mask,
        np.array(opp_features, dtype=np.float32)
    ])
    return features


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
