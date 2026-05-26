"""
Training Script for the FlatMC Random Forest Rollout Policy.

Usage:
    python scripts/train_rf_model.py [--data PATH] [--out PATH] [--estimators N]

This script loads game-state samples collected by ``generate_rf_data.py``,
extracts **115-dimensional** feature vectors (15 board features + 10 card
slots × 10 heuristic values), trains a ``scikit-learn``
``RandomForestClassifier``, and exports the fitted trees to a compressed
NumPy ``.npz`` file that can be loaded at inference time without
scikit-learn.

The model predicts the *rank-in-hand* index (0–9) of the card to play in
the player's sorted hand, rather than a raw card value.  This makes the
prediction target invariant to the exact card numbers dealt, significantly
improving generalisation.
"""

import os
import sys
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.append(os.getcwd())
from src.engine import Engine

# Module-level bullhead lookup (shared by calculate_row_score and extract_features).
# Mirrors the lookup table constructed in FlatMC.__init__ and RFPlayer._init_bullheads.
bullheads = [0] * 105
for card in range(1, 105):
    if card == 55:
        bullheads[card] = 7
    elif card % 11 == 0:
        bullheads[card] = 5
    elif card % 10 == 0:
        bullheads[card] = 3
    elif card % 5 == 0:
        bullheads[card] = 2
    else:
        bullheads[card] = 1

def calculate_row_score(row):
    """Return the total bullhead penalty for a board row."""
    return sum(bullheads[c] for c in row)

def extract_features(board, hand, unseen_cards):
    """
    Extract a 115-dimensional feature vector from a game state.

    This function must stay in exact sync with ``_extract_features_multi``
    in ``flat_mc.py`` and ``_extract_features`` in ``rf_player.py``.
    Any change to the feature layout here must be reflected in both
    inference implementations.

    Feature layout:
        [  0– 11]  Board: length, top card, penalty for each of 4 rows
                   (sorted ascending by top card).
        [12]       Board max row bullheads.
        [13]       Board min row bullheads.
        [14]       Turn number (hand size).
        [15–114]   Card Features: 10 slots (for sorted hand), each with 10 features:
                   1. Card Value
                   2. Card Bullheads
                   3. Is Under-Board (1 or 0)
                   4. Distance to Target Row
                   5. Unseen Cards in Gap
                   6. Distance to Next Closest Row
                   7. Target Row Length
                   8. Target Row Bullheads
                   9. Cheapest Available Row Bullheads
                   10. Difference to Lowest Tail

    Args:
        board (list[list[int]]): Current board rows.
        hand (list[int]): Current player hand.
        unseen_cards (list[int]): Cards unplayed and not in hand.

    Returns:
        list[float]: Feature vector of length 115.
    """
    sorted_board = sorted(board, key=lambda row: row[-1] if row else 0)
    
    board_features = []
    min_bullheads = 1000
    max_bullheads = -1
    
    row_lengths = []
    row_tops = []
    row_bullheads = []
    
    for row in sorted_board:
        if row:
            length = len(row)
            top = row[-1]
            b_heads = sum(bullheads[c] for c in row)
        else:
            length = 0
            top = 0
            b_heads = 0
            
        row_lengths.append(length)
        row_tops.append(top)
        row_bullheads.append(b_heads)
        
        board_features.extend([length, top, b_heads])
        if b_heads < min_bullheads: min_bullheads = b_heads
        if b_heads > max_bullheads: max_bullheads = b_heads
        
    if min_bullheads == 1000: min_bullheads = 0
    if max_bullheads == -1: max_bullheads = 0
    
    board_features.extend([max_bullheads, min_bullheads])
    turn_number = len(hand)
    board_features.append(turn_number)
    
    card_features = []
    sorted_hand = sorted(hand)
    unseen_set = set(unseen_cards)
    
    for i in range(10):
        if i < len(sorted_hand):
            card = sorted_hand[i]
            c_bullheads = bullheads[card]
            
            target_row_idx = -1
            max_val = -1
            for r in range(4):
                val = row_tops[r]
                if val < card and val > max_val:
                    max_val = val
                    target_row_idx = r
                    
            if target_row_idx != -1:
                is_under_board = 0
                target_tail = row_tops[target_row_idx]
                dist_to_target = card - target_tail
                
                unseen_in_gap = sum(1 for uc in unseen_set if target_tail < uc < card)
                
                next_closest_row_idx = -1
                max_val2 = -1
                for r in range(4):
                    val = row_tops[r]
                    if val < card and val > max_val2 and r != target_row_idx:
                        max_val2 = val
                        next_closest_row_idx = r
                
                if next_closest_row_idx != -1:
                    dist_to_next = card - row_tops[next_closest_row_idx]
                else:
                    dist_to_next = 1000
                    
                t_length = row_lengths[target_row_idx]
                t_bulls = row_bullheads[target_row_idx]
                cheap_avail = 0
                diff_to_lowest = 0
            else:
                is_under_board = 1
                dist_to_target = 1000
                unseen_in_gap = 1000
                dist_to_next = 1000
                t_length = 0
                t_bulls = 0
                
                cheap_avail = 1000
                for r in range(4):
                    if row_bullheads[r] < cheap_avail:
                        cheap_avail = row_bullheads[r]
                if cheap_avail == 1000: cheap_avail = 0
                
                min_tail = min([top for top in row_tops if top > 0] + [1000])
                if min_tail != 1000:
                    diff_to_lowest = min_tail - card
                else:
                    diff_to_lowest = 0
            
            card_features.extend([
                card, c_bullheads, is_under_board, dist_to_target,
                unseen_in_gap, dist_to_next, t_length, t_bulls,
                cheap_avail, diff_to_lowest
            ])
        else:
            card_features.extend([0] * 10)
            
    return board_features + card_features

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="results/rf_data.pkl")
    parser.add_argument("--out", type=str, default="src/players/b12705048/rf_model.npz")
    parser.add_argument("--estimators", type=int, default=50)
    args = parser.parse_args()
    
    print(f"Loading data from {args.data}...")
    try:
        with open(args.data, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {args.data} not found.")
        sys.exit(1)
        
    print(f"Loaded {len(data)} samples. Extracting features...")
    
    X = []
    y = []
    for sample in data:
        # Ignore old data without unseen_cards
        if "unseen_cards" not in sample:
            continue
        X.append(extract_features(sample["board"], sample["hand"], sample["unseen_cards"]))
        
        # Target variable is the index (0-9) of the played card in the sorted hand
        action_card = sample["action"]
        sorted_hand = sorted(sample["hand"])
        y.append(sorted_hand.index(action_card))
        
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    print(f"Feature matrix shape: {X.shape}, Target vector shape: {y.shape}")
    
    # Train the Random Forest.
    # max_depth=15 caps tree complexity to prevent overfitting and keeps
    # the exported NumPy arrays at a manageable size for fast inference.
    print(f"Training RandomForestClassifier with {args.estimators} estimators...")
    clf = RandomForestClassifier(n_estimators=args.estimators, max_depth=15, n_jobs=-1, random_state=42)
    clf.fit(X, y)
    print(f"Training accuracy: {clf.score(X, y):.4f}")
    
    # Export trees to NumPy arrays.
    # All trees are padded to the same node count so they can be stored in
    # dense (n_trees, max_nodes, ...) arrays, enabling vectorised traversal.
    print("Exporting trees to NumPy format...")
    tree_data = {}
    
    n_classes = len(clf.classes_)
    tree_data['classes'] = np.arange(10, dtype=np.int32)

    # Pad all trees to the same node count so we can store them in dense arrays.
    # Unused entries are filled with -1 (leaves) / 0 (thresholds/values).
    max_nodes = max(len(est.tree_.children_left) for est in clf.estimators_)
    n_trees = len(clf.estimators_)
    
    children_left_all = np.full((n_trees, max_nodes), -1, dtype=np.int32)
    children_right_all = np.full((n_trees, max_nodes), -1, dtype=np.int32)
    feature_all = np.zeros((n_trees, max_nodes), dtype=np.int32)
    threshold_all = np.zeros((n_trees, max_nodes), dtype=np.float32)
    # Output values are padded to always cover 10 class slots (0–9).
    value_all = np.zeros((n_trees, max_nodes, 10), dtype=np.float32)
    
    for i, estimator in enumerate(clf.estimators_):
        tree = estimator.tree_
        n_nodes = len(tree.children_left)
        
        children_left_all[i, :n_nodes] = tree.children_left
        children_right_all[i, :n_nodes] = tree.children_right
        feature_all[i, :n_nodes] = tree.feature
        threshold_all[i, :n_nodes] = tree.threshold
        
        values = tree.value[:, 0, :]
        row_sums = values.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # guard against empty leaves
        probas = values / row_sums
        
        for class_idx, class_val in enumerate(clf.classes_):
            value_all[i, :n_nodes, int(class_val)] = probas[:, class_idx]

    tree_data['children_left'] = children_left_all
    tree_data['children_right'] = children_right_all
    tree_data['feature'] = feature_all
    tree_data['threshold'] = threshold_all
    tree_data['value'] = value_all
    tree_data['n_estimators'] = np.array([n_trees])
    
    # Save as compressed .npz for compact on-disk size and fast np.load()
    np.savez_compressed(args.out, **tree_data)
    print(f"Model successfully saved to {args.out}")
