"""
Training Script for the FlatMC Random Forest Rollout Policy.

Usage:
    python scripts/train_rf_model.py [--data PATH] [--out PATH] [--estimators N]

This script loads game-state samples collected by ``generate_rf_data.py``,
extracts **143-dimensional** feature vectors (Deep CFR format), trains a ``scikit-learn``
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

from src.players.b12705048.features import extract_features

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
        # Ignore old data without scores (means old format)
        if "scores" not in sample:
            continue
            
        unseen_set = set(sample["unseen_cards"])
        features = extract_features(
            board=sample["board"],
            hand=sample["hand"],
            unseen=unseen_set,
            scores=sample["scores"],
            player_idx=0,
            round_num=sample["round_num"],
            history_matrix=sample["history_matrix"],
            score_history=sample["score_history"],
            board_history=sample["board_history"]
        )
        X.append(features)
        
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
