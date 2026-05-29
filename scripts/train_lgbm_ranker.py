"""
Training Script for the LightGBM LambdaMART Ranker.

This script loads game-state samples collected by ``generate_dataset.py``,
extracts point-wise features (State + Candidate Card), groups them into queries,
trains a LightGBM Ranker (lambdarank objective), and exports the trees
into a NumPy ``.npz`` format for pure-NumPy vectorised inference.
"""

import os
import sys
import json
import pickle
import numpy as np
import lightgbm as lgb

sys.path.append(os.getcwd())
from src.players.b12705048.core.features import extract_features

def parse_lgbm_tree_to_numpy(model, out_path="src/players/b12705048/agents/lgbm_model.npz"):
    """
    Parses a LightGBM model JSON dump into dense NumPy arrays for fast inference.
    """
    model_json = model.booster_.dump_model()
    tree_info = model_json["tree_info"]
    n_trees = len(tree_info)
    
    # First pass: find the maximum number of nodes across all trees to pre-allocate arrays
    max_nodes = 0
    for tree in tree_info:
        num_leaves = tree['num_leaves']
        # For a binary tree, max nodes = 2 * num_leaves - 1
        nodes = 2 * num_leaves - 1
        if nodes > max_nodes:
            max_nodes = nodes

    # Pre-allocate dense arrays
    # -1 represents a leaf node in children_left/right
    children_left = np.full((n_trees, max_nodes), -1, dtype=np.int32)
    children_right = np.full((n_trees, max_nodes), -1, dtype=np.int32)
    feature = np.zeros((n_trees, max_nodes), dtype=np.int32)
    threshold = np.zeros((n_trees, max_nodes), dtype=np.float32)
    value = np.zeros((n_trees, max_nodes), dtype=np.float32)

    for tree_idx, tree in enumerate(tree_info):
        structure = tree["tree_structure"]
        current_node_idx = 0
        
        # BFS or DFS to flatten the tree
        def traverse(node, node_id):
            nonlocal current_node_idx
            
            if "split_feature" in node:
                # It's an internal node
                feature[tree_idx, node_id] = node["split_feature"]
                threshold[tree_idx, node_id] = node["threshold"]
                
                # Allocate left child
                current_node_idx += 1
                left_id = current_node_idx
                children_left[tree_idx, node_id] = left_id
                traverse(node["left_child"], left_id)
                
                # Allocate right child
                current_node_idx += 1
                right_id = current_node_idx
                children_right[tree_idx, node_id] = right_id
                traverse(node["right_child"], right_id)
            else:
                # It's a leaf
                value[tree_idx, node_id] = node["leaf_value"]

        traverse(structure, 0)
        
    np.savez_compressed(
        out_path,
        children_left=children_left,
        children_right=children_right,
        feature=feature,
        threshold=threshold,
        value=value,
        n_estimators=np.array([n_trees], dtype=np.int32)
    )
    print(f"NumPy Model successfully saved to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="results/dataset.pkl")
    parser.add_argument("--out", type=str, default="src/players/b12705048/agents/lgbm_model.npz")
    parser.add_argument("--estimators", type=int, default=100)
    args = parser.parse_args()
    
    print(f"Loading data from {args.data}...")
    try:
        with open(args.data, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {args.data} not found.")
        sys.exit(1)
        
    print(f"Loaded {len(data)} samples. Extracting LambdaMART queries...")
    
    X = []
    y = []
    group = []
    
    for sample in data:
        if "scores" not in sample:
            continue
            
        unseen_set = set(sample["unseen_cards"])
        features_143 = extract_features(
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
        
        # State features: 0-14 and 115-142 (43 dims)
        state_features = np.concatenate([features_143[0:15], features_143[115:143]])
        
        sorted_hand = sorted(sample["hand"])
        n_hand = len(sorted_hand)
        action_card = sample["action"]
        
        # Generate K query items for this turn
        for slot in range(n_hand):
            card_features = features_143[15 + slot*10 : 15 + slot*10 + 10]
            pointwise_features = np.concatenate([state_features, card_features])
            
            X.append(pointwise_features)
            # Label = 1 if it's the played card, else 0
            if sorted_hand[slot] == action_card:
                y.append(1)
            else:
                y.append(0)
                
        # Group size is the number of cards in hand
        group.append(n_hand)
        
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    group = np.array(group, dtype=np.int32)
    
    print(f"Dataset Shape: {X.shape}, Num Queries: {len(group)}")
    
    print(f"Training LightGBM Ranker with {args.estimators} estimators...")
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=args.estimators,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42
    )
    
    ranker.fit(
        X, y,
        group=group
    )
    
    print("Exporting trees to NumPy format...")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    parse_lgbm_tree_to_numpy(ranker, args.out)
