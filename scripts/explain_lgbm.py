import os
import sys
import pickle
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import shap

sys.path.append(os.getcwd())
from src.players.b12705048.core.features import extract_features

FEATURE_NAMES = [
    # --- State Features (43 dims) ---
    "r1_len", "r1_top", "r1_bh",
    "r2_len", "r2_top", "r2_bh",
    "r3_len", "r3_top", "r3_bh",
    "r4_len", "r4_top", "r4_bh",
    "max_row_bh", "min_row_bh", "turn_number",
    "my_score", "opp1_score", "opp2_score", "opp3_score",
    "my_rank", "score_spread",
    "unseen_ratio", "unseen_q1", "unseen_q2", "unseen_q3", "unseen_q4",
    "pressure_r1", "pressure_r2", "pressure_r3", "pressure_r4",
    "rounds_played", "opp_mean_card", "opp_var_card", "opp_low_plays", "opp_high_plays",
    "total_penalties", "avg_penalty_size", "board_volatility",
    "opp1_aggression", "opp2_aggression", "opp3_aggression", "min_penalty_rate", "max_penalty_rate",
    
    # --- Card Features (10 dims) ---
    "card_value", "card_bullheads", "is_under_board", 
    "dist_to_target_row", "unseen_cards_in_gap", "dist_to_next_row",
    "target_row_len", "target_row_bh", "cheapest_row_bh", "diff_to_lowest_tail"
]

def main():
    data_path = "results/dataset.pkl"
    print(f"Loading data from {data_path}...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
        
    print("Extracting features for 1000 turns to keep SHAP fast...")
    X, y, group = [], [], []
    
    for sample in data[:1000]:  # Use a subset for fast explanation
        if "scores" not in sample: continue
            
        unseen_set = set(sample["unseen_cards"])
        features_143 = extract_features(
            board=sample["board"], hand=sample["hand"], unseen=unseen_set,
            scores=sample["scores"], player_idx=0, round_num=sample["round_num"],
            history_matrix=sample["history_matrix"], score_history=sample["score_history"],
            board_history=sample["board_history"]
        )
        
        state_features = np.concatenate([features_143[0:15], features_143[115:143]])
        sorted_hand = sorted(sample["hand"])
        action_card = sample["action"]
        
        for slot in range(len(sorted_hand)):
            card_features = features_143[15 + slot*10 : 15 + slot*10 + 10]
            X.append(np.concatenate([state_features, card_features]))
            y.append(1 if sorted_hand[slot] == action_card else 0)
                
        group.append(len(sorted_hand))
        
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    group = np.array(group, dtype=np.int32)
    
    print("Training LightGBM Ranker on subset...")
    ranker = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", n_estimators=50, num_leaves=31)
    ranker.fit(X, y, group=group, feature_name=FEATURE_NAMES)
    
    os.makedirs("results/explain", exist_ok=True)
    
    # 1. Feature Importance (Split & Gain)
    print("Generating Feature Importance plots...")
    fig, ax = plt.subplots(figsize=(10, 8))
    lgb.plot_importance(ranker, max_num_features=20, importance_type='split', ax=ax, title="Feature Importance (Split)")
    plt.tight_layout()
    plt.savefig("results/explain/lgbm_importance_split.png")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    lgb.plot_importance(ranker, max_num_features=20, importance_type='gain', ax=ax, title="Feature Importance (Gain)")
    plt.tight_layout()
    plt.savefig("results/explain/lgbm_importance_gain.png")
    
    # 2. SHAP Values
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(ranker)
    shap_values = explainer.shap_values(X)
    
    print("Generating SHAP Summary plot...")
    plt.figure(figsize=(12, 8))
    # For LightGBM Ranker, shap_values might be a list or array. SHAP handles it.
    shap.summary_plot(shap_values, X, feature_names=FEATURE_NAMES, show=False)
    plt.tight_layout()
    plt.savefig("results/explain/shap_summary.png")
    
    print("Saved explanation plots to results/explain/")

if __name__ == "__main__":
    main()
