import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine import Engine
from src.players.b12705048.agents.flatmc_sh import FlatMCSH
from src.models.feature_extractor import (
    build_feature_vector,
    build_target_matrix,
    get_gap_capacities,
    get_topological_gaps
)

def get_player_hand_at_round(history_matrix, target_round, player_idx):
    hand = []
    for r in range(target_round, len(history_matrix)):
        hand.append(history_matrix[r][player_idx])
    return hand

def get_unseen_cards_at_round(total_cards, board, hand, history_matrix, target_round):
    visible = set()
    for row in board:
        visible.update(row)
    visible.update(hand)
    for r in range(target_round):
        visible.update(history_matrix[r])
    return list(total_cards - visible)

def generate_games(num_games=10, save_path="data/dataset_l1.npz"):
    print(f"Generating {num_games} games using FlatMCSH self-play (for Level 1 trainer)...")
    
    players = [FlatMCSH(player_idx=i, time_limit=0.1, epsilon=0.2, tau=5.0) for i in range(4)]
    
    cfg = {
        "n_players": 4,
        "n_rounds": 10,
        "timeout": 1.0,
        "verbose": False
    }
    
    X_list, Y_list, C_list = [], [], []
    total_cards = set(range(1, 105))
    
    for game_id in tqdm(range(num_games)):
        engine = Engine(cfg, players)
        scores, history = engine.play_game()
        
        history_matrix = engine.history_matrix
        board_history = engine.board_history
        history_dict = {
            'history_matrix': history_matrix,
            'board_history': board_history,
            'score_history': engine.score_history
        }
        
        for r in range(9):
            board = board_history[r]
            sorted_row_ends = get_topological_gaps(board)
            
            for p_idx in range(4):
                my_hand = get_player_hand_at_round(history_matrix, r, p_idx)
                unseen = get_unseen_cards_at_round(total_cards, board, my_hand, history_matrix, r)
                
                X = build_feature_vector(history_dict, r, p_idx, unseen, len(my_hand))
                capacities = get_gap_capacities(sorted_row_ends, unseen)
                
                opp_indices = [i for i in range(4) if i != p_idx]
                opp_hands = [get_player_hand_at_round(history_matrix, r, opp_idx) for opp_idx in opp_indices]
                Y = build_target_matrix(board, opp_hands)
                
                X_list.append(X)
                Y_list.append(Y)
                C_list.append(capacities)
                
    X_arr = np.array(X_list, dtype=np.float32)
    Y_arr = np.array(Y_list, dtype=np.float32)
    C_arr = np.array(C_list, dtype=np.float32)
    
    print(f"\nGenerated {len(X_arr)} samples.")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, X=X_arr, Y=Y_arr, C=C_arr)
    print(f"Saved dataset to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--out", type=str, default="data/dataset_l1.npz")
    args = parser.parse_args()
    generate_games(num_games=args.games, save_path=args.out)
