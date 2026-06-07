import os
import sys
import numpy as np
from tqdm import tqdm
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from src.engine import Engine
from src.players.b12705048.agents.flatmc_cpp import FlatMCCPP
from src.players.b12705048.models.value_net.feature_extractor import build_value_feature_vector, build_value_target

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

def generate_games(num_games=10, save_path="value_dataset.npz"):
    print(f"Generating {num_games} games using highly optimized FlatMCCPP self-play...")
    
    # Use the ultimate baseline-crushing configuration (Explore=0.8)
    # tau=1.0, uniform_ratio=0.0, minmax_ratio=0.8, disable_heuristic_safe=True, epsilon_alpha=0.0
    print("Initializing FlatMCCPP players...")
    from src.engine import silenced_if
    with silenced_if(True):
        flatmccpp_pool = [
            FlatMCCPP(player_idx=i, time_limit=0.5, tau=1.0, uniform_ratio=0.0, 
                      minmax_ratio=0.8, disable_heuristic_safe=True, epsilon_alpha=0.0)
            for i in range(4)
        ]
    
    cfg = {
        "n_players": 4,
        "n_rounds": 10,
        "timeout": 1.0, # Generous timeout for data generation
        "verbose": False
    }
    
    X_list = []
    Y_list = []
    
    total_cards = set(range(1, 105))
    
    for game_id in tqdm(range(num_games)):
        # Pure self-play to generate high-quality tactical states
        players = flatmccpp_pool
                
        engine = Engine(cfg, players)
        scores, history = engine.play_game()
        
        history_matrix = engine.history_matrix
        board_history = engine.board_history
        
        history_dict = {
            'history_matrix': history_matrix,
            'board_history': board_history,
            'score_history': engine.score_history
        }
        
        # We extract states from rounds 0 to 8
        for r in range(9):
            board = board_history[r]
            
            for p_idx in range(4):
                my_hand = get_player_hand_at_round(history_matrix, r, p_idx)
                unseen = get_unseen_cards_at_round(total_cards, board, my_hand, history_matrix, r)
                
                # Input features (232 dims)
                X = build_value_feature_vector(history_dict, r, p_idx, my_hand, unseen, len(my_hand))
                
                # Target value (Expected future penalty)
                Y = build_value_target(history_dict, r, p_idx)
                
                X_list.append(X)
                Y_list.append(Y)
                
    X_arr = np.array(X_list, dtype=np.float32)
    Y_arr = np.array(Y_list, dtype=np.float32)
    
    print(f"\nGenerated {len(X_arr)} samples.")
    print(f"X shape: {X_arr.shape}")
    print(f"Y shape: {Y_arr.shape}")
    
    # Save the dataset
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, X=X_arr, Y=Y_arr)
    print(f"Saved dataset to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ML dataset for Value Network.")
    parser.add_argument("--games", type=int, default=10, help="Number of games to simulate.")
    parser.add_argument("--out", type=str, default="data/value_dataset.npz", help="Output .npz file path.")
    
    args = parser.parse_args()
    generate_games(args.games, args.out)
