import os
import sys
import numpy as np
from tqdm import tqdm
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.engine import Engine
from src.players.b12705048.agents.flatmc_baseline import FlatMCBaseline
from src.players.b12705048.agents.oracle_flatmc import OracleFlatMC
from src.players.b12705048.models.opp_net.feature_extractor import build_feature_vector

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

def generate_distillation_games(num_games=10, save_path=None, oracle_time=0.1):
    if save_path is None:
        save_path = "distillation_dataset.npz"
        
    print(f"Generating {num_games} games. Oracle time limit: {oracle_time}s per decision.")
    
    # We generate games using baseline agents to get a diverse set of states
    players_pool = [FlatMCBaseline(player_idx=i, time_limit=0.01) for i in range(4)]
    
    cfg = {
        "n_players": 4,
        "n_rounds": 10,
        "timeout": 1.0,
        "verbose": False
    }
    
    X_list = []  # Public features
    H_list = []  # One-hot hand mask
    Y_list = []  # Oracle target action (best card)
    
    total_cards = set(range(1, 105))
    
    for game_id in tqdm(range(num_games)):
        players = []
        for i in range(4):
            players.append(players_pool[i])
                
        engine = Engine(cfg, players)
        scores, history = engine.play_game()
        
        history_matrix = engine.history_matrix
        board_history = engine.board_history
        # Process rounds 0 to 8
        for r in range(9):
            board = board_history[r]
            
            history_dict = {
                'board': board,
                'round': r,
                'history_matrix': history_matrix[:r],
                'board_history': board_history[:r+1],
                'score_history': engine.score_history[:r+1]
            }
            
            for p_idx in range(4):
                my_hand = get_player_hand_at_round(history_matrix, r, p_idx)
                
                # Input features (334-dim V2)
                X = build_feature_vector(history_dict, r, p_idx, my_hand)
                
                # True opponent hands
                opp_hands = [get_player_hand_at_round(history_matrix, r, opp_idx) for opp_idx in range(4) if opp_idx != p_idx]
                
                # Ask Oracle for the true optimal move
                oracle = OracleFlatMC(player_idx=p_idx, time_limit=oracle_time, debug=False)
                best_card, _, _ = oracle.action(my_hand, history_dict, true_opp_hands=opp_hands)
                
                X_list.append(X)
                Y_list.append(best_card)
                
    X_arr = np.array(X_list, dtype=np.float32)
    Y_arr = np.array(Y_list, dtype=np.int64)
    
    print(f"\nGenerated {len(X_arr)} state-action pairs.")
    print(f"X shape: {X_arr.shape}")
    print(f"Y shape: {Y_arr.shape}")
    
    np.savez_compressed(save_path, X=X_arr, Y=Y_arr)
    print(f"Saved distillation dataset to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Knowledge Distillation dataset.")
    parser.add_argument("--games", type=int, default=10, help="Number of games to simulate.")
    parser.add_argument("--out", type=str, default="distillation_dataset.npz", help="Output .npz file path.")
    parser.add_argument("--oracle_time", type=float, default=0.1, help="Simulation budget for Oracle per decision.")
    
    args = parser.parse_args()
    generate_distillation_games(num_games=args.games, save_path=args.out, oracle_time=args.oracle_time)
