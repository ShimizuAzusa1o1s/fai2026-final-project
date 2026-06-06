import os
import sys
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.engine import Engine
from src.players.b12705048.agents.flatmc import FlatMC
from src.players.b12705048.models.feature_extractor import (
    build_feature_vector,
    build_target_matrix,
    get_gap_capacities,
    get_topological_gaps
)

# Import other agents for chaos training
from src.players.b12705048.agents.flatmc_baseline import FlatMCBaseline
from src.players.b12705048.agents.greedy import Maximizer
from src.players.TA.public_baselines2 import Baseline10

def get_player_hand_at_round(history_matrix, target_round, player_idx):
    """
    Since each player plays exactly 1 card per round, their hand at `target_round`
    consists entirely of the cards they play from `target_round` up to the end of the game.
    """
    hand = []
    for r in range(target_round, len(history_matrix)):
        hand.append(history_matrix[r][player_idx])
    return hand

def get_unseen_cards_at_round(total_cards, board, hand, history_matrix, target_round):
    """
    Unseen cards are total cards minus cards on the board, minus my own hand,
    minus cards already played by opponents in previous rounds.
    """
    visible = set()
    for row in board:
        visible.update(row)
    
    visible.update(hand)
    
    # Add cards played in previous rounds
    for r in range(target_round):
        visible.update(history_matrix[r])
        
    return list(total_cards - visible)

def generate_games(num_games=10, save_path="dataset.npz"):
    print(f"Generating {num_games} games using mixed-opponent (chaos) self-play...")
    
    # Initialize pools of different players to mix and match per game
    print("Initializing player pools...")
    flatmc_pool = [FlatMC(player_idx=i, time_limit=0.8) for i in range(4)]
    baseline_pool = [FlatMCBaseline(player_idx=i, time_limit=0.1) for i in range(4)]
    maximizer_pool = [Maximizer(player_idx=i) for i in range(4)]
    
    # Suppress output during Baseline10 init to avoid spam
    from src.engine import silenced_if
    with silenced_if(True):
        b10_pool = [Baseline10(player_idx=i) for i in range(4)]
    
    cfg = {
        "n_players": 4,
        "n_rounds": 10,
        "timeout": 1.0,
        "verbose": False
    }
    
    X_list = []
    Y_list = []
    C_list = [] # Capacities
    
    total_cards = set(range(1, 105))
    
    for game_id in tqdm(range(num_games)):
        # Randomly select a player from the pools for each seat
        players = []
        for i in range(4):
            r = np.random.rand()
            if r < 0.4:
                players.append(flatmc_pool[i])
            elif r < 0.6:
                players.append(baseline_pool[i])
            elif r < 0.8:
                players.append(maximizer_pool[i])
            else:
                players.append(b10_pool[i])
                
        engine = Engine(cfg, players)
        # Note: the engine automatically resets and plays
        scores, history = engine.play_game()
        
        history_matrix = engine.history_matrix
        board_history = engine.board_history
        
        # Construct the history dict exactly as feature_extractor expects it
        history_dict = {
            'history_matrix': history_matrix,
            'board_history': board_history,
            'score_history': engine.score_history
        }
        
        # We process rounds 0 to 8 (in round 9, everyone has 1 card, so predicting it is trivial and determinism breaks)
        for r in range(9):
            board = board_history[r]
            sorted_row_ends = get_topological_gaps(board)
            
            for p_idx in range(4):
                my_hand = get_player_hand_at_round(history_matrix, r, p_idx)
                unseen = get_unseen_cards_at_round(total_cards, board, my_hand, history_matrix, r)
                
                # Input features
                X = build_feature_vector(history_dict, r, p_idx, unseen, len(my_hand))
                
                # Capacities
                capacities = get_gap_capacities(sorted_row_ends, unseen)
                
                # True opponent hands
                opp_indices = [i for i in range(4) if i != p_idx]
                opp_hands = [get_player_hand_at_round(history_matrix, r, opp_idx) for opp_idx in opp_indices]
                
                # Target matrix
                Y = build_target_matrix(board, opp_hands)
                
                X_list.append(X)
                Y_list.append(Y)
                C_list.append(capacities)
                
    X_arr = np.array(X_list, dtype=np.float32)
    Y_arr = np.array(Y_list, dtype=np.float32)
    C_arr = np.array(C_list, dtype=np.float32)
    
    print(f"\nGenerated {len(X_arr)} samples.")
    print(f"X shape: {X_arr.shape}")
    print(f"Y shape: {Y_arr.shape}")
    print(f"C shape: {C_arr.shape}")
    
    np.savez_compressed(save_path, X=X_arr, Y=Y_arr, C=C_arr)
    print(f"Saved dataset to {save_path}")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ML dataset using FlatMCSH self-play.")
    parser.add_argument("--games", type=int, default=10, help="Number of games to simulate.")
    parser.add_argument("--out", type=str, default="dataset.npz", help="Output .npz file path.")
    
    args = parser.parse_args()
    
    generate_games(num_games=args.games, save_path=args.out)
