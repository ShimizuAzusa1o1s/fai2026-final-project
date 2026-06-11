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
from src.players.b12705048.models.student_net.feature_extractor import build_student_feature_vector

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
    
    import random
    from src.players.b12705048.agents.flatmc_cpp import FlatMCCPP
    from src.players.TA.public_baselines2 import Baseline10
    
    def build_l2_cpp(idx):
        return FlatMCCPP(player_idx=idx, time_limit=0.2, epsilon=0.2, tau=1.0, model_level=2, use_neural_determinization=True)
    def build_l1_cpp(idx):
        return FlatMCCPP(player_idx=idx, time_limit=0.2, epsilon=0.2, tau=1.0, model_level=1, use_neural_determinization=True)
    def build_b10(idx):
        return Baseline10(player_idx=idx)
    def build_flatmc_base(idx):
        return FlatMCBaseline(player_idx=idx, time_limit=0.1)
        
    builders = [build_l2_cpp, build_l1_cpp, build_b10, build_flatmc_base]
    weights = [0.4, 0.3, 0.2, 0.1]
    
    cfg = {
        "n_players": 4,
        "n_rounds": 10,
        "timeout": 1.0,
        "verbose": False
    }
    
    X_list = []  # Public features
    H_list = []  # One-hot hand mask
    Y_list = []  # Oracle target probabilities
    
    total_cards = set(range(1, 105))
    
    for game_id in tqdm(range(num_games)):
        players = []
        for i in range(4):
            builder = random.choices(builders, weights=weights, k=1)[0]
            players.append(builder(i))
                
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
                X = build_student_feature_vector(history_dict, r, p_idx, my_hand)
                
                # True opponent hands
                opp_hands = [get_player_hand_at_round(history_matrix, r, opp_idx) for opp_idx in range(4) if opp_idx != p_idx]
                
                # Ask Oracle for the true optimal move
                oracle = OracleFlatMC(player_idx=p_idx, time_limit=oracle_time, debug=False)
                best_card, stats_penalty, stats_visits = oracle.action(my_hand, history_dict, true_opp_hands=opp_hands)
                
                target_probs = np.zeros(105, dtype=np.float32)
                scores = {}
                for c in my_hand:
                    if stats_visits[c] > 0:
                        scores[c] = - (stats_penalty[c] / stats_visits[c])
                    else:
                        scores[c] = -10.0
                
                tau = 0.1
                max_score = max(scores.values())
                exp_scores = {c: np.exp((scores[c] - max_score) / tau) for c in my_hand}
                sum_exp = sum(exp_scores.values())
                
                for c in my_hand:
                    target_probs[c] = exp_scores[c] / sum_exp
                
                X_list.append(X)
                Y_list.append(target_probs)
                
    X_arr = np.array(X_list, dtype=np.float32)
    Y_arr = np.array(Y_list, dtype=np.float32)
    
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
