import os
import sys
import json
import torch
import numpy as np
import copy
from tqdm import tqdm
from joblib import Parallel, delayed

sys.path.append(os.getcwd())

from src.engine import Engine
from src.players.b12705048.alphazero import AlphaZeroPlayer
from src.players.b12705048.state_encoding import Encoding, get_state_dim, N_CARDS

def self_play_episode(model_path=None, num_games=10, timeout=1.0):
    """
    Run self-play games and collect training data.
    """
    state_dim = get_state_dim()
    n_actions = N_CARDS
    
    players = [AlphaZeroPlayer(i, model_path, n_playouts=200, time_limit=timeout) for i in range(4)]
    
    engine_config = {
        "n_players": 4,
        "n_rounds": 10,
        "verbose": False,
        "timeout": timeout,
        "timeout_buffer": 0.5
    }
    
    data = []
    
    for game in range(num_games):
        engine = Engine(engine_config, players)
        
        game_data = [] # Stores (state_vec, mask, target_probs, player_idx)
        
        # Override the game loop to inject data collection
        for round_idx in range(engine.n_rounds):
            engine.board_history.append([row.copy() for row in engine.board])
            
            history_state = {
                "board": engine.board,
                "scores": engine.scores,
                "round": engine.round,
                "history_matrix": engine.history_matrix,
                "board_history": engine.board_history,
                "score_history": engine.score_history,
            }
            
            current_played_cards = []
            round_actions = [0] * engine.n_players
            round_flags = [False] * engine.n_players
            
            # Step 1: Collect actions and save states
            round_data = [] # Stores (state_vec, mask, target_probs, player_idx)
            
            # FIRST PASS: Collect all actions purely without mutating state
            for p_idx, player in enumerate(engine.players):
                hand = engine.hands[p_idx].copy()
                
                # Get AlphaZero action and policy target
                best_action, target_probs = player.get_action_probs(history_state, hand, temperature=1.0)
                
                state_vec, mask = Encoding.encode_state(history_state, hand, p_idx)
                round_data.append((state_vec, mask, target_probs, p_idx))
                
                # Store intended action
                round_actions[p_idx] = best_action
                current_played_cards.append((best_action, p_idx))
                
            # SECOND PASS: Apply all mutations AFTER all actions are collected
            for card, p_idx in current_played_cards:
                engine.hands[p_idx].remove(card)
                
            engine.history_matrix.append(round_actions)
            engine.flags_matrix.append(round_flags)
            
            current_played_cards.sort(key=lambda x: x[0])
            for card, p_idx in current_played_cards:
                engine.process_card_placement(card, p_idx)
                
            engine.score_history.append(list(engine.scores))
            engine.round += 1
            game_data.extend(round_data)
            
        # After all rounds:
        final_scores = engine.scores
        for i in range(len(game_data)):
            state_vec, mask, target_probs, p_idx = game_data[i]
            
            my_score = final_scores[p_idx]
            opp_scores = sum(final_scores[j] for j in range(4) if j != p_idx) / 3.0
            diff = opp_scores - my_score
            val = max(-1.0, min(1.0, diff / 50.0))
            
            game_data[i] = (state_vec, mask, target_probs, val)
            
        data.extend(game_data)
            
    return data

def generate_parallel(num_games=10, n_jobs=4, model_path=None):
    if n_jobs == 1:
        # debugging mode
        return self_play_episode(model_path, num_games)
        
    games_per_job = num_games // n_jobs
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(self_play_episode)(model_path, games_per_job) for _ in range(n_jobs)
    )
    
    all_data = []
    for res in results:
        if res is not None:
            all_data.extend(res)
        
    return all_data

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    print("Generating self-play data...")
    data = generate_parallel(num_games=250, n_jobs=10, model_path=None)
    print(f"Generated {len(data)} training examples.")
    
    torch.save(data, "data/self_play_data.pt")
    print("Saved to data/self_play_data.pt")
