"""
Self-Play Data Generation Module
==================================

This module generates training data by running self-play games where AlphaZero
agents compete against each other. The collected game trajectories (states,
action probabilities, and final outcomes) are saved for supervised training.

Key components:
  - self_play_episode(): Runs multiple games and collects training data
  - generate_parallel(): Distributes self-play across multiple processes
"""

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
from src.players.b12705048.alphazero.alphazero import AlphaZeroPlayer
from src.players.b12705048.alphazero.state_encoding import Encoding, get_state_dim, N_CARDS

def self_play_episode(model_path=None, num_games=10, timeout=1.0, iteration=None, num_playouts=500):
    """
    Run self-play games and collect training data.
    
    This function executes multiple complete games of 6 Nimmt! with AlphaZero agents
    playing against each other. During each game, we record:
      - Game states (encoded board, scores, hands)
      - Action probabilities from MCTS/policy
      - Game outcomes (final scores converted to value targets)
    
    These trajectories become the supervised training signal for the neural network.
    
    Args:
        model_path (str, optional): Path to a trained model checkpoint to load.
                                   If None, agents use random initialization.
        num_games (int): Number of games to play in this episode. Default is 10.
        timeout (float): Time limit per move decision in seconds. Default is 1.0.
        iteration (int, optional): Training iteration number for adaptive MCTS playouts.
                                  If provided, playouts are reduced in early iterations.
        num_playouts (int): MCTS playouts per move. Default is 500.
    
    Returns:
        list: Training data tuples of form (state_vec, mask, target_probs, value)
              where state_vec is the encoded game state, mask indicates legal moves,
              target_probs are MCTS-improved policy targets, and value is the
              normalized final outcome from this player's perspective.
    """
    state_dim = get_state_dim()
    n_actions = N_CARDS
    
    # Create four AlphaZero agents (one per player)
    # n_playouts controls how much MCTS search to do per move
    # iteration parameter enables adaptive playouts (fewer playouts in early training)
    players = [AlphaZeroPlayer(i, model_path, n_playouts=num_playouts, time_limit=timeout, iteration=iteration) for i in range(4)]
    
    # Configure the game engine for 4-player 10-round games
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
        
        # Accumulate training data across all rounds of this game
        game_data = []  # Format: (state_vec, mask, target_probs, player_idx) during collection
        
        # Play all rounds of the game
        for round_idx in range(engine.n_rounds):
            # Save board state for history encoding
            engine.board_history.append([row.copy() for row in engine.board])
            
            # Build immutable view of current game state
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
            
            # Collect actions: Each player independently decides their move
            # using MCTS search over the policy network
            round_data = []  # Temporary storage: (state_vec, mask, target_probs, player_idx)
            
            # FIRST PASS: Query all players without modifying game state
            # This ensures all players act on the same board configuration
            for p_idx, player in enumerate(engine.players):
                hand = engine.hands[p_idx].copy()
                
                # Get action and MCTS-improved policy from AlphaZero
                # Temperature=1.0 for self-play exploration (stochastic selection)
                best_action, target_probs = player.get_action_probs(history_state, hand, temperature=1.0)
                
                # Encode the game state from this player's perspective
                state_vec, mask = Encoding.encode_state(history_state, hand, p_idx)
                round_data.append((state_vec, mask, target_probs, p_idx))
                
                # Store the action for execution
                round_actions[p_idx] = best_action
                current_played_cards.append((best_action, p_idx))
                
            # SECOND PASS: Apply all mutations after all actions are collected
            # This ensures deterministic card removal
            for card, p_idx in current_played_cards:
                engine.hands[p_idx].remove(card)
                
            # Update history for future rounds
            engine.history_matrix.append(round_actions)
            engine.flags_matrix.append(round_flags)
            
            # Process card placements in sorted order (card value order)
            current_played_cards.sort(key=lambda x: x[0])
            for card, p_idx in current_played_cards:
                engine.process_card_placement(card, p_idx)
                
            # Record end-of-round scores
            engine.score_history.append(list(engine.scores))
            engine.round += 1
            game_data.extend(round_data)
            
        # After game ends: Convert player indices to value targets
        # Value represents the relative game outcome from that player's perspective
        final_scores = engine.scores
        for i in range(len(game_data)):
            state_vec, mask, target_probs, p_idx = game_data[i]
            
            # Calculate value: relative score comparison
            # In 6 Nimmt!, lower scores are better (penalty points)
            my_score = final_scores[p_idx]
            # Average opponent score
            opp_scores = sum(final_scores[j] for j in range(4) if j != p_idx) / 3.0
            # Difference: positive if we did better (lower score)
            diff = opp_scores - my_score
            # Normalize to [-1, 1] range
            val = max(-1.0, min(1.0, diff / 50.0))
            
            # Replace player index with value in the data tuple
            game_data[i] = (state_vec, mask, target_probs, val)
            
        # Accumulate game data for return
        data.extend(game_data)
            
    return data

def generate_parallel(num_games=10, n_jobs=4, model_path=None, iteration=None, num_playouts=500, timeout=1.0):
    """
    Generate self-play data using parallel job scheduling.
    
    Distributes game generation across multiple CPU processes to speed up
    data collection. Each process independently runs self_play_episode and
    returns collected training data, which are aggregated and returned.
    
    Args:
        num_games (int): Total number of games to generate. Default is 10.
        n_jobs (int): Number of parallel processes to use. Default is 4.
        model_path (str, optional): Path to model checkpoint. If None, uses
                                   random initialization.
        iteration (int, optional): Training iteration number for adaptive MCTS playouts.
        num_playouts (int): MCTS playouts per move. Default is 500.
        timeout (float): Time limit per move in seconds. Default is 1.0.
    
    Returns:
        list: Aggregated training data from all parallel processes.
    """
    if n_jobs == 1:
        # Single-process mode: useful for debugging
        return self_play_episode(model_path, num_games, timeout=timeout, iteration=iteration, num_playouts=num_playouts)
        
    # Divide games evenly across workers
    games_per_job = num_games // n_jobs
    
    # Launch parallel jobs using joblib's Parallel backend
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(self_play_episode)(model_path, games_per_job, timeout=timeout, iteration=iteration, num_playouts=num_playouts) for _ in range(n_jobs)
    )
    
    # Aggregate results from all processes
    all_data = []
    for res in results:
        if res is not None:
            all_data.extend(res)
        
    return all_data


if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments from train_loop.sh
    parser = argparse.ArgumentParser(description="Generate self-play training data")
    parser.add_argument("--num_games", type=int, default=100,
                        help="Number of games to generate")
    parser.add_argument("--num_parallel_jobs", type=int, default=10,
                        help="Number of parallel processes")
    parser.add_argument("--num_playouts", type=int, default=500,
                        help="MCTS playouts per move (passed to AlphaZeroPlayer)")
    parser.add_argument("--time_limit", type=float, default=1.0,
                        help="Time limit per move in seconds")
    parser.add_argument("--c_puct", type=float, default=1.5,
                        help="PUCT exploration constant (passed to MCTS)")
    parser.add_argument("--iteration", type=int, default=None,
                        help="Training iteration number for adaptive MCTS playouts")
    args = parser.parse_args()
    
    # Create output directories if needed
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Determine model path: use latest checkpoint if it exists
    model_path = "models/latest.pt" if os.path.exists("models/latest.pt") else None
    
    print(f"Generating self-play data...")
    print(f"  Games: {args.num_games}, Parallel jobs: {args.num_parallel_jobs}")
    print(f"  Playouts: {args.num_playouts}, Time limit: {args.time_limit}s")
    print(f"  Model: {model_path or '(random initialization)'}")
    if args.iteration is not None:
        print(f"  Iteration: {args.iteration} (adaptive playouts enabled)")
    
    # Generate training data from self-play games
    data = generate_parallel(
        num_games=args.num_games,
        n_jobs=args.num_parallel_jobs,
        model_path=model_path,
        iteration=args.iteration,
        num_playouts=args.num_playouts,
        timeout=args.time_limit
    )
    
    print(f"Generated {len(data)} training examples ({len(data)//40 if len(data) else 0} games)")
    
    # Save training data to disk for use by train.py
    torch.save(data, "data/self_play_data.pt")
    print("Saved to data/self_play_data.pt")
