"""
Model Evaluation Script
========================

Compares a candidate model against the current best model by playing
head-to-head games. Used in the training loop to prevent catastrophic
forgetting — a new model is only accepted if it demonstrates genuine
improvement.

Usage:
    python scripts/evaluate.py --new models/latest.pt --best models/best.pt
    python scripts/evaluate.py --new models/latest.pt --best models/best.pt --num_games 30 --threshold 0.55

Exit codes:
    0 = new model passed threshold (accept)
    1 = new model failed threshold (reject)
"""

import os
import sys
import argparse
import time

sys.path.append(os.getcwd())

from src.engine import Engine
from src.players.b12705048.alphazero.alphazero import AlphaZeroPlayer


def evaluate(new_model_path, best_model_path, num_games=20, time_limit=0.9, n_playouts=200):
    """
    Compare two models by playing head-to-head games.
    
    Runs num_games games with alternating seat positions to eliminate
    positional advantage. In each game, 2 players use the new model and
    2 use the best model.
    
    The new model "wins" a game if its 2 players have a lower average
    penalty score than the best model's 2 players.
    
    Args:
        new_model_path (str): Path to candidate model checkpoint
        best_model_path (str): Path to current best model checkpoint
        num_games (int): Number of evaluation games. Default is 20.
        time_limit (float): Time limit per move in seconds. Default is 0.9.
        n_playouts (int): MCTS playouts per move during evaluation. Default is 200.
    
    Returns:
        float: Win rate of the new model (0.0 to 1.0)
    """
    new_wins = 0
    new_total_score = 0
    best_total_score = 0
    
    for game_idx in range(num_games):
        # Alternate seat arrangement for fairness
        # Even games: new at seats 0,2; best at seats 1,3
        # Odd games:  best at seats 0,2; new at seats 1,3
        if game_idx % 2 == 0:
            players = [
                AlphaZeroPlayer(0, new_model_path, n_playouts=n_playouts, time_limit=time_limit, device='cpu'),
                AlphaZeroPlayer(1, best_model_path, n_playouts=n_playouts, time_limit=time_limit, device='cpu'),
                AlphaZeroPlayer(2, new_model_path, n_playouts=n_playouts, time_limit=time_limit, device='cpu'),
                AlphaZeroPlayer(3, best_model_path, n_playouts=n_playouts, time_limit=time_limit, device='cpu'),
            ]
            new_indices = [0, 2]
            best_indices = [1, 3]
        else:
            players = [
                AlphaZeroPlayer(0, best_model_path, n_playouts=n_playouts, time_limit=time_limit, device='cpu'),
                AlphaZeroPlayer(1, new_model_path, n_playouts=n_playouts, time_limit=time_limit, device='cpu'),
                AlphaZeroPlayer(2, best_model_path, n_playouts=n_playouts, time_limit=time_limit, device='cpu'),
                AlphaZeroPlayer(3, new_model_path, n_playouts=n_playouts, time_limit=time_limit, device='cpu'),
            ]
            new_indices = [1, 3]
            best_indices = [0, 2]
        
        engine_config = {
            'n_players': 4,
            'n_rounds': 10,
            'verbose': False,
            'timeout': time_limit + 0.5,  # Extra buffer for engine timeout
            'timeout_buffer': 1.0
        }
        
        engine = Engine(engine_config, players)
        scores, _ = engine.play_game()
        
        # Compare average scores (lower is better in 6 Nimmt!)
        new_avg = sum(scores[i] for i in new_indices) / 2.0
        best_avg = sum(scores[i] for i in best_indices) / 2.0
        
        new_total_score += new_avg
        best_total_score += best_avg
        
        won = new_avg < best_avg
        if won:
            new_wins += 1
            
        print(f"  Game {game_idx+1:2d}/{num_games}: New avg={new_avg:5.1f} vs Best avg={best_avg:5.1f} {'✓ WIN' if won else '✗ LOSS'}")
    
    win_rate = new_wins / num_games
    return win_rate, new_total_score / num_games, best_total_score / num_games


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate candidate model against current best")
    parser.add_argument("--new", required=True, help="Path to candidate model")
    parser.add_argument("--best", required=True, help="Path to current best model")
    parser.add_argument("--num_games", type=int, default=20, help="Number of evaluation games")
    parser.add_argument("--threshold", type=float, default=0.55, help="Win rate threshold to accept")
    parser.add_argument("--n_playouts", type=int, default=200, help="MCTS playouts per move")
    parser.add_argument("--time_limit", type=float, default=0.9, help="Time limit per move (seconds)")
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print(f"Model Evaluation: {args.new} vs {args.best}")
    print(f"  Games: {args.num_games}, Threshold: {args.threshold:.0%}")
    print(f"  Playouts: {args.n_playouts}, Time limit: {args.time_limit}s")
    print(f"{'='*60}")
    
    start = time.perf_counter()
    win_rate, new_avg, best_avg = evaluate(
        args.new, args.best, args.num_games, args.time_limit, args.n_playouts
    )
    elapsed = time.perf_counter() - start
    
    print(f"{'='*60}")
    print(f"Results ({elapsed:.1f}s):")
    print(f"  Win rate: {win_rate:.1%} ({int(win_rate * args.num_games)}/{args.num_games})")
    print(f"  Avg penalty: New={new_avg:.1f}, Best={best_avg:.1f}")
    
    if win_rate >= args.threshold:
        print(f"  ✓ PASSED threshold {args.threshold:.0%} — new model accepted!")
        sys.exit(0)
    else:
        print(f"  ✗ FAILED threshold {args.threshold:.0%} — keeping previous best.")
        sys.exit(1)
