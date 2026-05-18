import sys
import os
import copy
import json
import time
import argparse
import re
import multiprocessing as mp
from datetime import datetime

# Ensure ./src can be imported
sys.path.append(os.getcwd())

from src.engine import Engine
from src.players.b12705048.flat_mc import FlatMC

def compact_json_dumps(data):
    """
    Dumps json with indent=4 but collapses inner lists of primitives to a single line.
    """
    text = json.dumps(data, indent=4)
    def collapse(match):
        content = match.group(1)
        items = [x.strip() for x in content.split(',')]
        return '[' + ', '.join(items) + ']'
    return re.sub(r'\[\s*([^\[\]\{\}]+?)\s*\]', collapse, text)

def play_one_game(game_id):
    # Instantiate 4 FlatMC players
    players = [FlatMC(player_idx=i) for i in range(4)]
    
    engine_config = {
        "n_players": 4,
        "n_rounds": 10,
        "verbose": False
    }
    
    try:
        engine = Engine(engine_config, players)
        initial_hands = copy.deepcopy(getattr(engine, 'hands', []))
        
        start_time = time.time()
        final_scores, history = engine.play_game()
        duration = time.time() - start_time
        
        return {
            "game_id": game_id,
            "duration": duration,
            "initial_hands": initial_hands,
            "final_scores": final_scores,
            "history": history
        }
    except Exception as e:
        return {
            "game_id": game_id,
            "duration": 0,
            "error": str(e)
        }

def run_logging(num_games=100, workers=4):
    output_dir = os.path.join("results", "logging")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(output_dir, f"flat_mc_{num_games}games_{timestamp}.json")

    print(f"--- Starting Logging for {num_games} Games ---")
    print(f"Players: 4x FlatMC")
    print(f"Workers: {workers}")
    print(f"This process might take some time (approx. 6 mins per game, parallelized).")
    
    results = []
    start_t = time.time()
    
    with mp.Pool(processes=workers) as pool:
        for i, res in enumerate(pool.imap_unordered(play_one_game, range(num_games))):
            results.append(res)
            if "error" in res:
                print(f"Game {res['game_id']:03d} encountered error: {res['error']}")
            else:
                print(f"Game {res['game_id']:03d} finished in {res['duration']:.2f}s | Scores: {res['final_scores']} | Progress: {i+1}/{num_games}")

    total_time = time.time() - start_t
    print(f"\nAll {num_games} games completed in {total_time:.2f}s.")
    print(f"Average time per game: {total_time/num_games:.2f}s (wall clock)")
    
    print(f"Saving results to {output_file}...")
    try:
        with open(output_file, 'w') as f:
            f.write(compact_json_dumps(results))
        print("Save successful.")
    except Exception as e:
        print(f"Error saving to file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log multiple games of FlatMC.")
    parser.add_argument("--num_games", type=int, default=100, help="Number of games to run.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers.")
    args = parser.parse_args()
    
    run_logging(args.num_games, args.workers)
