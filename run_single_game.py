import json
import copy
import sys
import os
import argparse
import re
from datetime import datetime

# Ensure ./src can be imported
sys.path.append(os.getcwd())

from src.engine import Engine
from src.game_utils import load_players, _preprocess_player_config

# --- Example Configuration ---
# GAME_CONFIG = {
#     "players": [
#         ["src.players.TA.random_player", "RandomPlayer", None],
#         ["src.players.TA.random_player", "RandomPlayer", None],
#         ["src.players.TA.min_player", "MinPlayer", None],
#         ["src.players.TA.max_player", "MaxPlayer", None]
#     ],
#     "engine": {
#         "n_players": 4,
#         "n_rounds": 10,
#         "verbose": True,
#         # "seed": 42
#     },
#     "num_games": 1
# }

def compact_json_dumps(data):
    """
    Dumps json with indent=4 but collapses inner lists of primitives to a single line.
    """
    text = json.dumps(data, indent=4)
    # Match [ ... ] containing only non-bracket characters (leaf lists)
    # and collapse them.
    def collapse(match):
        content = match.group(1)
        # remove newlines and extra spaces
        items = [x.strip() for x in content.split(',')]
        return '[' + ', '.join(items) + ']'
    
    # We use a non-greedy match for the content inside brackets
    # This regex looks for [ followed by stuff that isn't [ or ] or { or } then ]
    return re.sub(r'\[\s*([^\[\]\{\}]+?)\s*\]', collapse, text)

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        sys.exit(1)

def run_simulation():
    parser = argparse.ArgumentParser(description="Run a single game simulation.")
    parser.add_argument("--config", type=str, help="Path to a JSON configuration file.", required=True)
    args = parser.parse_args()

    print("--- Starting Single Game Simulation ---")
    
    # 1. Prepare Configuration
    print(f"Loading configuration from {args.config}...")
    game_config = load_config(args.config)
    # Extract filename without extension for output naming
    config_name = os.path.splitext(os.path.basename(args.config))[0]

    config = copy.deepcopy(game_config)
    config = _preprocess_player_config(config)
    engine_config = config.get("engine", {})
    
    # 2. Load Players
    print("Loading players...")
    players_classes = load_players(config, verbose=True)
    
    players_instances = []
    config_players = config.get("players", [])
    
    for j, p_cls in enumerate(players_classes):
        player_conf = config_players[j]
        player_args = player_conf.get("args")
        
        try:
            if player_args is None:
                player_instance = p_cls(player_idx=j)
            else:
                player_instance = p_cls(player_idx=j, **player_args)
            players_instances.append(player_instance)
        except Exception as e:
            print(f"Error instantiating player {j}: {e}")
            return

    # 3. Initialize Engine
    print("Initializing Engine...")
    try:
        engine = Engine(engine_config, players_instances)
    except Exception as e:
        print(f"Error initializing Engine: {e}")
        return

    # 4. Capture Initial Hands
    initial_hands = copy.deepcopy(engine.hands)
    print("Initial hands captured.")

    # 5. Run Game
    print("Running game...")
    final_scores, history = engine.play_game()
    print("Game finished.")
    print(f"Final Scores: {final_scores}")

    # 6. Construct Output
    output_data = {
        "config": game_config,
        "game_results": {
            "initial_hands": initial_hands,
            "final_scores": final_scores,
            "history": history
        }
    }

    # 7. Save to File
    # Determine output path
    # If config specifies output_file, use it.
    # Otherwise, generate: results/game/<timestamp>_<config_name>.json
    
    output_file = game_config.get("output_file")
    
    if not output_file:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}_{config_name}.json"
        
        # Ensure results directory exists
        results_dir = os.path.join("results", "game")
        os.makedirs(results_dir, exist_ok=True)
        
        output_file = os.path.join(results_dir, filename)

    print(f"Saving results to {output_file}...")
    try:
        # Create parent directory if output_file has one
        parent_dir = os.path.dirname(output_file)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
            
        with open(output_file, 'w') as f:
            f.write(compact_json_dumps(output_data))
        print("Save successful.")
    except Exception as e:
        print(f"Error saving to file: {e}")

if __name__ == "__main__":
    run_simulation()
