"""
Training Data Generator for the Random Forest Rollout Policy.

This script simulates games between FlatMCo1 (acting as the expert) and a mix
of public baselines, recording the board state, hand, unseen cards, and chosen
action for every turn taken by FlatMCo1 (Player 0).  The collected dataset is
saved as a pickled list of dicts to ``results/rf_data.pkl`` and is consumed by
``train_rf_model.py``.

Usage:
    python scripts/generate_rf_data.py

Configuration (edit constants in ``__main__`` block):
    num_games_total  -- Total games to generate (default: 50,000 across workers).
    time_limit       -- FlatMCo1 wall-clock budget per action in seconds.
    num_workers      -- Number of parallel worker processes.

Output format (each element of the pickled list):
    {
        "board"        : list[list[int]],  # 4 board rows at decision time
        "hand"         : list[int],        # FlatMCo1's full hand
        "unseen_cards" : list[int],        # cards neither on board nor in hand
        "action"       : int               # card value FlatMCo1 chose to play
    }
"""

import os
import sys
import time
import pickle
import copy
import multiprocessing

sys.path.append(os.getcwd())

from src.engine import Engine
from src.game_utils import load_players, _preprocess_player_config


def generate_games(worker_id, num_games, time_limit):
    """
    Worker function: simulate ``num_games`` games and collect FlatMCo1 decisions.

    FlatMCo1 always occupies seat 0.  Opponents are drawn from a fixed mix of
    public baselines and a random player to expose the model to diverse
    opponent behaviours.

    Args:
        worker_id (int): Worker index (unused but required by ``starmap``).
        num_games (int): Number of complete games to simulate.
        time_limit (float): Wall-clock budget per FlatMCo1 action in seconds.

    Returns:
        list[dict]: Collected state-action samples (one per FlatMCo1 turn).
    """
    # Configure the four-player game: FlatMCo1 + three diverse opponents
    config = {
        "players": [
            ["src.players.b12705048.agents.flat_mc_o1", "FlatMCo1"],
            ["src.players.TA.public_baselines1", "Baseline5"],
            ["src.players.TA.public_baselines2", "Baseline10"],
            ["src.players.TA.random_player", "RandomPlayer"]
        ],
        "engine": {
            "n_players": 4,
            "n_rounds": 10,
            "verbose": False,
            "timeout": time_limit,
            "timeout_buffer": 5.0
        }
    }

    config = _preprocess_player_config(config)
    players_classes = load_players(config, verbose=False)

    collected_data = []

    for i in range(num_games):
        # Instantiate fresh player objects for each game
        players_instances = []
        for j, p_cls in enumerate(players_classes):
            players_instances.append(p_cls(player_idx=j))
            if j == 0:  # Set FlatMCo1's time limit explicitly
                players_instances[j].time_limit = time_limit

        engine = Engine(config["engine"], players_instances)

        for _ in range(engine.n_rounds):
            engine.board_history.append([row.copy() for row in engine.board])

            # Build the full history dict that players receive
            history_state = {
                "board": engine.board,
                "scores": engine.scores,
                "round": engine.round,
                "history_matrix": engine.history_matrix,
                "board_history": engine.board_history,
                "score_history": engine.score_history,
            }

            # Snapshot FlatMCo1's state BEFORE any player acts this round
            p0_hand = engine.hands[0].copy()
            p0_board = [row.copy() for row in engine.board]
            
            p0_scores = engine.scores.copy()
            p0_round = engine.round
            p0_history_matrix = [r.copy() for r in engine.history_matrix]
            p0_score_history = [s.copy() for s in engine.score_history]
            p0_board_history = [[r.copy() for r in b] for b in engine.board_history]

            # Compute the set of unseen cards (not on board, not in hand) so
            # that the RF can estimate interception risk during feature extraction
            visible_cards = set()
            for row in p0_board:
                visible_cards.update(row)
            for past_round in engine.history_matrix:
                visible_cards.update(past_round)
            for board_hist in engine.board_history:
                for row in board_hist:
                    visible_cards.update(row)

            unseen_cards = list(set(range(1, 105)) - visible_cards - set(p0_hand))

            # Collect actions from all players, recording only FlatMCo1's decision
            round_actions = [0] * engine.n_players
            round_flags = [False] * engine.n_players
            current_played_cards = []

            for p_idx, player in enumerate(engine.players):
                hand = engine.hands[p_idx]
                played_card = player.action(hand.copy(), copy.deepcopy(history_state))

                if p_idx == 0:
                    # Record the (state, action) pair for the training set
                    collected_data.append({
                        "board": p0_board,
                        "hand": p0_hand,
                        "unseen_cards": unseen_cards,
                        "scores": p0_scores,
                        "round_num": p0_round,
                        "history_matrix": p0_history_matrix,
                        "score_history": p0_score_history,
                        "board_history": p0_board_history,
                        "action": played_card
                    })

                hand.remove(played_card)
                current_played_cards.append((played_card, p_idx))
                round_actions[p_idx] = played_card

            engine.history_matrix.append(round_actions)
            engine.flags_matrix.append(round_flags)

            # Resolve the trick: place all cards in ascending order
            current_played_cards.sort(key=lambda x: x[0])
            for card, p_idx in current_played_cards:
                engine.process_card_placement(card, p_idx)
            engine.score_history.append(list(engine.scores))

            engine.round += 1

    return collected_data


if __name__ == "__main__":
    print("Starting data generation for Random Forest...")
    num_games_total = 1000
    time_limit = 0.5
    num_workers = 8

    games_per_worker = num_games_total // num_workers

    start_time = time.time()

    with multiprocessing.Pool(num_workers) as pool:
        results = pool.starmap(
            generate_games,
            [(i, games_per_worker, time_limit) for i in range(num_workers)]
        )

    # Flatten per-worker lists into a single dataset
    all_data = []
    for r in results:
        all_data.extend(r)

    print(f"Generated {len(all_data)} samples in {time.time() - start_time:.2f} seconds.")

    os.makedirs("results", exist_ok=True)
    with open("results/rf_data.pkl", "wb") as f:
        pickle.dump(all_data, f)
    print("Saved dataset to results/rf_data.pkl")
