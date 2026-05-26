"""
FlatMC Performance Profiler.

Runs a single ``FlatMC.action()`` call under ``cProfile`` and prints a
ranked breakdown of where time is spent.  Use this script to identify
bottlenecks in the Monte Carlo loop and to measure simulation throughput
(the number of ``_predict_proba`` calls completed within the 0.9 s budget).

Usage:
    python profile_flatmc.py
"""

import time
import os
import sys
import cProfile
import pstats
import io

sys.path.append(os.getcwd())
from src.players.b12705048.flat_mc import FlatMC

if __name__ == '__main__':
    print("Initializing FlatMC...")
    player = FlatMC(player_idx=0)
    print(f"RF Model loaded: {player.rf_model is not None}")

    # Representative mid-game state: 10-card hand, non-trivial board
    hand = [5, 12, 44, 55, 104, 33, 22, 11, 77, 99]
    history = {
        'board': [[1, 2], [3], [4], [50, 51, 52]],
        'history_matrix': [],
        'board_history': [],
    }

    print("Running action() with RF policy profiling...")
    pr = cProfile.Profile()
    pr.enable()

    start_time = time.time()
    action = player.action(hand, history)
    end_time = time.time()

    pr.disable()

    # Print the top-30 hotspots sorted by total (self) time
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats(30)
    print(s.getvalue())

    print(f"Action chosen: {action}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
