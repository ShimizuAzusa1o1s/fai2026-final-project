import sys
import os
import random
sys.path.insert(0, os.path.abspath("."))

from src.players.b12705048.agents.flatmc_ppo import FlatMCPPO

agent = FlatMCPPO(player_idx=0, n_ply=1)

# Mock history
history = {
    'board': [[1, 2], [5], [10, 11, 12], [20]],
    'history_matrix': [],
    'board_history': [],
    'scores': [0, 0, 0, 0],
    'score_history': []
}

hand = [15, 25, 35, 45, 55, 65, 75, 85, 95, 104]

print("Calling action...")
try:
    action = agent.action(hand, history)
    print("Action returned:", action)
except Exception as e:
    print("Exception:", e)
    import traceback
    traceback.print_exc()
