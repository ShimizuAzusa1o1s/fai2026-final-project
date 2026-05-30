import sys
import time
import os

# Ensure the root is in sys.path
sys.path.append(os.getcwd())

from src.players.b12705048.agents.flatmc import FlatMC
from src.players.b12705048.agents.flatmc_ppo import FlatMCPPO

def run_profile():
    # Setup a dummy history and hand
    hand = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    board = [[1], [2], [3], [4]]
    history = {
        'board': board,
        'history_matrix': [],
        'board_history': [],
        'scores': [0, 0, 0, 0],
        'score_history': []
    }
    
    print("Initializing FlatMC...")
    agent_mc = FlatMC(player_idx=0)
    agent_mc.time_limit = 1.0 # 1 second budget
    
    print("Running FlatMC for ~1.0 seconds...")
    start_mc = time.time()
    card_mc = agent_mc.action(hand, history)
    end_mc = time.time()
    print(f"FlatMC returned {card_mc} in {end_mc - start_mc:.3f} seconds.\n")

    print("Initializing FlatMCPPO...")
    agent_ppo = FlatMCPPO(player_idx=0, n_ply=1)
    agent_ppo.time_limit = 1.0 # 1 second budget
    
    print("Running FlatMCPPO for ~1.0 seconds...")
    start_ppo = time.time()
    card_ppo = agent_ppo.action(hand, history)
    end_ppo = time.time()
    print(f"FlatMCPPO returned {card_ppo} in {end_ppo - start_ppo:.3f} seconds.\n")

if __name__ == "__main__":
    run_profile()
