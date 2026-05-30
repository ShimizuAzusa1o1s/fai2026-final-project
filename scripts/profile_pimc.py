import time
from src.players.b12705048.agents.pimc_ppo import PimcPUCTPlayer

def run_profile():
    agent = PimcPUCTPlayer(player_idx=0, model_path="src/players/b12705048/agents/stage3_model_final")
    agent.time_limit = 0.1
    
    hand = [10, 25, 33, 42, 55] # late game
    history = {
        'board': [[1, 2], [3], [4], [5]],
        'history_matrix': [],
        'board_history': [],
        'scores': [0, 0, 0, 0],
        'score_history': []
    }
    
    # Warmup
    print("Warming up PIMC...")
    agent.action(hand, history)
    
    print("\nProfiling with 0.1s limit...")
    start_time = time.perf_counter()
    action = agent.action(hand, history)
    end_time = time.perf_counter()
    
    print(f"Chosen action: {action}")
    print(f"Wall-clock time taken: {end_time - start_time:.4f}s")
    
if __name__ == "__main__":
    run_profile()
