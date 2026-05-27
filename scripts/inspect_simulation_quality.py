import sys
import os
import json
import time
import numpy as np
from joblib import Parallel, delayed

sys.path.append(os.getcwd())

from src.engine import Engine
from src.players.b12705048.agents.flat_mc import FlatMC
from src.players.b12705048.agents.flat_mc_o1 import FlatMCo1
from src.players.TA.public_baselines2 import Baseline8, Baseline9, Baseline10

# We will run N games for each configuration.
# To make it fair, we seed each game 'g' with seed = g.
# This ensures all configurations face identical starting conditions (duplicate comparison).
NUM_GAMES = 80
NUM_WORKERS = 10

def run_single_game(game_idx, agent_cls, time_limit, agent_name):
    # Opponents are Baseline8, Baseline9, Baseline10
    # The MC agent is seated at index 0 for consistency, which is fair under duplicate seeds
    mc_agent = agent_cls(player_idx=0, time_limit=time_limit)
    
    opponents = [
        Baseline8(player_idx=1),
        Baseline9(player_idx=2),
        Baseline10(player_idx=3)
    ]
    
    players = [mc_agent] + opponents
    
    # Track simulations run per action
    sims_per_turn = []
    orig_action = mc_agent.action
    def instrumented_action(hand, history):
        res = orig_action(hand, history)
        sims_per_turn.append(mc_agent.last_simulations)
        return res
    mc_agent.action = instrumented_action
    
    cfg = {
        "n_cards": 104,
        "n_players": 4,
        "n_rounds": 10,
        "verbose": False,
        "seed": 2000 + game_idx,  # Deterministic seed per game
        "timeout": None,
    }
    
    try:
        engine = Engine(cfg, players)
        scores, full_history = engine.play_game()
        
        # Calculate fractional ranks
        ranks = [0.0] * 4
        for i, score in enumerate(scores):
            better_count = sum(1 for s in scores if s < score)
            same_count = sum(1 for s in scores if s == score)
            ranks[i] = (2 * better_count + same_count + 1) / 2.0
            
        is_dq = 0 in full_history.get("disqualified_players", []) or 0 in full_history.get("exception_counts", {})
        
        return {
            "game_idx": game_idx,
            "score": scores[0],
            "rank": ranks[0],
            "avg_sims_per_turn": sum(sims_per_turn) / len(sims_per_turn) if sims_per_turn else 0,
            "dq": is_dq
        }
    except Exception as e:
        return {
            "game_idx": game_idx,
            "score": 1000,
            "rank": 4.0,
            "avg_sims_per_turn": 0,
            "dq": True,
            "error": str(e)
        }

def run_experiment_for_config(agent_cls, time_limit, agent_name):
    print(f"Running {agent_name} with time_limit={time_limit}s...")
    
    results = Parallel(n_jobs=NUM_WORKERS)(
        delayed(run_single_game)(g, agent_cls, time_limit, agent_name)
        for g in range(NUM_GAMES)
    )
    
    valid_results = [r for r in results if not r.get("dq", False)]
    dq_count = sum(1 for r in results if r.get("dq", False))
    
    if not valid_results:
        return {
            "agent_name": agent_name,
            "time_limit": time_limit,
            "avg_score": float('nan'),
            "avg_rank": float('nan'),
            "avg_simulations": 0,
            "dq_count": dq_count
        }
        
    avg_score = np.mean([r["score"] for r in valid_results])
    avg_rank = np.mean([r["rank"] for r in valid_results])
    avg_sims = np.mean([r["avg_sims_per_turn"] for r in valid_results])
    
    return {
        "agent_name": agent_name,
        "time_limit": time_limit,
        "avg_score": float(avg_score),
        "avg_rank": float(avg_rank),
        "avg_simulations": float(avg_sims),
        "dq_count": dq_count
    }

if __name__ == "__main__":
    configs = [
        (FlatMC, 0.05, "FlatMC"),
        (FlatMC, 0.10, "FlatMC"),
        (FlatMC, 0.20, "FlatMC"),
        (FlatMC, 0.50, "FlatMC"),
        (FlatMC, 0.95, "FlatMC"),
        
        (FlatMCo1, 0.05, "FlatMCo1"),
        (FlatMCo1, 0.10, "FlatMCo1"),
        (FlatMCo1, 0.20, "FlatMCo1"),
        (FlatMCo1, 0.50, "FlatMCo1"),
        (FlatMCo1, 0.90, "FlatMCo1"),
    ]
    
    summary = []
    for agent_cls, time_limit, name in configs:
        res = run_experiment_for_config(agent_cls, time_limit, name)
        summary.append(res)
        print(f"Result: {res['agent_name']} (t={res['time_limit']}s): "
              f"Score={res['avg_score']:.2f}, Rank={res['avg_rank']:.2f}, "
              f"Sims={res['avg_simulations']:.0f}, DQ={res['dq_count']}")
              
    os.makedirs("results", exist_ok=True)
    with open("results/simulation_quality_results.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    print("\n\nExperiment Summary Table:")
    print("| Agent | Time Limit (s) | Avg Simulations / Decision | Avg Rank (lower is better) | Avg Penalty (lower is better) | DQ |")
    print("|---|---|---|---|---|---|")
    for r in summary:
        print(f"| {r['agent_name']} | {r['time_limit']:.2f} | {r['avg_simulations']:,.0f} | {r['avg_rank']:.3f} | {r['avg_score']:.2f} | {r['dq_count']} |")
