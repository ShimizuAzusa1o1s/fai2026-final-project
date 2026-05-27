import sys
import time

def profile_agent(agent_class, name, time_limit=0.9):
    try:
        agent = agent_class(player_idx=0)
    except Exception as e:
        print(f"Error instantiating {name}: {e}")
        return 0
        
    if hasattr(agent, 'time_limit'):
        agent.time_limit = time_limit

    board = [[10, 13], [20, 22], [30], [40, 42, 44]]
    hand = [12, 55, 60, 104, 15]
    history_matrix = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 11, 14, 16],
        [17, 18, 19, 21],
        [23, 24, 25, 26]
    ]
    score_history = [
        [0, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 2, 3, 0],
        [5, 2, 3, 0],
        [5, 2, 3, 1]
    ]
    board_history = [
        [[1], [2], [3], [4]] for _ in range(5)
    ]
    history = {
        'board': board,
        'history_matrix': history_matrix,
        'board_history': board_history,
        'scores': [5, 2, 3, 1],
        'round': 5,
        'score_history': score_history
    }
    total_sims = 0

    def trace(frame, event, arg):
        nonlocal total_sims
        if event == 'return' and frame.f_code.co_name == 'action':
            if 'stats_visits' in frame.f_locals:
                total_sims = sum(frame.f_locals['stats_visits'].values())
        return trace

    sys.setprofile(trace)
    start_t = time.perf_counter()
    try:
        agent.action(hand, history)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error running {name}.action(): {e}")
    finally:
        sys.setprofile(None)
    elapsed = time.perf_counter() - start_t
    
    print(f"{name:<15} : {total_sims:8,} simulations in {elapsed:.3f}s")
    return total_sims

if __name__ == '__main__':
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

    agents = []
    
    try:
        from src.players.b12705048.agents.flat_mc import FlatMC
        agents.append((FlatMC, 'FlatMC'))
    except ImportError:
        pass
        
    try:
        from src.players.b12705048.agents.flat_mc_o1 import FlatMCo1
        agents.append((FlatMCo1, 'FlatMCo1'))
    except ImportError:
        pass
        
    try:
        from src.players.b12705048.agents.rf_flat_mc import FlatMC as RFFlatMC
        agents.append((RFFlatMC, 'RFFlatMC'))
    except ImportError:
        pass
        
    try:
        from src.players.b12705048.agents.ucb_rf_mc import UCB_RF_MC
        agents.append((UCB_RF_MC, 'UCB_RF_MC'))
    except ImportError:
        pass
        
    try:
        from src.players.b12705048.agents.segment_mc import SegmentMC
        agents.append((SegmentMC, 'SegmentMC'))
    except ImportError:
        pass

    try:
        from src.players.b12705048.agents.segment_mc_o1 import SegmentMCo1
        agents.append((SegmentMCo1, 'SegmentMCo1'))
    except ImportError:
        pass

    try:
        from src.players.b12705048.agents.rf_flat_mc_o1 import FlatMC as RFFlatMCo1
        agents.append((RFFlatMCo1, 'RFFlatMCo1'))
    except ImportError:
        pass

    try:
        from src.players.b12705048.agents.ucb_rf_mc_o1 import UCB_RF_MC as UCB_RF_MCo1
        agents.append((UCB_RF_MCo1, 'UCB_RF_MCo1'))
    except ImportError:
        pass

    print(f"Profiling MCTS variants (time limit: 0.9s)...\n")
    for agent_cls, name in agents:
        profile_agent(agent_cls, name, 0.9)
