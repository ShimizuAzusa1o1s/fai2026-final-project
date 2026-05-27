import time
import sys
import numpy as np
import random

sys.path.append("/home/azusa_in_linux/workspace/2026fai/final-project")

# We will time exactly one batch of 5000 games for FlatMCo1
# And time exactly 5000 games for FlatMC
# Then compute games per second for both.

def profile_flat_mc_o1():
    from src.players.b12705048.agents.flat_mc_o1 import FlatMCo1
    agent = FlatMCo1(player_idx=0)
    agent.batch_size = 5000
    
    board = [[10], [20], [30], [40]]
    hand = [12, 55, 60, 104, 15, 88, 99, 13, 2, 44]
    
    # We will just run the agent.action but with a small time_limit and see how many stats_visits it gets.
    # To do this cleanly, we can use a trace.
    # Or, we can just patch `time.perf_counter` to only return "done" after X batches, 
    # but we just want to know how long 1 batch takes.
    
    t0 = time.time()
    agent.time_limit = 1.0 # 1 second
    best = agent.action(hand, [{'board': board}])
    t1 = time.time()
    
    print(f"FlatMCo1 finished in {t1-t0:.4f}s")
    return agent

def benchmark_o1_batch():
    from src.players.b12705048.agents.flat_mc_o1 import FlatMCo1
    agent = FlatMCo1(player_idx=0)
    
    # Setup state
    board = [[10], [20], [30], [40]]
    hand = [12, 55, 60, 104, 15, 88, 99, 13, 2, 44]
    candidates = hand
    n_turns = len(hand)
    
    unseen_cards = list(set(range(1,105)) - set([10,20,30,40]) - set(hand))
    unseen_mask_base = np.zeros(105, dtype=bool)
    unseen_mask_base[unseen_cards] = True
    
    orig_tails = np.array([row[-1] for row in board], dtype=np.int32)
    orig_lengths = np.array([len(row) for row in board], dtype=np.int32)
    orig_rbulls = np.array([sum(agent.bullhead_lookup[c] for c in row) for row in board], dtype=np.int32)
    
    actual_batch_size = 5000
    sims_per_cand = 5000 // len(candidates)
    actual_batch_size = sims_per_cand * len(candidates)
    
    t0 = time.time()
    
    # ONE BATCH EXECUTION
    tails = np.tile(orig_tails, (actual_batch_size, 1))
    lengths = np.tile(orig_lengths, (actual_batch_size, 1))
    rbulls = np.tile(orig_rbulls, (actual_batch_size, 1))
    penalties = np.zeros((actual_batch_size, 4), dtype=np.int32)
    
    unseen_mask = np.tile(unseen_mask_base, (actual_batch_size, 1))
    rand_weights = np.random.rand(actual_batch_size, 105)
    rand_weights[~unseen_mask] = -1.0
    perm = np.argsort(-rand_weights, axis=1)
    
    opp_indices = [1, 2, 3]
    hands_array = np.zeros((actual_batch_size, 4, n_turns), dtype=np.int32)
    hands_array[:, opp_indices[0], :] = perm[:, 0:n_turns]
    hands_array[:, opp_indices[1], :] = perm[:, n_turns:2*n_turns]
    hands_array[:, opp_indices[2], :] = perm[:, 2*n_turns:3*n_turns]
    
    c_idx = 0
    for c in candidates:
        start_b = c_idx * sims_per_cand
        end_b = start_b + sims_per_cand
        my_rest = [x for x in hand if x != c]
        rest_arr = np.array(my_rest, dtype=np.int32)
        my_hands_chunk = np.tile(rest_arr, (sims_per_cand, 1))
        
        if len(my_rest) > 0:
            rand_my = np.random.rand(sims_per_cand, len(my_rest))
            my_perm = np.argsort(rand_my, axis=1)
            my_hands_chunk = np.take_along_axis(my_hands_chunk, my_perm, axis=1)
            
        hands_array[start_b:end_b, 0, 0] = c
        if len(my_rest) > 0:
            hands_array[start_b:end_b, 0, 1:] = my_hands_chunk
        c_idx += 1
        
    for t in range(n_turns):
        played_cards = hands_array[:, :, t]
        sort_idx = np.argsort(played_cards, axis=1)
        sorted_cards = np.take_along_axis(played_cards, sort_idx, axis=1)
        sorted_players = sort_idx
        
        for i in range(4):
            current_cards = sorted_cards[:, i]
            current_players = sorted_players[:, i]
            valid = np.where(current_cards[:, None] > tails, tails, -1)
            target_rows = np.argmax(valid, axis=1)
            invalid_mask = np.max(valid, axis=1) == -1
            
            scores = rbulls * 1000 + lengths * 10 + np.arange(4)
            min_rows = np.argmin(scores, axis=1)
            target_rows = np.where(invalid_mask, min_rows, target_rows)
            
            b_idx = np.arange(actual_batch_size)
            target_lengths = lengths[b_idx, target_rows]
            target_bullheads = rbulls[b_idx, target_rows]
            
            penalty_condition = invalid_mask | (target_lengths == 5)
            normal_cond = ~penalty_condition
            card_bulls = agent.bullhead_lookup[current_cards]
            
            if np.any(penalty_condition):
                pc = penalty_condition
                b_pc = b_idx[pc]
                p_players = current_players[pc]
                penalties[b_pc, p_players] += target_bullheads[pc]
                lengths[b_pc, target_rows[pc]] = 1
                tails[b_pc, target_rows[pc]] = current_cards[pc]
                rbulls[b_pc, target_rows[pc]] = card_bulls[pc]
                
            if np.any(normal_cond):
                nc = normal_cond
                b_nc = b_idx[nc]
                lengths[b_nc, target_rows[nc]] += 1
                tails[b_nc, target_rows[nc]] = current_cards[nc]
                rbulls[b_nc, target_rows[nc]] += card_bulls[nc]
                
    t1 = time.time()
    
    elapsed = t1 - t0
    games_per_sec = actual_batch_size / elapsed
    print(f"FlatMCo1: {actual_batch_size} games simulated in {elapsed:.4f}s")
    print(f"FlatMCo1 Speed: {games_per_sec:,.0f} games/second")
    
def benchmark_standard():
    import sys
    try:
        from src.players.b12705048.agents.flat_mc import FlatMC
    except ImportError:
        FlatMC = None
        print("FlatMC not found, skipping standard benchmark.")
        return
        
    agent = FlatMC(player_idx=0)
    
    board = [[10], [20], [30], [40]]
    hand = [12, 55, 60, 104, 15, 88, 99, 13, 2, 44]
    unseen_cards = list(set(range(1,105)) - set([10,20,30,40]) - set(hand))
    n_turns = len(hand)
    opp_indices = [1, 2, 3]
    orig_row_bullheads = [sum(agent.bullhead_lookup[c] for c in row) for row in board]
    
    n_sims = 500 # run fewer for python implementation
    
    t0 = time.time()
    for _ in range(n_sims):
        random.shuffle(unseen_cards)
        candidate = hand[0]
        
        sim_board = [row[:] for row in board]
        sim_row_bullheads = orig_row_bullheads[:]
        sim_hands = [None] * 4
        sim_hands[opp_indices[0]] = unseen_cards[0:n_turns]
        sim_hands[opp_indices[1]] = unseen_cards[n_turns:2*n_turns]
        sim_hands[opp_indices[2]] = unseen_cards[2*n_turns:3*n_turns]
        
        my_sim_hand = [c for c in hand if c != candidate]
        random.shuffle(my_sim_hand)
        sim_hands[0] = my_sim_hand
        
        penalties = [0.0, 0.0, 0.0, 0.0]
        pending_actions = [(candidate, 0)]
        for opp_idx in opp_indices:
            pending_actions.append((sim_hands[opp_idx].pop(), opp_idx))
            
        agent._resolve_trick(sim_board, sim_row_bullheads, pending_actions, penalties)
        
        for _ in range(n_turns - 1):
            pending_actions = [
                (sim_hands[0].pop(), 0),
                (sim_hands[1].pop(), 1),
                (sim_hands[2].pop(), 2),
                (sim_hands[3].pop(), 3)
            ]
            agent._resolve_trick(sim_board, sim_row_bullheads, pending_actions, penalties)
            
    t1 = time.time()
    elapsed = t1 - t0
    games_per_sec = n_sims / elapsed
    print(f"FlatMC: {n_sims} games simulated in {elapsed:.4f}s")
    print(f"FlatMC Speed: {games_per_sec:,.0f} games/second")

if __name__ == "__main__":
    benchmark_o1_batch()
    benchmark_standard()
