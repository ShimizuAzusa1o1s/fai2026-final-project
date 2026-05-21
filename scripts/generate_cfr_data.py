import sys
import os
import torch
import random
import copy
import pickle
import numpy as np
import argparse
from tqdm import tqdm

# Ensure ./src can be imported
sys.path.append(os.getcwd())
from src.players.b12705048.deep_cfr_net import StateEncoder, RegretNet, PolicyNet, INPUT_DIM

# Pre-compute bullhead values
BULLHEADS = [0] * 105
for c in range(1, 105):
    if c == 55: BULLHEADS[c] = 7
    elif c % 11 == 0: BULLHEADS[c] = 5
    elif c % 10 == 0: BULLHEADS[c] = 3
    elif c % 5 == 0: BULLHEADS[c] = 2
    else: BULLHEADS[c] = 1
BULLHEADS = tuple(BULLHEADS)

class ReplayBuffer:
    """O(1) ring buffer for storing (state, target, mask) triples."""
    def __init__(self, capacity=200000):
        self.capacity = capacity
        self.data = []
        self.pos = 0
        
    def add(self, state, target, mask):
        entry = (state, target, mask)
        if len(self.data) < self.capacity:
            self.data.append(entry)
        else:
            self.data[self.pos] = entry
        self.pos = (self.pos + 1) % self.capacity
        
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)
            
    def load(self, path):
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
        self.pos = len(self.data) % self.capacity

def get_strategy(regret_net, state_tensor, hand, device):
    with torch.no_grad():
        # Move tensor to correct device for inference, then move output back to CPU
        regrets = regret_net(state_tensor.unsqueeze(0).to(device)).squeeze(0).cpu()
    
    strategy = torch.zeros(104)
    positive_regret_sum = 0.0
    
    for c in hand:
        r = max(regrets[c-1].item(), 0.0)
        strategy[c-1] = r
        positive_regret_sum += r
        
    if positive_regret_sum > 0:
        strategy /= positive_regret_sum
    else:
        prob = 1.0 / len(hand)
        for c in hand:
            strategy[c-1] = prob
            
    return strategy

def resolve_trick(board, row_bullheads, pending_actions, penalties):
    pending_actions.sort(key=lambda x: x[0])
    
    for card, p_idx in pending_actions:
        target_row = -1
        max_val = -1
        
        for r in range(4):
            val = board[r][-1]
            if val < card and val > max_val:
                max_val = val
                target_row = r
                
        if target_row != -1:
            if len(board[target_row]) == 5:
                penalties[p_idx] += row_bullheads[target_row]
                board[target_row] = [card]
                row_bullheads[target_row] = BULLHEADS[card]
            else:
                board[target_row].append(card)
                row_bullheads[target_row] += BULLHEADS[card]
        else:
            min_score = 999999
            target_row = -1
            for r in range(4):
                score = row_bullheads[r] * 1000 + len(board[r]) * 10 + r
                if score < min_score:
                    min_score = score
                    target_row = r
            penalties[p_idx] += row_bullheads[target_row]
            board[target_row] = [card]
            row_bullheads[target_row] = BULLHEADS[card]


def terminal_estimate(hand, board, row_bullheads):
    """
    Estimate future penalty via greedy single-player simulation.
    Much better than the naive sum(BULLHEADS[c] for c in hand) which
    assumes every remaining card causes a penalty.
    """
    if not hand:
        return 0.0
        
    penalty = 0.0
    sim_tails = [row[-1] for row in board]
    sim_lengths = [len(row) for row in board]
    sim_bh = row_bullheads[:]
    
    for c in sorted(hand):
        # Find target row (same logic as game engine)
        target = -1
        min_gap = 105
        for r in range(4):
            gap = c - sim_tails[r]
            if 0 < gap < min_gap:
                min_gap = gap
                target = r
        
        if target == -1:
            # Low card: forced to take cheapest row
            cheapest = min(range(4), key=lambda r: (sim_bh[r], sim_lengths[r], r))
            penalty += sim_bh[cheapest]
            sim_tails[cheapest] = c
            sim_lengths[cheapest] = 1
            sim_bh[cheapest] = BULLHEADS[c]
        elif sim_lengths[target] >= 5:
            # 6th card rule: take the row
            penalty += sim_bh[target]
            sim_tails[target] = c
            sim_lengths[target] = 1
            sim_bh[target] = BULLHEADS[c]
        else:
            sim_tails[target] = c
            sim_lengths[target] += 1
            sim_bh[target] += BULLHEADS[c]
    
    return penalty


class MCCFR_Traverser:
    def __init__(self, regret_net, device):
        self.regret_net = regret_net
        self.device = device
        self.regret_buffer = ReplayBuffer(capacity=200000)
        self.policy_buffer = ReplayBuffer(capacity=200000)
        self.exploration_prob = 0.1
        
    def traverse(self, hand, opp_hands, board, row_bullheads, player_idx, depth=0, max_depth=5):
        if not hand or depth >= max_depth:
            return terminal_estimate(hand, board, row_bullheads)
            
        state_tensor = StateEncoder.encode(hand, board, round_num=depth)
        legal_mask = StateEncoder.get_legal_mask(hand)
        strategy = get_strategy(self.regret_net, state_tensor, hand, self.device)
        
        opp_actions = []
        for i in range(4):
            if i != player_idx:
                opp_hand = opp_hands[i]
                
                if random.random() < self.exploration_prob:
                    action = random.choice(opp_hand)
                else:
                    opp_state = StateEncoder.encode(opp_hand, board, round_num=depth) 
                    opp_strat = get_strategy(self.regret_net, opp_state, opp_hand, self.device)
                    probs = [opp_strat[c-1].item() for c in opp_hand]
                    action = random.choices(opp_hand, weights=probs, k=1)[0]
                    
                opp_actions.append((action, i))
                
        action_values = {}
        expected_value = 0.0
        
        for action in hand:
            sim_board = [r[:] for r in board]
            sim_row_bullheads = row_bullheads[:]
            penalties = [0.0, 0.0, 0.0, 0.0]
            
            pending = opp_actions[:]
            pending.append((action, player_idx))
            
            resolve_trick(sim_board, sim_row_bullheads, pending, penalties)
            immediate_penalty = penalties[player_idx]
            
            next_hand = [c for c in hand if c != action]
            next_opp_hands = {}
            for opp_card, opp_idx in opp_actions:
                next_opp_hands[opp_idx] = [c for c in opp_hands[opp_idx] if c != opp_card]
                
            future_penalty = self.traverse(
                next_hand, next_opp_hands, sim_board, sim_row_bullheads, player_idx, depth+1, max_depth
            )
            
            total_penalty = immediate_penalty + future_penalty
            action_values[action] = total_penalty
            expected_value += strategy[action - 1].item() * total_penalty
            
        regrets = torch.zeros(104)
        for action in hand:
            regrets[action - 1] = expected_value - action_values[action]
            
        # Store (state, target, mask) triples — always on CPU
        self.regret_buffer.add(state_tensor, regrets, legal_mask)
        self.policy_buffer.add(state_tensor, strategy, legal_mask)
        
        return expected_value

def generate_data(num_games=1000, data_dir="results/deep_cfr"):
    # Dynamically select GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Data Generator using device: {device}")
    
    regret_net = RegretNet().to(device)
    weight_path = "src/players/b12705048/weights/regret_net.pt"
    
    if os.path.exists(weight_path):
        try:
            regret_net.load_state_dict(torch.load(weight_path, map_location=device))
            regret_net.eval()
            print(f"Loaded trained Regret Network for data generation.")
        except RuntimeError as e:
            print(f"[Warning] Incompatible weights at {weight_path}, starting fresh: {e}")
    else:
        print("Using untrained Regret Network (first iteration).")
        
    traverser = MCCFR_Traverser(regret_net, device)
    
    print(f"Generating Deep CFR data for {num_games} games using External Sampling...")
    
    for _ in tqdm(range(num_games)):
        deck = list(range(1, 105))
        random.shuffle(deck)
        
        board = [[deck.pop()] for _ in range(4)]
        row_bullheads = [BULLHEADS[r[0]] for r in board]
        hands = {i: [deck.pop() for _ in range(10)] for i in range(4)}
        
        player_idx = random.randint(0, 3)
        hand = hands[player_idx]
        opp_hands = {i: hands[i] for i in range(4) if i != player_idx}
        
        traverser.traverse(hand, opp_hands, board, row_bullheads, player_idx, depth=0, max_depth=5)
        
    os.makedirs(data_dir, exist_ok=True)
    regret_path = os.path.join(data_dir, "regret_buffer.pkl")
    policy_path = os.path.join(data_dir, "policy_buffer.pkl")
    
    traverser.regret_buffer.save(regret_path)
    traverser.policy_buffer.save(policy_path)
    
    print(f"\nSaved {len(traverser.regret_buffer.data)} state-regret pairs to {regret_path}")
    print(f"Saved {len(traverser.policy_buffer.data)} state-strategy pairs to {policy_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_games", type=int, default=100, help="Number of games to simulate")
    parser.add_argument("--data_dir", type=str, default="results/deep_cfr", help="Directory to save memory-consuming replay buffers")
    args = parser.parse_args()
    
    generate_data(args.num_games, args.data_dir)
