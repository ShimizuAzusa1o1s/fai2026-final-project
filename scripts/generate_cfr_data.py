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
from src.players.b12705048.deep_cfr_net import StateEncoder, RegretNet, PolicyNet

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
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.data = []
        
    def add(self, state, target):
        if len(self.data) >= self.capacity:
            self.data.pop(0) 
        self.data.append((state, target))
        
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)
            
    def load(self, path):
        with open(path, 'rb') as f:
            self.data = pickle.load(f)

def get_strategy(regret_net, state_tensor, hand):
    with torch.no_grad():
        regrets = regret_net(state_tensor.unsqueeze(0)).squeeze(0)
    
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

class MCCFR_Traverser:
    def __init__(self, regret_net):
        self.regret_net = regret_net
        self.regret_buffer = ReplayBuffer(capacity=50000)
        self.policy_buffer = ReplayBuffer(capacity=50000)
        self.exploration_prob = 0.1
        
    def traverse(self, hand, opp_hands, board, row_bullheads, player_idx, depth=0, max_depth=5):
        if not hand or depth >= max_depth:
            return sum(BULLHEADS[c] for c in hand)
            
        state_tensor = StateEncoder.encode(hand, board)
        strategy = get_strategy(self.regret_net, state_tensor, hand)
        
        opp_actions = []
        for i in range(4):
            if i != player_idx:
                opp_hand = opp_hands[i]
                
                if random.random() < self.exploration_prob:
                    action = random.choice(opp_hand)
                else:
                    opp_state = StateEncoder.encode(opp_hand, board) 
                    opp_strat = get_strategy(self.regret_net, opp_state, opp_hand)
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
            
        self.regret_buffer.add(state_tensor, regrets)
        self.policy_buffer.add(state_tensor, strategy)
        
        return expected_value

def generate_data(num_games=1000, data_dir="results/deep_cfr"):
    regret_net = RegretNet() 
    weight_path = "src/players/b12705048/weights/regret_net.pt"
    if os.path.exists(weight_path):
        regret_net.load_state_dict(torch.load(weight_path))
        regret_net.eval()
        print(f"Loaded trained Regret Network for data generation.")
    else:
        print("Using untrained Regret Network (first iteration).")
        
    traverser = MCCFR_Traverser(regret_net)
    
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
        
    # Save Buffers to configured data directory
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
