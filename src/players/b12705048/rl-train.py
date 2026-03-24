import torch
import torch.nn as nn
import torch.optim as optim
import random
import json
import numpy as np
import sys
import os
from collections import deque

# Ensure we can import src modules
sys.path.append(os.getcwd())
from src.engine import Engine
from src.players.TA.random_player import RandomPlayer

# --- 0. Environment Wrapper ---
class RLPlayer:
    def __init__(self, player_idx):
        self.player_idx = player_idx
        self.action_to_play = None

    def action(self, hand, history):
        return self.action_to_play

class NimmtEnv:
    def __init__(self):
        self.config = {
            "n_players": 4,
            "n_rounds": 10,
            "verbose": False
        }
        self.players = [
            RLPlayer(0),
            RandomPlayer(1),
            RandomPlayer(2),
            RandomPlayer(3)
        ]
        self.engine = Engine(self.config, self.players)
        self.rl_player = self.players[0]
        
    def _get_state(self):
        state = np.zeros(256, dtype=np.float32)
        hand = self.engine.hands[0]
        for card in hand:
            if 1 <= card <= 104:
                state[card - 1] = 1.0 # 0-103: Hand
        for row in self.engine.board:
            for card in row:
                if 1 <= card <= 104:
                    state[104 + card - 1] = 1.0 # 104-207: Board
        # Leftover 24 variables can be used for extra features like scores, etc.
        return state
        
    def get_current_hand(self):
        return self.engine.hands[0]

    def reset(self):
        self.engine.reset()
        return self._get_state()

    def step(self, action):
        self.rl_player.action_to_play = action
        old_score = self.engine.scores[0]
        self.engine.play_round()
        self.engine.round += 1
        new_score = self.engine.scores[0]
        reward = -(new_score - old_score)
        done = (self.engine.round >= self.engine.n_rounds)
        return self._get_state(), reward, done

def get_best_valid_action(q_values, current_hand):
    # Action is an index from 0 to 103, corresponding to card 1 to 104.
    valid_actions = [card - 1 for card in current_hand]
    best_action = max(valid_actions, key=lambda a: q_values[a].item())
    return best_action + 1 # return actual card

# --- 1. The Neural Network (Must perfectly match your lightweight inferencer) ---
class PyTorchDQN(nn.Module):
    def __init__(self, input_size=256, hidden_size=128, output_size=104):
        super(PyTorchDQN, self).__init__()
        # We use a simple 2-layer network to keep the math fast for the tournament
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# --- 2. The Replay Buffer (Memory) ---
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (torch.FloatTensor(np.array(state)), 
                torch.LongTensor(action), 
                torch.FloatTensor(reward), 
                torch.FloatTensor(np.array(next_state)), 
                torch.FloatTensor(done))
    
    def __len__(self):
        return len(self.buffer)

# --- 3. The Training Loop Skeleton ---
def train_agent():
    # Hyperparameters
    BATCH_SIZE = 128
    GAMMA = 0.90            # Discount factor for future rewards
    EPSILON_START = 1.0     # 100% exploration initially
    EPSILON_END = 0.05      # 5% exploration later
    EPSILON_DECAY = 30000   # How fast to decay exploration
    TARGET_UPDATE = 50      # How often to update target network
    EPISODES = 100000       # Number of games to play
    
    # Initialize Networks
    policy_net = PyTorchDQN()
    target_net = PyTorchDQN()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=0.0005)
    memory = ReplayBuffer()
    
    steps_done = 0

    print("Starting Training...")
    
    for episode in range(EPISODES):
        # Initialize your 6 Nimmt! game environment here
        env = NimmtEnv()
        state = env.reset()
        done = False
        
        while not done:
            current_hand = env.get_current_hand()
            
            # 1. Epsilon-Greedy Action Selection
            eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1. * steps_done / EPSILON_DECAY)
            steps_done += 1
            
            if random.random() > eps_threshold:
                # EXPLOIT: Use the network to pick the best valid card
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = policy_net(state_tensor).squeeze()
                    # Filter Q-values to only consider valid cards in hand
                    action = get_best_valid_action(q_values, current_hand)
            else:
                # EXPLORE: Pick a random valid card
                action = random.choice(current_hand)
                
            # 2. Step the Environment
            next_state, reward, done = env.step(action) 
            # Note: Reward should be negative bullheads!
            
            # Action for network is 0-103
            action_idx = action - 1
            
            # 3. Store in Memory
            memory.push(state, action_idx, reward, next_state, done)
            state = next_state
            
            # 4. Optimize the Model
            if len(memory) > BATCH_SIZE:
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
                
                # Get current Q values
                current_q = policy_net(states).gather(1, actions.unsqueeze(1))
                
                # Get max next Q values from target network
                next_q = target_net(next_states).max(1)[0].detach()
                expected_q = rewards + (GAMMA * next_q * (1 - dones))
                
                # Compute Loss (Mean Squared Error)
                loss = nn.MSELoss()(current_q.squeeze(), expected_q)
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        # Update target network periodically
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        if episode % 100 == 0:
            print(f"Episode {episode}/{EPISODES} completed.")

    # --- 4. Export the Brain to JSON ---
    print("\nTraining complete! Exporting weights...")
    state_dict = policy_net.state_dict()
    
    export_data = {
        "W1": state_dict['fc1.weight'].tolist(),
        "b1": state_dict['fc1.bias'].tolist(),
        "W2": state_dict['fc2.weight'].tolist(),
        "b2": state_dict['fc2.bias'].tolist()
    }
    
    with open('dqn_weights.json', 'w') as f:
        json.dump(export_data, f)
        
    print("Saved 'dqn_weights.json'. You can now load this in your pure Python agent!")

if __name__ == "__main__":
    train_agent()