import os
import sys
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.players.b12705048.models.muzero_net.model import MuZeroNet
from src.players.b12705048.agents.muzero_mcts import LatentMCTS
from training.muzero.env_wrapper import MuZeroEnv
from training.muzero.replay_buffer import ReplayBuffer

def play_game(model, env, device):
    obs, legal_actions = env.reset()
    
    trajectory = {
        'obs': [],
        'actions': [],
        'rewards': [],
        'policies': [],
        'values': []
    }
    
    done = False
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        
        mcts = LatentMCTS(model, num_simulations=50)
        root = mcts.run(obs_tensor, legal_actions)
        
        policy = mcts.get_action_policy(root, temperature=1.0)
        value = root.value()
        
        actions = list(policy.keys())
        probs = list(policy.values())
        action = int(np.random.choice(actions, p=probs))
        
        trajectory['obs'].append(obs)
        trajectory['actions'].append(action)
        trajectory['policies'].append(policy)
        trajectory['values'].append(value)
        
        next_obs, reward, done, info = env.step(action)
        
        trajectory['rewards'].append(reward)
        
        obs = next_obs
        legal_actions = info['legal_actions']
        
    return trajectory

def train_muzero():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MuZeroNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    buffer = ReplayBuffer(capacity=5000)
    
    batch_size = 32
    unroll_steps = 3
    training_steps = 100
    games_per_step = 2
    
    print(f"Starting MuZero Training Loop on {device}...")
    env = MuZeroEnv(time_limit=0.01) # extremely fast opponents for self-play
    
    for step in range(training_steps):
        model.eval()
        
        # 1. Generate Data via Self-Play
        for _ in range(games_per_step):
            traj = play_game(model, env, device)
            buffer.push(traj)
            
        if len(buffer) < batch_size:
            continue
            
        # 2. Sample and Train
        model.train()
        obs_b, act_b, rew_b, pol_b, val_b = buffer.sample(batch_size, unroll_steps)
        
        obs_tensor = torch.tensor(np.array(obs_b), dtype=torch.float32, device=device)
        act_tensor = torch.tensor(act_b, dtype=torch.long, device=device) # [B, K]
        rew_tensor = torch.tensor(rew_b, dtype=torch.float32, device=device) # [B, K]
        pol_tensor = torch.tensor(np.array(pol_b), dtype=torch.float32, device=device) # [B, K, 105]
        val_tensor = torch.tensor(val_b, dtype=torch.float32, device=device) # [B, K]
        
        hidden_state, _, _ = model.initial_inference(obs_tensor)
        
        loss = 0
        
        # Unrolled BPTT
        for k in range(unroll_steps):
            actions_onehot = torch.zeros(batch_size, 105, device=device)
            actions_onehot.scatter_(1, act_tensor[:, k].unsqueeze(1), 1.0)
            
            # Recurrent inference using hallucinated hidden state
            hidden_state, reward_pred, policy_logits, value_pred = model.recurrent_inference(hidden_state, actions_onehot)
            
            # Reward Loss
            r_loss = F.mse_loss(reward_pred.squeeze(1), rew_tensor[:, k])
            
            # Value Loss
            v_loss = F.mse_loss(value_pred.squeeze(1), val_tensor[:, k])
            
            # Policy Loss (Soft Cross Entropy)
            log_probs = F.log_softmax(policy_logits, dim=1)
            p_loss = -(pol_tensor[:, k] * log_probs).sum(dim=1).mean()
            
            # Accumulate loss over unroll steps
            loss += (r_loss + v_loss + p_loss) / unroll_steps
            
            # Scale gradient by 0.5 to prevent exploding recurrent gradients (per MuZero paper)
            hidden_state.register_hook(lambda grad: grad * 0.5)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Step {step+1}/{training_steps} | Loss: {loss.item():.4f}")
            
    # Save weights
    os.makedirs('src/players/b12705048/models/muzero_net', exist_ok=True)
    torch.save(model.state_dict(), 'src/players/b12705048/models/muzero_net/weights.pth')
    print("Training complete. Weights saved to src/players/b12705048/models/muzero_net/weights.pth")

if __name__ == "__main__":
    train_muzero()
