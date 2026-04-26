import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

import sys
sys.path.append(os.getcwd())

from src.players.b12705048.model import TinyAlphaZeroNet
from src.players.b12705048.state_encoding import get_state_dim, N_CARDS

class SelfPlayDataset(Dataset):
    def __init__(self, data):
        self.states = [d[0] for d in data]
        self.masks = [d[1] for d in data]
        self.probs = [d[2] for d in data]
        self.values = [d[3] for d in data]
        
    def __len__(self):
        return len(self.states)
        
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.states[idx]),
            torch.FloatTensor(self.masks[idx]),
            torch.FloatTensor(self.probs[idx]),
            torch.FloatTensor([self.values[idx]])
        )

def train_model(model_path=None, data_path="data/self_play_data.pt", save_path="models/latest.pt", epochs=5, batch_size=256, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    state_dim = get_state_dim()
    n_actions = N_CARDS
    
    model = TinyAlphaZeroNet(state_dim, n_actions).to(device)
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded existing model from {model_path}")
        
    # Load dataset
    print(f"Loading data from {data_path}...")
    data = torch.load(data_path, weights_only=False)
    
    # Process the data
    # The format exported in self_play.py currently yields tuples of length 4.
    processed_data = []
    
    # The old logic returned the data correctly formatted.
    # self_play_episode yielded: (state_vec, mask, target_probs, val)
    # the last element p_idx was replaced by val later in the self_play_episode function.
    for i in range(len(data)):
        state_vec, mask, target_probs, val_tuple = data[i]
        
        # Check if the last param is a tuple or a float (sometimes due to how I replaced it, it may be (state_vec, mask, prob, val))
        if isinstance(val_tuple, tuple):
             val = val_tuple[-1]
        else:
             val = val_tuple
             
        processed_data.append((state_vec, mask, target_probs, val))
        
    dataset = SelfPlayDataset(processed_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    for epoch in range(epochs):
        model.train()
        total_loss, pol_loss, val_loss = 0, 0, 0
        
        for state, mask, target_p, target_v in loader:
            state = state.to(device)
            mask = mask.to(device)
            target_p = target_p.to(device)
            target_v = target_v.to(device)
            
            optimizer.zero_grad()
            
            p, v = model(state, mask)
            
            # Policy loss: Cross Entropy between target distribution and predicted distribution
            # Ensure targets sum to 1
            target_p_sum = target_p.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            target_p = target_p / target_p_sum
            
            # Use KL divergence manually
            kl_div = -(target_p * torch.log(p.clamp(min=1e-8))).sum(dim=-1).mean()
            
            # Value loss: MSE
            mse = nn.MSELoss()(v, target_v)
            
            loss = kl_div + mse
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pol_loss += kl_div.item()
            val_loss += mse.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f} | Policy: {pol_loss/len(loader):.4f} | Value: {val_loss/len(loader):.4f}")
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved latest model to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data", type=str, default="data/self_play_data.pt")
    parser.add_argument("--save", type=str, default="models/alphazero_latest.pt")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    
    train_model(args.model, args.data, args.save, args.epochs)
