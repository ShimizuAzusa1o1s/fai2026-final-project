"""
Neural Network Training Module
===============================

This module optimizes the AlphaZero neural network using supervised learning
on self-play generated trajectories. The network is trained to:
  1. Predict action probabilities (policy head) from MCTS-improved targets
  2. Predict game outcomes (value head) from final scores

The training uses a combined loss: policy KL divergence + value MSE.

Key components:
  - SelfPlayDataset: PyTorch dataset wrapper for training data
  - train_model(): Optimizes the network for specified epochs
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

import sys
sys.path.append(os.getcwd())

from src.players.b12705048.alphazero.model import TinyAlphaZeroNet
from src.players.b12705048.alphazero.state_encoding import get_state_dim, N_CARDS


class SelfPlayDataset(Dataset):
    """
    PyTorch Dataset wrapper for self-play training data.
    
    Converts the raw training data tuples into tensors suitable for batch training.
    Expects data in format: [(state_vec, mask, target_probs, value), ...]
    """
    
    def __init__(self, data):
        """
        Initialize dataset from raw self-play data.
        
        Args:
            data (list): Training tuples from self_play.py
        """
        # Extract individual components from data tuples
        self.states = [d[0] for d in data]
        self.masks = [d[1] for d in data]
        self.probs = [d[2] for d in data]
        self.values = [d[3] for d in data]
        
    def __len__(self):
        """Return dataset size."""
        return len(self.states)
        
    def __getitem__(self, idx):
        """
        Return a single training example as PyTorch tensors.
        
        Returns:
            tuple: (state, mask, policy_target, value_target)
        """
        return (
            torch.FloatTensor(self.states[idx]),
            torch.FloatTensor(self.masks[idx]),
            torch.FloatTensor(self.probs[idx]),
            torch.FloatTensor([self.values[idx]])
        )


def train_model(model_path=None, data_path="data/self_play_data.pt", 
                save_path="models/latest.pt", epochs=5, batch_size=256, lr=5e-4):
    """
    Train the AlphaZero network on self-play data.
    
    This function:
      1. Loads or initializes the neural network
      2. Loads self-play training data
      3. Optimizes the network for specified epochs using Adam optimizer
      4. Saves the trained model
    
    Args:
        model_path (str, optional): Path to existing model checkpoint to fine-tune.
                                   If None, trains from random initialization.
        data_path (str): Path to self-play training data (PyTorch tensor file).
        save_path (str): Path where to save the trained model.
        epochs (int): Number of training epochs. Default is 5.
        batch_size (int): Batch size for training. Default is 256.
        lr (float): Learning rate for Adam optimizer. Default is 5e-4.
    """
    # Determine device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Initialize network architecture
    state_dim = get_state_dim()
    n_actions = N_CARDS
    
    model = TinyAlphaZeroNet(state_dim, n_actions).to(device)
    
    # Load pre-trained weights if available
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded existing model from {model_path}")
        
    # Load self-play training data from disk
    print(f"Loading data from {data_path}...")
    data = torch.load(data_path, weights_only=False)
    
    # Preprocess and validate training data format
    # The format from self_play_episode: (state_vec, mask, target_probs, value)
    processed_data = []
    
    for i in range(len(data)):
        state_vec, mask, target_probs, val_tuple = data[i]
        
        # Handle potential format inconsistencies
        # Ensure val_tuple is a scalar (not a tuple wrapper)
        if isinstance(val_tuple, tuple):
            val = val_tuple[-1]
        else:
            val = val_tuple
             
        processed_data.append((state_vec, mask, target_probs, val))
        
    # Create PyTorch dataset and dataloader
    dataset = SelfPlayDataset(processed_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize optimizer: Adam with L2 regularization
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Training loop
    for epoch in range(epochs):
        model.train()  # Set model to training mode (enables dropout, batch norm updates)
        total_loss, pol_loss, val_loss = 0, 0, 0
        
        # Process batches of training data
        for state, mask, target_p, target_v in loader:
            # Move batch to training device
            state = state.to(device)
            mask = mask.to(device)
            target_p = target_p.to(device)
            target_v = target_v.to(device)
            
            # Forward pass: compute network predictions
            optimizer.zero_grad()
            p, v = model(state, mask)
            
            # Policy loss: KL divergence between MCTS targets and network predictions
            # Normalize target probabilities to ensure they sum to 1
            target_p_sum = target_p.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            target_p = target_p / target_p_sum
            
            # KL divergence: -sum(target * log(predicted))
            # Measures how well the network's policy matches MCTS's improved policy
            kl_div = -(target_p * torch.log(p.clamp(min=1e-8))).sum(dim=-1).mean()
            
            # Value loss: Mean Squared Error between value targets and predictions
            # Targets are normalized game outcomes ([-1, 1])
            mse = nn.MSELoss()(v, target_v)
            
            # Combined loss: weighted sum of policy and value objectives
            loss = kl_div + mse
            
            # Backward pass: compute gradients
            loss.backward()
            optimizer.step()
            
            # Accumulate loss statistics
            total_loss += loss.item()
            pol_loss += kl_div.item()
            val_loss += mse.item()
            
        # Print epoch statistics
        avg_total = total_loss / len(loader)
        avg_policy = pol_loss / len(loader)
        avg_value = val_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_total:.4f} | Policy: {avg_policy:.4f} | Value: {avg_value:.4f}")
        
    # Save trained model weights to disk
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved latest model to {save_path}")


if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments from train_loop.sh
    parser = argparse.ArgumentParser(description="Train AlphaZero network on self-play data")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to existing model checkpoint to fine-tune")
    parser.add_argument("--data", type=str, default="data/self_play_data.pt",
                        help="Path to self-play training data")
    parser.add_argument("--save", type=str, default="models/alphazero_latest.pt",
                        help="Path to save trained model")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate for Adam optimizer")
    args = parser.parse_args()
    
    # Execute training
    train_model(
        model_path=args.model,
        data_path=args.data,
        save_path=args.save,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
