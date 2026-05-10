"""
AlphaZero Neural Network Architecture
====================================

Implements neural networks used by AlphaZero for policy and value prediction.

Two architectures are available:
  - AlphaZeroNet: Deeper network with residual connections and LayerNorm (default)
  - TinyAlphaZeroNet: Original 2-layer MLP (kept for loading old checkpoints)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """
    Pre-activation residual block with LayerNorm.
    
    Architecture: x → LN → ReLU → Linear → LN → ReLU → Linear → + x
    Skip connection preserves gradient flow through deep networks.
    """
    
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.ln1(self.fc1(x)))
        out = self.ln2(self.fc2(out))
        return F.relu(out + residual)


class AlphaZeroNet(nn.Module):
    """
    Deeper neural network with residual connections for AlphaZero.
    
    Architecture:
      - Input projection: Linear(state_dim → 256) + LayerNorm + ReLU
      - 2 Residual blocks (256-d each with skip connections)
      - Compression: Linear(256 → 128) + LayerNorm + ReLU
      - Policy head: Linear(128 → 104) + mask + softmax
      - Value head: Linear(128 → 64) + ReLU + Linear(64 → 1) + tanh
    
    Compared to TinyAlphaZeroNet:
      - 2 residual blocks add depth without vanishing gradients
      - LayerNorm stabilizes training
      - Deeper value head (2 layers) for better outcome prediction
    """
    
    def __init__(self, state_dim, n_actions=104):
        super().__init__()
        
        # Input projection
        self.input_fc = nn.Linear(state_dim, 256)
        self.input_ln = nn.LayerNorm(256)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(256),
            ResidualBlock(256),
        ])
        
        # Compression layer
        self.compress_fc = nn.Linear(256, 128)
        self.compress_ln = nn.LayerNorm(128)
        
        # Policy head
        self.policy_fc = nn.Linear(128, n_actions)
        
        # Deeper value head
        self.value_fc1 = nn.Linear(128, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, state_tensor, legal_mask):
        """
        Forward pass through the network.
        
        Args:
            state_tensor (torch.Tensor): Batch of encoded states, shape (batch_size, state_dim)
            legal_mask (torch.Tensor): Legal action mask, shape (batch_size, 104)
        
        Returns:
            tuple: (policy_probs, value)
        """
        # Input projection
        x = F.relu(self.input_ln(self.input_fc(state_tensor)))
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Compression
        x = F.relu(self.compress_ln(self.compress_fc(x)))
        
        # Policy head with action masking
        policy_logits = self.policy_fc(x)
        policy_logits = policy_logits - 1e9 * (1.0 - legal_mask)
        policy_probs = F.softmax(policy_logits, dim=-1)
        
        # Value head
        v = F.relu(self.value_fc1(x))
        value = torch.tanh(self.value_fc2(v))
        
        return policy_probs, value

    def predict(self, state, mask):
        """
        Inference function for MCTS — numpy in, numpy out.
        
        Args:
            state (np.ndarray): Encoded game state, shape (state_dim,)
            mask (np.ndarray): Legal action mask, shape (104,)
        
        Returns:
            tuple: (policy, value) as numpy arrays
        """
        device = getattr(self, 'device', torch.device('cpu'))
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        mask_t = torch.FloatTensor(mask).unsqueeze(0).to(device)
        
        with torch.no_grad():
            p, v = self(state_t, mask_t)
            
        p = p.cpu().squeeze(0).numpy()
        v = v.cpu().squeeze(0).item()
        
        p = p * mask
        sum_p = p.sum()
        if sum_p > 0:
            p = p / sum_p
        else:
            p = mask / mask.sum()
            
        return p, v


class TinyAlphaZeroNet(nn.Module):
    """
    Original compact neural network (kept for loading old checkpoints).
    
    Architecture:
      - Shared layers: Linear(state_dim → 256) + ReLU → Linear(256 → 128) + ReLU
      - Policy head: Linear(128 → 104) + mask + softmax
      - Value head: Linear(128 → 1) + tanh
    """
    
    def __init__(self, state_dim, n_actions=104):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.policy_fc = nn.Linear(128, n_actions)
        self.value_fc = nn.Linear(128, 1)

    def forward(self, state_tensor, legal_mask):
        x = F.relu(self.fc1(state_tensor))
        x = F.relu(self.fc2(x))
        
        policy_logits = self.policy_fc(x)
        policy_logits = policy_logits - 1e9 * (1.0 - legal_mask)
        policy_probs = F.softmax(policy_logits, dim=-1)
        
        value = torch.tanh(self.value_fc(x))
        return policy_probs, value

    def predict(self, state, mask):
        device = getattr(self, 'device', torch.device('cpu'))
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        mask_t = torch.FloatTensor(mask).unsqueeze(0).to(device)
        
        with torch.no_grad():
            p, v = self(state_t, mask_t)
            
        p = p.cpu().squeeze(0).numpy()
        v = v.cpu().squeeze(0).item()
        
        p = p * mask
        sum_p = p.sum()
        if sum_p > 0:
            p = p / sum_p
        else:
            p = mask / mask.sum()
            
        return p, v
