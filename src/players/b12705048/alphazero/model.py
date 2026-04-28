"""
AlphaZero Neural Network Architecture
====================================

Implements the neural network used by AlphaZero for policy and value prediction.

Architecture:
  - Input: Encoded game state (233 dimensions)
  - Shared layers: Two fully-connected layers (256 -> 128 units)
  - Policy head: Maps to 104 actions (one per card)
  - Value head: Single value output in [-1, 1] range

The network is trained via supervised learning on self-play trajectories.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TinyAlphaZeroNet(nn.Module):
    """
    Compact neural network for AlphaZero policy and value estimation.
    
    This network combines:
      1. Shared representation layers: Learn general game understanding
      2. Policy head: Outputs probability distribution over legal actions
      3. Value head: Estimates game outcome from this position
    
    Both heads are trained jointly on self-play data.
    """
    
    def __init__(self, state_dim, n_actions=104):
        """
        Initialize network architecture.
        
        Args:
            state_dim (int): Input state vector dimension (typically 233)
            n_actions (int): Number of actions (cards). Default is 104 for 6 Nimmt!
        """
        super().__init__()
        
        # SHARED TORSO: Learn compact state representation
        # Input dimension: state_dim (encoded board, hand, visible cards, scores, round)
        self.fc1 = nn.Linear(state_dim, 256)  # First layer: expansion to 256 units
        self.fc2 = nn.Linear(256, 128)        # Second layer: compression to 128 units
        
        # POLICY HEAD: Output probability distribution over actions
        # Maps from 128-d representation to 104-d action space
        self.policy_fc = nn.Linear(128, n_actions)
        
        # VALUE HEAD: Output scalar value estimate
        # Maps from 128-d representation to single value in [-1, 1]
        self.value_fc = nn.Linear(128, 1)

    def forward(self, state_tensor, legal_mask):
        """
        Forward pass through the network.
        
        Args:
            state_tensor (torch.Tensor): Batch of encoded states, shape (batch_size, state_dim)
            legal_mask (torch.Tensor): Legal action mask, shape (batch_size, 104)
                                      1.0 for legal actions, 0.0 for illegal
        
        Returns:
            tuple: (policy_probs, value)
                   policy_probs: Probability distribution over actions, shape (batch_size, 104)
                   value: Value estimate, shape (batch_size, 1) in range [-1, 1]
        """
        # SHARED LAYERS: Extract features from state
        x = F.relu(self.fc1(state_tensor))   # Apply ReLU after first layer for non-linearity
        x = F.relu(self.fc2(x))              # Apply ReLU after second layer
        
        # POLICY HEAD: Compute action logits
        policy_logits = self.policy_fc(x)
        
        # ACTION MASKING: Prevent illegal actions
        # Set illegal action logits to very negative value, so softmax -> ~0
        INF = 1e9
        policy_logits = policy_logits - INF * (1.0 - legal_mask)
        
        # Convert logits to probabilities via softmax
        # Softmax ensures: sum(probabilities) = 1 and all >= 0
        policy_probs = F.softmax(policy_logits, dim=-1)
        
        # VALUE HEAD: Estimate game outcome
        # Tanh activation maps to [-1, 1] range
        # -1: very bad outcome (losing), +1: very good outcome (winning)
        value = torch.tanh(self.value_fc(x))
        
        return policy_probs, value

    def predict(self, state, mask):
        """
        Inference function for policy and value prediction.
        
        Wrapper around forward() for use in MCTS and gameplay:
          - Handles numpy input (from game state)
          - Runs inference in no_grad() mode (faster, no backprop)
          - Returns numpy outputs (compatible with game code)
        
        Args:
            state (np.ndarray): Encoded game state, shape (state_dim,)
            mask (np.ndarray): Legal action mask, shape (104,) with 0/1 values
        
        Returns:
            tuple: (policy, value)
                   policy: Action probabilities, shape (104,) sum = 1
                   value: Value estimate (scalar float)
        """
        device = getattr(self, 'device', torch.device('cpu'))
        
        # Convert numpy arrays to batched tensors
        # Unsqueeze adds batch dimension: (D,) -> (1, D)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        mask_t = torch.FloatTensor(mask).unsqueeze(0).to(device)
        
        # Forward pass without gradient computation (faster)
        with torch.no_grad():
            p, v = self(state_t, mask_t)
            
        # Convert back to numpy for compatibility with game code
        p = p.cpu().squeeze(0).numpy()  # Remove batch dimension
        v = v.cpu().squeeze(0).item()   # Extract scalar value
        
        # Post-process policy: enforce masking and normalize
        # Multiply by mask to zero out illegal actions (in case softmax didn't fully zero them)
        p = p * mask
        sum_p = p.sum()
        if sum_p > 0:
            # Renormalize to ensure valid probability distribution
            p = p / sum_p
        else:
            # Fallback if somehow all probabilities are 0
            # Use uniform distribution over legal actions
            p = mask / mask.sum()
            
        return p, v
