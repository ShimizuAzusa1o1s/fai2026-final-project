import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TinyAlphaZeroNet(nn.Module):
    def __init__(self, state_dim, n_actions=104):
        super().__init__()
        # Shared torso
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        
        # Policy head
        self.policy_fc = nn.Linear(128, n_actions)
        
        # Value head
        self.value_fc = nn.Linear(128, 1)

    def forward(self, state_tensor, legal_mask):
        x = F.relu(self.fc1(state_tensor))
        x = F.relu(self.fc2(x))
        
        # Policy output
        policy_logits = self.policy_fc(x)
        
        # Masking illegal actions with a very negative number
        INF = 1e9
        policy_logits = policy_logits - INF * (1.0 - legal_mask)
        
        # Softmax to get probabilities
        policy_probs = F.softmax(policy_logits, dim=-1)
        
        # Value output in [-1, 1] range (tanh)
        value = torch.tanh(self.value_fc(x))
        
        return policy_probs, value

    def predict(self, state, mask):
        # Convert to single-batch tensor
        state_t = torch.FloatTensor(state).unsqueeze(0)
        mask_t = torch.FloatTensor(mask).unsqueeze(0)
        
        with torch.no_grad():
            p, v = self(state_t, mask_t)
            
        p = p.squeeze(0).numpy()
        v = v.squeeze(0).item()
        
        # Ensure mask is exactly 0
        p = p * mask
        sum_p = p.sum()
        if sum_p > 0:
            p = p / sum_p
        else:
            # Fallback if somehow all 0
            p = mask / mask.sum()
            
        return p, v
