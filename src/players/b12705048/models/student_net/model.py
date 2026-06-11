import torch
import torch.nn as nn
import torch.nn.functional as F

class StudentPolicyNet(nn.Module):
    """
    A pure policy network for Teacher-Student Distillation.
    Takes imperfect information state (V2 334-dim features)
    and predicts the perfectly optimal action (target from the Oracle).
    """
    def __init__(self, obs_dim=334, action_dim=105, hidden_dim=512):
        super().__init__()
        
        self.input_dim = obs_dim
        
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(p=0.3)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs, hand_mask=None):
        """
        obs: [batch_size, 334] - Complete game state representation
        hand_mask: [batch_size, 105] - Optional, if None it is extracted from obs
        """
        x = F.relu(self.ln1(self.fc1(obs)))
        x = self.dropout(x)
        
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.dropout(x)
        
        x = F.relu(self.ln4(self.fc4(x)))
        x = self.dropout(x)
        
        logits = self.policy_head(x)
        
        if hand_mask is None:
            # Extract hand mask from the first 104 dims of the observation (padded to 105)
            # Obs layout: 0-103 is my_hand_mask (0-indexed internally)
            batch_size = obs.size(0)
            device = obs.device
            hand_mask = torch.zeros((batch_size, 105), dtype=torch.float32, device=device)
            hand_mask[:, 1:105] = obs[:, 0:104]
            
        # Mask out illegal actions (cards not in hand)
        # Apply a large negative number before softmax
        logits = logits.masked_fill(hand_mask == 0, -1e9)
        
        return logits
