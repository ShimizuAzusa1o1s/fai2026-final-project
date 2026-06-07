import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueNetwork(nn.Module):
    """
    Predicts the expected additional penalty a player will take by the end
    of the round, given an intermediate board state and their hand.
    
    Input: 232 dimensions
        - 12: Public Board
        - 3: Player State
        - 9: Opponents State
        - 104: Player Hand Mask
        - 104: Unseen Card Mask
    Output: 1 dimension (Expected Penalty)
    """
    def __init__(self, input_dim=232, hidden_dims=[512, 256, 128]):
        super(ValueNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = h_dim
            
        self.feature_extractor = nn.Sequential(*layers)
        self.value_head = nn.Linear(prev_dim, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Tensor of shape (batch_size, 232).
        Returns:
            torch.Tensor: Tensor of shape (batch_size, 1).
        """
        features = self.feature_extractor(x)
        value = self.value_head(features)
        
        # Penalties are non-negative, so we use ReLU on the final output
        # to ensure it never predicts a negative penalty.
        return F.relu(value)
