import torch
import torch.nn as nn
import torch.nn.functional as F

class TopologicalOpponentNet(nn.Module):
    def __init__(self, input_dim=334, hidden_dim1=256, hidden_dim2=128):
        super(TopologicalOpponentNet, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        # Outputs 15 logits (3 opponents * 5 topological buckets)
        self.fc3 = nn.Linear(hidden_dim2, 15)
        
    def forward(self, x, gap_capacities=None):
        """
        x: (batch_size, 125)
        gap_capacities: (batch_size, 5) optional tensor containing the count of valid cards per bucket
        """
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        logits = self.fc3(out) # (batch_size, 15)
        
        # Reshape to (batch_size, 3, 5) for the 3 opponents and 5 buckets
        logits = logits.view(-1, 3, 5)
        
        if gap_capacities is not None:
            # gap_capacities shape: (batch_size, 5)
            # Create a boolean mask: True if capacity == 0
            mask = (gap_capacities == 0).unsqueeze(1).expand(-1, 3, -1) # (batch_size, 3, 5)
            
            # Mask out impossible buckets by setting their logits to a large negative number
            # This ensures their Softmax probability is exactly 0.0
            logits = logits.masked_fill(mask, -1e9)
            
        # Apply Softmax across the 5 buckets (dim=2) to get a probability distribution
        probs = F.softmax(logits, dim=2)
        
        return probs

def compute_kl_loss(pred_probs, target_probs, gap_capacities=None):
    """
    Computes KL Divergence loss, with optional masking to prevent penalizing the network
    for buckets that are physically impossible in the current board state.
    """
    # PyTorch KLDivLoss expects input in log-space
    log_probs = torch.log(pred_probs + 1e-10)
    
    # Calculate pointwise KL Divergence: target * (log(target) - log_probs)
    # Target may be 0, so we handle it cleanly
    kl_div = target_probs * (torch.log(target_probs + 1e-10) - log_probs)
    
    if gap_capacities is not None:
        # Mask out the impossible gaps from the loss calculation
        mask = (gap_capacities > 0).unsqueeze(1).expand(-1, 3, -1).float()
        kl_div = kl_div * mask
        
    # Sum over the 5 buckets and 3 opponents, mean over the batch
    loss = kl_div.sum(dim=(1, 2)).mean()
    return loss
