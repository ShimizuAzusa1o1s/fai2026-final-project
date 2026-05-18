import torch
import torch.nn as nn
import torch.nn.functional as F

# Pre-compute bullheads for fast lookup when engineering features
BULLHEADS = [0] * 105
for c in range(1, 105):
    if c == 55: BULLHEADS[c] = 7
    elif c % 11 == 0: BULLHEADS[c] = 5
    elif c % 10 == 0: BULLHEADS[c] = 3
    elif c % 5 == 0: BULLHEADS[c] = 2
    else: BULLHEADS[c] = 1
BULLHEADS = tuple(BULLHEADS)

class StateEncoder:
    """
    Utility class to mathematically encode the 6 Nimmt! game state 
    into a flat Tensor suitable for Neural Networks.
    """
    @staticmethod
    def encode(hand, board):
        """
        Encodes the current hand and board into a 636-dimensional tensor.
        - 104 dims for hand
        - 416 dims for board (4 rows * 104 dims)
        - 104 dims for unseen cards
        - 4 dims for normalized row lengths (length / 5.0)
        - 4 dims for normalized row ends (card / 104.0)
        - 4 dims for normalized row bullheads (sum / 35.0)
        
        Returns:
            torch.Tensor: Shape (636,)
        """
        # Card values are 1-104, we map them to 0-103 indices
        tensor = torch.zeros(636, dtype=torch.float32)
        
        visible_cards = set()
        
        # 1. Encode Hand (Indices 0 to 103)
        for card in hand:
            tensor[card - 1] = 1.0
            visible_cards.add(card)
            
        # 2. Encode Board (Indices 104 to 519)
        for row_idx, row in enumerate(board):
            offset = 104 + (row_idx * 104)
            for card in row:
                tensor[offset + card - 1] = 1.0
                visible_cards.add(card)
                
            # --- NEW: Explicit Engineered Features ---
            # Index 624-627: Row Lengths
            tensor[624 + row_idx] = len(row) / 5.0
            
            # Index 628-631: Row Ends
            tensor[628 + row_idx] = row[-1] / 104.0
            
            # Index 632-635: Row Bullheads
            row_bheads = sum(BULLHEADS[c] for c in row)
            tensor[632 + row_idx] = row_bheads / 35.0
                
        # 3. Encode Unseen Cards (Indices 520 to 623)
        for card in range(1, 105):
            if card not in visible_cards:
                tensor[520 + card - 1] = 1.0
                
        return tensor

    @staticmethod
    def get_legal_mask(hand):
        """
        Returns a boolean mask of shape (104,) indicating which cards are legal to play.
        """
        mask = torch.zeros(104, dtype=torch.bool)
        for card in hand:
            mask[card - 1] = True
        return mask


class RegretNet(nn.Module):
    """
    The Regret Network predicts the Counterfactual Regret for playing each of the 104 cards.
    """
    def __init__(self, input_dim=636, hidden_dims=[512, 512, 512], output_dim=104):
        super(RegretNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(h_dim)) # LayerNorm heavily stabilizes RL training
            prev_dim = h_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # x shape: (batch_size, 636)
        # Returns shape: (batch_size, 104)
        return self.network(x)


class PolicyNet(nn.Module):
    """
    The Policy Network represents the final optimal strategy (Nash Equilibrium).
    """
    def __init__(self, input_dim=636, hidden_dims=[512, 512, 512], output_dim=104):
        super(PolicyNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(h_dim))
            prev_dim = h_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x, legal_mask=None):
        """
        x shape: (batch_size, 636)
        legal_mask: boolean tensor of shape (batch_size, 104)
        """
        logits = self.network(x)
        
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, -1e9)
            
        probs = F.softmax(logits, dim=-1)
        return probs
