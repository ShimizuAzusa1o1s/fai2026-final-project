"""
Phase 2: Opponent Hand Prediction Model Architecture

Two model options:
1. FastMLP: 3-layer feedforward network (simple, fast inference ~1ms)
2. TransformerModel: Sequence-aware (slower but more accurate)

For the 0.90-second time constraint, we recommend FastMLP by default.
The model predicts: Given observable state, what's P(opponent holds card i) for each card?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FastMLP(nn.Module):
    """
    Fast MLP for opponent hand prediction.
    
    Input: Observable state (524 features = 5x104 cards + 4 scores)
    Output: Probability distribution over 104 cards for opponent's hand
    
    Architecture:
    - Input: 524
    - Hidden 1: 512 (ReLU + Dropout)
    - Hidden 2: 256 (ReLU + Dropout)
    - Hidden 3: 128 (ReLU + Dropout)
    - Output: 104 (Sigmoid for multi-label prediction)
    
    Inference time: ~1-2ms (CPU)
    """
    
    def __init__(self, input_size: int = 524, hidden_size: int = 512):
        super(FastMLP, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 104)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Batch of observable states, shape (batch_size, 524)
            
        Returns:
            Predicted hand probabilities, shape (batch_size, 104)
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        
        x = torch.sigmoid(self.fc4(x))  # Multi-label: each card independent
        
        return x


class LSTMModel(nn.Module):
    """
    Sequence-aware LSTM for opponent hand prediction.
    
    More sophisticated model that processes game history as a sequence.
    Input: Sequence of board states (history)
    Output: Belief state over opponent's remaining cards
    
    Architecture:
    - LSTM layers to process history
    - Attention mechanism to focus on relevant actions
    - Output: 104-dimensional belief state
    
    Inference time: ~5-10ms (CPU)
    
    Note: This is slower but can be more accurate if card play history matters.
    """
    
    def __init__(self, state_size: int = 524, hidden_size: int = 128, num_layers: int = 2):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=state_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 104)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for LSTM.
        
        Args:
            x: Sequence of states, shape (batch_size, seq_len, 520)
            
        Returns:
            Hand probability, shape (batch_size, 104)
        """
        # LSTM processes sequence
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Optional: Apply attention on the sequence
        attn_out, _ = self.attention(last_hidden.unsqueeze(1), lstm_out, lstm_out)
        attn_out = attn_out.squeeze(1)
        
        # Decode to hand prediction
        x = F.relu(self.fc1(attn_out))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = torch.sigmoid(self.fc3(x))
        
        return x


class TransformerModel(nn.Module):
    """
    Transformer-based model for opponent hand prediction.
    
    Uses self-attention to focus on relevant game history.
    Can capture long-range dependencies in play patterns.
    
    Inference time: ~3-5ms (CPU)
    """
    
    def __init__(self, state_size: int = 520, hidden_size: int = 256, num_heads: int = 4):
        super(TransformerModel, self).__init__()
        
        self.embedding = nn.Linear(state_size, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.3,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 104)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Transformer.
        
        Args:
            x: Sequence of states, shape (batch_size, seq_len, 524)
            
        Returns:
            Hand probability, shape (batch_size, 104)
        """
        # Embed to hidden dimension
        x = self.embedding(x)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Take last token's representation
        x = x[:, -1, :]
        
        # Decode to hand
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = torch.sigmoid(self.fc3(x))
        
        return x


def create_model(model_type: str = "fastmlp", **kwargs) -> nn.Module:
    """
    Factory function to create opponent hand prediction model.
    
    Args:
        model_type: "fastmlp" (recommended), "lstm", or "transformer"
        **kwargs: Additional arguments passed to model constructor
        
    Returns:
        Initialized model
    """
    if model_type == "fastmlp":
        return FastMLP(**kwargs)
    elif model_type == "lstm":
        return LSTMModel(**kwargs)
    elif model_type == "transformer":
        return TransformerModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model instantiation
    print("Testing model architectures...")
    
    # FastMLP
    model = FastMLP()
    x = torch.randn(32, 524)  # 524-dim feature vector
    y = model(x)
    print(f"✓ FastMLP: input {x.shape} -> output {y.shape}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # LSTM
    model = LSTMModel()
    x = torch.randn(32, 10, 524)  # 10 time steps, 524 dims
    y = model(x)
    print(f"✓ LSTMModel: input {x.shape} -> output {y.shape}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Transformer
    model = TransformerModel()
    x = torch.randn(32, 10, 524)  # 10 time steps, 524 dims
    y = model(x)
    print(f"✓ TransformerModel: input {x.shape} -> output {y.shape}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nRecommendation: Use FastMLP for 0.90s time constraint")
    print("  - Fastest inference (~1-2ms)")
    print("  - Smallest model (~350K params)")
    print("  - Good accuracy for simple belief state")
