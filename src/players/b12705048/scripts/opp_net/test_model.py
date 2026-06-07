import os
import sys
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
from src.players.b12705048.models.opp_net.model import TopologicalOpponentNet, compute_kl_loss

def test_pipeline():
    dataset_path = "test_dataset.npz"
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found. Run generate_dataset.py first.")
        return
        
    data = np.load(dataset_path)
    X = torch.tensor(data['X'], dtype=torch.float32)
    Y = torch.tensor(data['Y'], dtype=torch.float32)
    C = torch.tensor(data['C'], dtype=torch.float32)
    
    print(f"Loaded dataset: X {X.shape}, Y {Y.shape}, C {C.shape}")
    
    # Initialize model
    model = TopologicalOpponentNet(input_dim=125)
    
    # Forward pass with mask
    probs = model(X, gap_capacities=C)
    
    print(f"Predictions shape: {probs.shape}")
    
    # Verify mask
    # If C == 0, probs should be 0
    mask = (C == 0).unsqueeze(1).expand(-1, 3, -1)
    masked_probs = probs[mask]
    
    if len(masked_probs) > 0:
        max_prob_in_masked = masked_probs.max().item()
        print(f"Max probability assigned to 0-capacity gap: {max_prob_in_masked:.10f}")
        assert max_prob_in_masked < 1e-6, "Masking failed! Assigned probability to an impossible gap."
    else:
        print("No zero-capacity gaps found in this batch to test masking.")
        
    # Verify probabilities sum to 1
    row_sums = probs.sum(dim=2)
    assert torch.allclose(row_sums, torch.ones_like(row_sums)), "Probabilities do not sum to 1!"
    
    # Compute loss
    loss = compute_kl_loss(probs, Y, gap_capacities=C)
    print(f"Initial untrained loss: {loss.item():.4f}")
    print("\nSUCCESS! The pipeline is fully integrated.")

if __name__ == "__main__":
    test_pipeline()
