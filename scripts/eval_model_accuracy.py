import os
import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.players.b12705048.models.opp_net.model import TopologicalOpponentNet, compute_kl_loss

def evaluate_accuracy(dataset_path="data/dataset_l3.npz", weights_path="src/players/b12705048/models/opp_net/weights_l3.pth"):
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found.")
        return
    if not os.path.exists(weights_path):
        print(f"Error: Weights {weights_path} not found.")
        return

    print(f"Loading dataset from {dataset_path}...")
    data = np.load(dataset_path)
    X = torch.tensor(data['X'], dtype=torch.float32)
    Y = torch.tensor(data['Y'], dtype=torch.float32)
    C = torch.tensor(data['C'], dtype=torch.float32)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")
    
    model = TopologicalOpponentNet(input_dim=125).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    dataset = TensorDataset(X, Y, C)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    total_loss = 0.0
    total_correct = 0
    total_predictions = 0

    with torch.no_grad():
        for batch_x, batch_y, batch_c in loader:
            batch_x, batch_y, batch_c = batch_x.to(device), batch_y.to(device), batch_c.to(device)
            
            probs = model(batch_x, gap_capacities=batch_c)
            loss = compute_kl_loss(probs, batch_y, gap_capacities=batch_c)
            total_loss += loss.item() * batch_x.size(0)
            
            # Calculate top-1 accuracy (argmax prediction vs argmax target)
            # shape of probs: (batch, 3, 5)
            # shape of batch_y: (batch, 3, 5)
            pred_classes = torch.argmax(probs, dim=2)
            target_classes = torch.argmax(batch_y, dim=2)
            
            # Mask out impossible predictions? Not strictly necessary for top-1 if probabilities are correct.
            # Only count opponents where target is meaningful (some targets might be all 0 if hand size is 0, but usually hand > 0)
            target_sums = torch.sum(batch_y, dim=2)
            valid_mask = target_sums > 0
            
            correct = (pred_classes == target_classes) & valid_mask
            
            total_correct += correct.sum().item()
            total_predictions += valid_mask.sum().item()

    avg_loss = total_loss / len(dataset)
    accuracy = total_correct / max(1, total_predictions)
    
    print("================== Evaluation Results ==================")
    print(f"Model Weights: {weights_path}")
    print(f"Dataset:       {dataset_path}")
    print(f"Samples:       {len(dataset)}")
    print(f"Total Loss:    {avg_loss:.4f} (KL Divergence)")
    print(f"Top-1 Acc:     {accuracy * 100:.2f}% ({total_correct}/{total_predictions})")
    print("========================================================")

if __name__ == "__main__":
    evaluate_accuracy()
