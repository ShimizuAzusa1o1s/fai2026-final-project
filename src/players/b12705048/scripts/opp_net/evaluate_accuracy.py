import os
import sys
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
from src.players.b12705048.models.opp_net.model import TopologicalOpponentNet, compute_kl_loss

def evaluate_model(dataset_filename="large_dataset.npz"):
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", dataset_filename)
    # Fallback to local execution directory if not found in data/
    if not os.path.exists(dataset_path):
        dataset_path = dataset_filename
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "opp_net", "weights.pth")
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found.")
        return
        
    if not os.path.exists(model_path):
        print(f"Error: Model weights {model_path} not found.")
        return
        
    print("Loading dataset...")
    data = np.load(dataset_path)
    X = torch.tensor(data['X'], dtype=torch.float32)
    Y = torch.tensor(data['Y'], dtype=torch.float32)
    C = torch.tensor(data['C'], dtype=torch.float32)
    
    print(f"Dataset shape: X {X.shape}, Y {Y.shape}, C {C.shape}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TopologicalOpponentNet(input_dim=125).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    batch_size = 1024
    num_samples = len(X)
    
    total_kl_loss = 0.0
    total_l1_error = 0.0
    top1_correct = 0
    top1_total = 0
    
    print("Evaluating model...")
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_x = X[i:end_idx].to(device)
            batch_y = Y[i:end_idx].to(device)
            batch_c = C[i:end_idx].to(device)
            
            probs = model(batch_x, gap_capacities=batch_c)
            loss = compute_kl_loss(probs, batch_y, gap_capacities=batch_c)
            
            total_kl_loss += loss.item() * (end_idx - i)
            
            # L1 Error
            l1 = torch.abs(probs - batch_y)
            # mask out impossible gaps
            mask = (batch_c > 0).unsqueeze(1).expand(-1, 3, -1).float()
            l1_masked = l1 * mask
            total_l1_error += l1_masked.sum().item()
            
            # Top-1 Accuracy:
            # Did the model assign highest probability to a gap that the opponent actually has at least one card in?
            # probs shape: (batch, 3, 5)
            # batch_y shape: (batch, 3, 5)
            
            pred_top1 = torch.argmax(probs, dim=2) # (batch, 3)
            # check if batch_y at pred_top1 is > 0
            
            # Gather the true y values at the predicted top1 indices
            true_y_at_top1 = torch.gather(batch_y, 2, pred_top1.unsqueeze(2)).squeeze(2) # (batch, 3)
            
            top1_correct += (true_y_at_top1 > 0).sum().item()
            top1_total += true_y_at_top1.numel()

    avg_kl_loss = total_kl_loss / num_samples
    avg_l1_error = total_l1_error / (num_samples * 3 * 5) # average over all components
    accuracy = top1_correct / top1_total if top1_total > 0 else 0
    
    print("-" * 40)
    print("Evaluation Results:")
    print(f"Average KL Divergence Loss: {avg_kl_loss:.4f}")
    print(f"Average L1 Error (per bucket): {avg_l1_error:.4f}")
    print(f"Top-1 Accuracy (predicting a non-empty bucket): {accuracy * 100:.2f}%")
    print(f"Total Opponent Hands Evaluated: {top1_total}")
    print("-" * 40)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="large_dataset.npz")
    args = parser.parse_args()
    
    evaluate_model(args.data)
