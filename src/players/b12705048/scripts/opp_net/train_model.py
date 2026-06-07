import os
import sys
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.players.b12705048.models.opp_net.model import TopologicalOpponentNet, compute_kl_loss

def train_model(dataset_path=None, epochs=50, batch_size=256, lr=1e-3, level=1):
    if dataset_path is None:
        dataset_path = f"dataset_l{level}.npz"
        
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return

    print("Loading dataset...")
    data = np.load(dataset_path)
    X = torch.tensor(data['X'], dtype=torch.float32)
    Y = torch.tensor(data['Y'], dtype=torch.float32)
    C = torch.tensor(data['C'], dtype=torch.float32)
    
    dataset = TensorDataset(X, Y, C)
    
    # 80-20 Train/Val split
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = TopologicalOpponentNet(input_dim=125).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "models", "opp_net")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"weights_l{level}.pth")
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y, batch_c in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_c = batch_c.to(device)
            
            optimizer.zero_grad()
            
            probs = model(batch_x, gap_capacities=batch_c)
            loss = compute_kl_loss(probs, batch_y, gap_capacities=batch_c)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y, batch_c in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_c = batch_c.to(device)
                
                probs = model(batch_x, gap_capacities=batch_c)
                loss = compute_kl_loss(probs, batch_y, gap_capacities=batch_c)
                
                val_loss += loss.item() * batch_x.size(0)
                
        val_loss /= len(val_dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            best_flag = "*"
        else:
            best_flag = ""
            
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} {best_flag}")
        
    print(f"\nTraining complete. Best model saved to {save_path} with Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--level", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    
    train_model(dataset_path=args.data, epochs=args.epochs, batch_size=args.batch, lr=args.lr, level=args.level)
