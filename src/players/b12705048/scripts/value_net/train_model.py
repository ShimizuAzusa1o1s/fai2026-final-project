import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from src.players.b12705048.models.value_net.model import ValueNetwork

def train(data_path="data/value_dataset.npz", save_path="src/players/b12705048/models/value_net/weights.pth", epochs=50, batch_size=256, lr=1e-3):
    print(f"Loading dataset from {data_path}...")
    try:
        data = np.load(data_path)
        X = data['X']
        Y = data['Y']
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print(f"Dataset loaded: X={X.shape}, Y={Y.shape}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    
    dataset = TensorDataset(X_tensor, Y_tensor)
    
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    model = ValueNetwork(input_dim=232).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1:03d}/{epochs:03d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved new best model to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Value Network.")
    parser.add_argument("--data", type=str, default="data/value_dataset.npz", help="Path to training dataset.")
    parser.add_argument("--out", type=str, default="src/players/b12705048/models/value_net/weights.pth", help="Output weights path.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    
    args = parser.parse_args()
    train(args.data, args.out, args.epochs, args.batch, args.lr)
