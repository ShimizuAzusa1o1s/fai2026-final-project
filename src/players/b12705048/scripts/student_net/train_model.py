import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.players.b12705048.models.student_net.model import StudentPolicyNet

def train_student(dataset_path, epochs=10, batch_size=256, lr=1e-3, save_dir="."):
    print(f"Loading dataset from {dataset_path}...")
    data = np.load(dataset_path)
    X = torch.tensor(data['X'], dtype=torch.float32)
    Y = torch.tensor(data['Y'], dtype=torch.float32)
    
    dataset = TensorDataset(X, Y)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StudentPolicyNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    print(f"Training on {device}...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_X)
            
            log_probs = F.log_softmax(logits, dim=1)
            loss = criterion(log_probs, batch_Y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            preds = logits.argmax(dim=1)
            true_best = batch_Y.argmax(dim=1)
            train_correct += (preds == true_best).sum().item()
            train_total += batch_X.size(0)
            
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                
                logits = model(batch_X)
                log_probs = F.log_softmax(logits, dim=1)
                loss = criterion(log_probs, batch_Y)
                
                val_loss += loss.item() * batch_X.size(0)
                preds = logits.argmax(dim=1)
                true_best = batch_Y.argmax(dim=1)
                val_correct += (preds == true_best).sum().item()
                val_total += batch_X.size(0)
                
        train_loss /= train_total
        train_acc = train_correct / train_total
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "student_weights.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved trained weights to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="distillation_dataset.npz")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "student_net"))
    
    args = parser.parse_args()
    train_student(args.data, args.epochs, args.batch_size, args.lr, args.out_dir)
