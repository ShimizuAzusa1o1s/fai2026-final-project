import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.getcwd())
from src.players.b12705048.core.features import extract_features

class OpponentModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 53 features -> 64 -> 32 -> 1
        self.net = nn.Sequential(
            nn.Linear(53, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # x is (B, 10, 53)
        # return shape: (B, 10)
        B, num_cards, f_dim = x.shape
        x_flat = x.view(B * num_cards, f_dim)
        scores_flat = self.net(x_flat)
        return scores_flat.view(B, num_cards)

def train_model(dataset_path="results/dataset.pkl", output_weights="src/players/b12705048/agents/nn_weights.npz"):
    print(f"Loading dataset from {dataset_path}...")
    try:
        with open(dataset_path, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {dataset_path} not found.")
        sys.exit(1)
        
    print(f"Loaded {len(data)} samples. Extracting 53-dim features for CrossEntropy training...")
    
    X_all = []
    Y_all = []
    Mask_all = []
    
    for sample in data:
        if "scores" not in sample:
            continue
            
        unseen_set = set(sample["unseen_cards"])
        features_143 = extract_features(
            board=sample["board"],
            hand=sample["hand"],
            unseen=unseen_set,
            scores=sample["scores"],
            player_idx=0,
            round_num=sample["round_num"],
            history_matrix=sample["history_matrix"],
            score_history=sample["score_history"],
            board_history=sample["board_history"]
        )
        
        # State features: 0-14 and 115-142 (43 dims)
        state_features = np.concatenate([features_143[0:15], features_143[115:143]])
        
        sorted_hand = sorted(sample["hand"])
        n_hand = len(sorted_hand)
        action_card = sample["action"]
        
        # (10, 53) feature array for this sample
        sample_x = np.zeros((10, 53), dtype=np.float32)
        sample_mask = np.zeros(10, dtype=bool)
        played_idx = 0
        
        for slot in range(10):
            if slot < n_hand:
                card_features = features_143[15 + slot*10 : 15 + slot*10 + 10]
                sample_x[slot] = np.concatenate([state_features, card_features])
                sample_mask[slot] = True
                
                if sorted_hand[slot] == action_card:
                    played_idx = slot
            else:
                # pad with zeros
                pass
                
        X_all.append(sample_x)
        Y_all.append(played_idx)
        Mask_all.append(sample_mask)
        
    X_tensor = torch.tensor(np.array(X_all), dtype=torch.float32) # (N, 10, 53)
    Y_tensor = torch.tensor(np.array(Y_all), dtype=torch.long)    # (N,)
    Mask_tensor = torch.tensor(np.array(Mask_all), dtype=torch.bool) # (N, 10)
    
    N = len(X_tensor)
    print(f"Extracted {N} valid examples.")
    
    dataset = TensorDataset(X_tensor, Y_tensor, Mask_tensor)
    loader = DataLoader(dataset, batch_size=2048, shuffle=True)
    
    model = OpponentModel()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        for x_b, y_b, mask_b in loader:
            optimizer.zero_grad()
            scores = model(x_b)  # (B, 10)
            
            # Mask invalid cards by setting score to -inf
            scores = scores.masked_fill(~mask_b, -1e9)
            
            loss = criterion(scores, y_b)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(y_b)
            preds = scores.argmax(dim=1)
            correct += (preds == y_b).sum().item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/N:.4f} | Accuracy: {correct/N:.4f}")
        
    print("Training complete. Exporting weights to numpy arrays...")
    
    w1 = model.net[0].weight.data.numpy().T  # (53, 64)
    b1 = model.net[0].bias.data.numpy()      # (64,)
    w2 = model.net[2].weight.data.numpy().T  # (64, 32)
    b2 = model.net[2].bias.data.numpy()      # (32,)
    w3 = model.net[4].weight.data.numpy().T  # (32, 1)
    b3 = model.net[4].bias.data.numpy()      # (1,)
    
    os.makedirs(os.path.dirname(output_weights), exist_ok=True)
    np.savez_compressed(
        output_weights, 
        w1=w1, b1=b1, 
        w2=w2, b2=b2, 
        w3=w3, b3=b3
    )
    print(f"Weights exported to {output_weights}")

if __name__ == "__main__":
    train_model()
