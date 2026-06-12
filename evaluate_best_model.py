import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class OldModel(nn.Module):
    def __init__(self, input_dim=125, hidden_dim1=256, hidden_dim2=128):
        super(OldModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 15)
        
    def forward(self, x, gap_capacities=None):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        logits = self.fc3(out)
        logits = logits.view(-1, 3, 5)
        if gap_capacities is not None:
            mask = (gap_capacities == 0).unsqueeze(1).expand(-1, 3, -1)
            logits = logits.masked_fill(mask, -1e9)
        return F.softmax(logits, dim=2)

def compute_kl_loss(pred_probs, target_probs, gap_capacities=None):
    log_probs = torch.log(pred_probs + 1e-10)
    kl_div = target_probs * (torch.log(target_probs + 1e-10) - log_probs)
    if gap_capacities is not None:
        mask = (gap_capacities > 0).unsqueeze(1).expand(-1, 3, -1).float()
        kl_div = kl_div * mask
    loss = kl_div.sum(dim=(1, 2)).mean()
    return loss

def main():
    model = OldModel()
    model.load_state_dict(torch.load('data/best_model.pth', map_location='cpu'))
    model.eval()
    
    data = np.load('data/dataset_l3.npz')
    X, Y, C = data['X'], data['Y'], data['C']
    
    # Check if legacy unnormalized
    if X.shape[1] == 125 and np.max(X[:, 0:4]) > 5.0:
        X[:, 0:4] /= 104.0
        X[:, 4:8] /= 5.0
        X[:, 8:12] /= 25.0
        X[:, 116::3] /= 10.0
        X[:, 117::3] /= 100.0
        
    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)
    C_t = torch.tensor(C, dtype=torch.float32)
    
    with torch.no_grad():
        preds = model(X_t, gap_capacities=C_t)
        loss = compute_kl_loss(preds, Y_t, gap_capacities=C_t)
        
    pred_classes = torch.argmax(preds, dim=2)
    target_classes = torch.argmax(Y_t, dim=2)
    
    mask = (C_t > 0).unsqueeze(1).expand(-1, 3, -1)
    # Actually just calculate top-1 accuracy on valid buckets
    correct = (pred_classes == target_classes).sum().item()
    total = pred_classes.numel()
    
    print(f"Old Model Top-1 Accuracy: {correct / total * 100:.2f}%")
    print(f"Old Model KL Loss: {loss.item():.4f}")

if __name__ == '__main__':
    main()
