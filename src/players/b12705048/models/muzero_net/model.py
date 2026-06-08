import torch
import torch.nn as nn
import torch.nn.functional as F

class RepresentationNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, hidden_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        hidden = F.relu(self.fc2(x))
        # Min-max normalize hidden state to bounded domain (helps training stability in MuZero)
        hidden = (hidden - hidden.min(dim=-1, keepdim=True)[0]) / (
            hidden.max(dim=-1, keepdim=True)[0] - hidden.min(dim=-1, keepdim=True)[0] + 1e-5
        )
        return hidden

class DynamicsNetwork(nn.Module):
    def __init__(self, hidden_dim, action_dim):
        super().__init__()
        # Concatenate hidden state and one-hot action
        self.fc1 = nn.Linear(hidden_dim + action_dim, 256)
        
        self.fc_reward = nn.Linear(256, 1)
        self.fc_hidden = nn.Linear(256, hidden_dim)

    def forward(self, hidden_state, action):
        """
        hidden_state: [batch_size, hidden_dim]
        action: [batch_size, action_dim] (one-hot encoded)
        """
        x = torch.cat([hidden_state, action], dim=-1)
        x = F.relu(self.fc1(x))
        
        reward = self.fc_reward(x)
        next_hidden = F.relu(self.fc_hidden(x))
        
        # Min-max normalize hidden state
        next_hidden = (next_hidden - next_hidden.min(dim=-1, keepdim=True)[0]) / (
            next_hidden.max(dim=-1, keepdim=True)[0] - next_hidden.min(dim=-1, keepdim=True)[0] + 1e-5
        )
        return next_hidden, reward

class PredictionNetwork(nn.Module):
    def __init__(self, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, 256)
        
        self.fc_policy = nn.Linear(256, action_dim)
        self.fc_value = nn.Linear(256, 1)

    def forward(self, hidden_state):
        x = F.relu(self.fc1(hidden_state))
        policy_logits = self.fc_policy(x)
        value = self.fc_value(x)
        return policy_logits, value

class MuZeroNet(nn.Module):
    """
    Core MuZero Neural Network Architecture.
    Combines Representation, Dynamics, and Prediction networks.
    """
    def __init__(self, obs_dim=230, hidden_dim=128, action_dim=105):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        self.representation = RepresentationNetwork(obs_dim, hidden_dim)
        self.dynamics = DynamicsNetwork(hidden_dim, action_dim)
        self.prediction = PredictionNetwork(hidden_dim, action_dim)

    def initial_inference(self, obs):
        """
        Produce initial hidden state, value, and policy from the real observation.
        """
        hidden_state = self.representation(obs)
        policy_logits, value = self.prediction(hidden_state)
        return hidden_state, policy_logits, value

    def recurrent_inference(self, hidden_state, action):
        """
        Produce next hidden state, reward, value, and policy from current hidden state and action.
        """
        next_hidden_state, reward = self.dynamics(hidden_state, action)
        policy_logits, value = self.prediction(next_hidden_state)
        return next_hidden_state, reward, policy_logits, value
