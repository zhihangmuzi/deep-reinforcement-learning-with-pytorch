import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 100)
        self.fc1.weight.data.normal_(0, 0.3)
        self.fc2 = nn.Linear(100, action_dim)
        self.fc2.weight.data.normal_(0, 0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action = 2.0 * torch.tanh(self.fc2(x))
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 100)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(100, 1)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], 1)))
        val = self.fc2(x)
        return val


