import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from math import log


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 100)
        self.fc2 = nn.Linear(100, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        probs = F.softmax(self.fc2(x), dim=1)
        return probs


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.fc2(x)
        return value
