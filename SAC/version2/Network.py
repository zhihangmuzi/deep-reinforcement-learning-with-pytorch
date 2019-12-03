import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_range=1., log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_range = action_range

        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)

        self.mu_head = nn.Linear(512, action_dim)
        self.log_std_head = nn.Linear(512, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std


class softQNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(softQNet, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        x = F.relu(self.fc1(sa))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_val = self.fc4(x)
        return q_val
