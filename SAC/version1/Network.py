import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, min_log_std=-10, max_log_std=2):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_head = nn.Linear(256, action_dim)
        self.log_sigma_head = nn.Linear(256, action_dim)

        self.max_action = max_action
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_sigma = F.relu(self.log_sigma_head(x))
        log_sigma = torch.clamp(log_sigma, self.min_log_std, self.max_log_std)
        return mu, log_sigma


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.val = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        val = self.val(x)
        return val


class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_dim+action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        x = F.relu(self.fc1(sa))
        x = F.relu(self.fc2(x))
        qval = self.fc3(x)
        return qval





