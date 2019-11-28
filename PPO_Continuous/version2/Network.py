import torch
import torch.nn as nn
import torch.nn.functional as F


# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(Actor, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 64)
#         self.fc2 = nn.Linear(64, 8)
#         self.mu_head = nn.Linear(8, action_dim)
#         self.sigma_head = nn.Linear(8, action_dim)
#
#     def forward(self, x):
#         x = F.leaky_relu(self.fc1(x))
#         x = F.leaky_relu(self.fc2(x))
#
#         mu = 2.0 * torch.tanh(self.mu_head(x))
#         sigma = F.softplus(self.sigma_head(x))
#
#         return mu, sigma
#
#
# class Critic(nn.Module):
#     def __init__(self, state_dim):
#         super(Critic, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 64)
#         self.fc2 = nn.Linear(64, 8)
#         self.state_value = nn.Linear(8, 1)
#
#     def forward(self, x):
#         x = F.leaky_relu(self.fc1(x))
#         x = F.leaky_relu(self.fc2(x))
#         value = self.state_value(x)
#         return value


# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(Actor, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 100)
#         self.fc2 = nn.Linear(100, 100)
#         self.mu = nn.Linear(100, action_dim)
#         self.sigma = nn.Linear(100, action_dim)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         mu = 2.0 * F.tanh(self.mu(x))
#         sigma = F.softplus(self.sigma(x))
#         return mu, sigma
#
#
# class Critic(nn.Module):
#     def __init__(self, state_dim):
#         super(Critic, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 100)
#         self.fc2 = nn.Linear(100, 100)
#         self.fcv = nn.Linear(100, 1)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         value = self.fcv(x)
#         return value


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fca1 = nn.Linear(state_dim, 30)
        self.fca2 = nn.Linear(30, 30)
        self.mu_head = nn.Linear(30, action_dim)
        self.std_head = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, x):
        x = F.relu(self.fca1(x))
        x = F.relu(self.fca2(x))
        action_mean = 2.0 * torch.tanh(self.mu_head(x))
        action_log_std = self.std_head.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        return action_mean, action_std


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fcv1 = nn.Linear(state_dim, 30)
        self.fcv2 = nn.Linear(30, 30)
        self.val_head = nn.Linear(30, 1)

    def forward(self, x):
        v_x = F.relu(self.fcv1(x))
        v_x = F.relu(self.fcv2(v_x))
        value = self.val_head(v_x)
        return value




