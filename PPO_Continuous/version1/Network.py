import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fca1 = nn.Linear(state_dim, 30)
        self.fca2 = nn.Linear(30, 30)
        self.mu_head = nn.Linear(30, action_dim)
        self.std_head = nn.Parameter(torch.zeros(1, action_dim))

        self.fcv1 = nn.Linear(state_dim, 30)
        self.fcv2 = nn.Linear(30, 30)
        self.val_head = nn.Linear(30, 1)

    def forward(self):
        raise NotImplementedError

    def actor_layer(self, x):
        x = F.relu(self.fca1(x))
        x = F.relu(self.fca2(x))
        action_mean = 2.0 * torch.tanh(self.mu_head(x))
        action_log_std = self.std_head.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        return action_mean, action_std

    def critic_layer(self, x):
        v_x = F.relu(self.fcv1(x))
        v_x = F.relu(self.fcv2(v_x))
        value = self.val_head(v_x)
        return value

    def act(self, state):
        action_mean, action_std = self.actor_layer(state)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        logprob = dist.log_prob(action)
        action = action.clamp(-2, 2)
        return action.item(), logprob.item()

    def evaluate(self, state, action):
        action_mean, action_std = self.actor_layer(state)
        dist = Normal(torch.squeeze(action_mean), torch.squeeze(action_std))
        log_prob = dist.log_prob(torch.squeeze(action))
        dist_entropy = dist.entropy()

        value = self.critic_layer(state)
        return log_prob.view(-1, 1), value, dist_entropy.view(-1, 1)










