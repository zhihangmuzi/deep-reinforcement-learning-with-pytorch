import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fca1 = nn.Linear(state_dim, 100)
        self.fca2 = nn.Linear(100, 100)
        self.mu_head = nn.Linear(100, action_dim)

        self.action_var = torch.full((action_dim,), 0.5 * 0.5)

        self.fcv1 = nn.Linear(state_dim, 100)
        self.fcv2 = nn.Linear(100, 100)
        self.val_head = nn.Linear(100, 1)

    def forward(self):
        raise NotImplementedError

    def actor_layer(self, x):
        x = F.relu(self.fca1(x))
        x = F.relu(self.fca2(x))
        action_mean = 2.0 * torch.tanh(self.mu_head(x))
        return action_mean

    def critic_layer(self, x):
        v_x = F.relu(self.fcv1(x))
        v_x = F.relu(self.fcv2(v_x))
        value = self.val_head(v_x)
        return value

    def act(self, state):
        action_mean = self.actor_layer(state)
        cov_mat = torch.diag(self.action_var)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        logprob = dist.log_prob(action)
        action = action.clamp(-2, 2)
        return action.item(), logprob.item()

    def evaluate(self, state, action):
        action_mean = self.actor_layer(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mean, cov_mat)
        log_prob = dist.log_prob(torch.squeeze(action))
        dist_entropy = dist.entropy()
        value = self.critic_layer(state)
        return log_prob.view(-1, 1), value, dist_entropy.view(-1, 1)
