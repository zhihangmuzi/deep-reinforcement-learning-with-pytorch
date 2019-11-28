import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fca1 = nn.Linear(state_dim, 100)
        self.fca2 = nn.Linear(100, action_dim)

        self.fcv1 = nn.Linear(state_dim, 100)
        self.fcv2 = nn.Linear(100, 1)

    def forward(self, x):
        raise NotImplementedError

    def actor_layer(self, x):
        x = F.relu(self.fca1(x))
        probs = F.softmax(self.fca2(x), dim=1)
        return probs

    def critic_layer(self, x):
        x = F.relu(self.fcv1(x))
        value = self.fcv2(x)
        return value

    def act(self, state):
        probs = self.actor_layer(state)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        return action.item(), log_prob.item()

    def evaluate(self, state, action):
        probs = self.actor_layer(state)
        m = Categorical(probs)
        log_prob = m.log_prob(torch.squeeze(action))
        value = self.critic_layer(state)
        dist_entropy = m.entropy()
        return log_prob.view(-1, 1), value, dist_entropy.view(-1, 1)



