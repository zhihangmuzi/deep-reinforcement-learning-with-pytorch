import torch
import numpy as np
from torch.autograd import Variable

from PG.Network import Net


class PG(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 learning_rate=0.01,
                 reward_decay=0.95,
                 device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = reward_decay
        self.device = device

        self.net = Net()

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self.batch_states, self.batch_actions, self.batch_rewards = [], [], []

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def choose_action(self, state):
        prob = self.net(Variable(torch.Tensor(state).float())).detach().numpy()
        action = np.random.choice(np.arange(self.action_dim), p=prob)
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        self.optimizer.zero_grad()

        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        state_tensor = torch.Tensor(self.ep_obs).float()
        reward_tensor = torch.Tensor(discounted_ep_rs_norm).float()
        action_tensor = torch.Tensor(self.ep_as).long()

        log_probs = torch.log(self.net(state_tensor))

        selected_log_probs = reward_tensor * log_probs[np.arange(len(action_tensor)), action_tensor]

        loss = -selected_log_probs.mean()

        loss.backward()
        self.optimizer.step()
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_obs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
