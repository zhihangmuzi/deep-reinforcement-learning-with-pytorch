import torch
import numpy as np
from torch.autograd import Variable

from AC.Network import A_Net, C_Net


class AC(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 lr_a=0.001,
                 lr_c=0.01,
                 reward_decay=0.9,
                 device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.gamma = reward_decay
        self.device = device

        self.actor = A_Net()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.lr_a)

        self.critic = C_Net()
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.lr_c)

    def choose_action(self, s):
        observation = torch.unsqueeze(torch.Tensor(s).float(), 0)
        probs = self.actor(observation).detach().numpy()[0]
        return np.random.choice(self.action_dim, p=probs)

    def actor_learn(self, s, a, td):
        state = torch.unsqueeze(torch.Tensor(s).float(), 0)
        log_prob = torch.log(self.actor(Variable(state))[0][a])
        exp_v = torch.mean(log_prob * torch.Tensor(td).float()) * -1
        self.actor_optimizer.zero_grad()
        exp_v.backward()
        self.actor_optimizer.step()
        return exp_v

    def critic_learn(self, s, r, s_):
        observation = torch.unsqueeze(torch.Tensor(s).float(), 0)
        observation_ = torch.unsqueeze(torch.Tensor(s_).float(), 0)
        v = self.critic(Variable(observation))
        v_ = self.critic(Variable(observation_)).detach()
        td_error = r + self.gamma * v_ - v
        loss = td_error ** 2
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        return td_error
