import torch
import numpy as np
from torch.distributions import Normal
import torch.nn.functional as F

from SAC.version2.Network import Actor, softQNet
from SAC.version2.ReplayBuffer import ReplayBuffer
from SAC.version2.utils import hard_update, soft_update


class SAC(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 lr_a=0.0003,
                 lr_q=0.0003,
                 lr_alpha=0.0003,
                 memory_size=10000,
                 tau=0.005,
                 reward_decay=0.99,
                 batch_size=100,
                 device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.lr_a = lr_a
        self.lr_q = lr_q
        self.lr_alpha = lr_alpha
        self.capacity = memory_size
        self.tau = tau
        self.gamma = reward_decay
        self.batch_size = batch_size
        self.device = device

        self.actor = Actor(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
        self.soft_q1_net = softQNet(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
        self.soft_q2_net = softQNet(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
        self.target_soft_q1_net = softQNet(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
        self.target_soft_q2_net = softQNet(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.soft_q1_optimizer = torch.optim.Adam(self.soft_q1_net.parameters(), lr=self.lr_q)
        self.soft_q2_optimizer = torch.optim.Adam(self.soft_q2_net.parameters(), lr=self.lr_q)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr_alpha)

        hard_update(self.target_soft_q1_net, self.soft_q1_net)
        hard_update(self.target_soft_q2_net, self.soft_q2_net)

        self.ReplayBuffer = ReplayBuffer(self.capacity)
        self.num_training = 1

        self.target_entropy = -2
        self.alpha = 1.

    def choose_action(self, state):
        state = torch.Tensor(state).float().unsqueeze(0).to(self.device)
        mu, log_std = self.actor(state)
        std = torch.exp(log_std)
        m = Normal(mu, std)
        action_val = m.sample()
        action = torch.tanh(action_val).detach().cpu().numpy()
        return action.item()

    def store(self, s, a, r, s_, d):
        transition = (s, a, r, s_, np.float(d))
        self.ReplayBuffer.add(transition)

    def evaluate(self, state):
        mu, log_sigma = self.actor(state)
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        noise = Normal(0, 1)

        z = noise.sample()
        action = torch.tanh(mu + sigma * z.to(self.device))
        log_prob = dist.log_prob(mu + sigma * z.to(self.device)) - torch.log(1 - action.pow(2) + 1e-5)
        return action, log_prob, z, mu, log_sigma

    def update(self):
        if self.num_training % 500 == 0:
            print("Training ... {} times ".format(self.num_training))
        for _ in range(1):
            batch_memory = self.ReplayBuffer.sample(self.batch_size)

            state = torch.Tensor(np.asarray([e[0] for e in batch_memory])).float().to(self.device)
            action = torch.Tensor(np.asarray([e[1] for e in batch_memory])).float().reshape((-1, 1)).to(self.device)
            reward = torch.Tensor(np.asarray([e[2] for e in batch_memory])).float().reshape((-1, 1)).to(self.device)
            next_state = torch.Tensor(np.asarray([e[3] for e in batch_memory])).float().to(self.device)
            done = torch.Tensor(np.asarray([e[4] for e in batch_memory])).float().reshape((-1, 1)).to(self.device)

            predicted_q_value1 = self.soft_q1_net(state, action)
            predicted_q_value2 = self.soft_q2_net(state, action)
            new_action, log_prob, z, mean, log_std = self.evaluate(state)
            new_next_action, next_log_prob, _, _, _ = self.evaluate(next_state)
            reward = 10 * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6)

            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

            target_q_min = torch.min(self.target_soft_q1_net(next_state, new_next_action),
                                     self.target_soft_q2_net(next_state, new_next_action)) - self.alpha * next_log_prob
            target_q_value = reward + (1 - done) * self.gamma * target_q_min  # if done==1, only reward
            q_value_loss1 = F.mse_loss(predicted_q_value1, target_q_value.detach())
            q_value_loss2 = F.mse_loss(predicted_q_value2, target_q_value.detach())

            self.soft_q1_optimizer.zero_grad()
            q_value_loss1.backward()
            self.soft_q1_optimizer.step()
            self.soft_q2_optimizer.zero_grad()
            q_value_loss2.backward()
            self.soft_q2_optimizer.step()

            predicted_new_q_value = torch.min(self.soft_q1_net(state, new_action), self.soft_q2_net(state, new_action))
            policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            soft_update(self.target_soft_q1_net, self.soft_q1_net, self.tau)
            soft_update(self.target_soft_q2_net, self.soft_q2_net, self.tau)

            self.num_training += 1
