import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils import clip_grad_norm_

from SAC.version1.Network import Actor, Critic, QNet
from SAC.version1.ReplayBuffer import ReplayBuffer
from SAC.version1.utils import hard_update, soft_update


class SAC(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 learning_rate=0.001,
                 memory_size=10000,
                 tau=0.005,
                 reward_decay=0.99,
                 batch_size=100,
                 device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.lr = learning_rate
        self.gamma = reward_decay
        self.tau = tau
        self.capacity = memory_size
        self.batch_size = batch_size
        self.device = device

        self.actor = Actor(state_dim=self.state_dim, action_dim=self.action_dim, max_action=self.max_action).to(self.device)
        self.critic = Critic(state_dim=self.state_dim).to(self.device)
        self.target_critic = Critic(state_dim=self.state_dim).to(self.device)
        self.q1_net = QNet(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
        self.q2_net = QNet(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.q1_net_optimizer = torch.optim.Adam(self.q1_net.parameters(), lr=self.lr)
        self.q2_net_optimizer = torch.optim.Adam(self.q2_net.parameters(), lr=self.lr)

        self.ReplayBuffer = ReplayBuffer(self.capacity)
        self.num_transition = 0
        self.num_training = 1

        self.alpha = 1.

        hard_update(self.target_critic, self.critic)

    def choose_action(self, state):
        state = torch.Tensor(state).float().to(self.device)
        mu, log_sigma = self.actor(state)
        sigma = torch.exp(log_sigma)
        m = Normal(mu, sigma)
        action_val = m.sample()
        action = torch.tanh(action_val).detach().cpu().numpy()
        return action.item()  # return a scalar, float32

    def store(self, s, a, r, s_, d):
        transition = (s, a, r, s_, np.float(d))
        self.ReplayBuffer.add(transition)

    def evaluate(self, state):
        mu, log_sigma = self.actor(state)
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        noise = Normal(0, 1)

        z = noise.sample()
        action = torch.tanh(mu + sigma*z.to(self.device))
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

            target_value = self.target_critic(next_state)
            next_q_value = reward + (1 - done) * self.gamma * target_value

            expected_value = self.critic(state)
            expected_q1 = self.q1_net(state, action)
            expected_q2 = self.q2_net(state, action)
            sample_action, log_prob, z, batch_mu, batch_log_sigma = self.evaluate(state)
            expected_new_q = torch.min(self.q1_net(state, sample_action), self.q2_net(state, action))
            next_value = expected_new_q - self.alpha * log_prob

            # !!!Note that the actions are sampled according to the current policy,
            # instead of replay buffer. (From original paper)
            val_loss = F.mse_loss(expected_value, next_value.detach()).mean()  # J_V

            # Dual Q net
            q1_loss = F.mse_loss(expected_q1, next_q_value.detach()).mean()  # J_Q
            q2_loss = F.mse_loss(expected_q2, next_q_value.detach()).mean()

            pi_loss = (self.alpha * log_prob - expected_new_q).mean()  # according to original paper

            # mini batch gradient descent
            self.critic.zero_grad()
            val_loss.backward(retain_graph=True)
            clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()

            self.q1_net_optimizer.zero_grad()
            q1_loss.backward(retain_graph=True)
            clip_grad_norm_(self.q1_net.parameters(), 0.5)
            self.q1_net_optimizer.step()

            self.q2_net_optimizer.zero_grad()
            q2_loss.backward(retain_graph=True)
            clip_grad_norm_(self.q2_net.parameters(), 0.5)
            self.q2_net_optimizer.step()

            self.actor_optimizer.zero_grad()
            pi_loss.backward(retain_graph=True)
            clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            # update target v net update
            soft_update(self.target_critic, self.critic, self.tau)

            self.num_training += 1









