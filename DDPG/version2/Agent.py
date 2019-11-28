import torch
import torch.nn as nn
import numpy as np

from DDPG.version2.Network import Actor, Critic
from DDPG.version2.ReplayBuffer import ReplayBuffer
from DDPG.version2.utils import soft_update, hard_update


class DDPG(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 lr_a=0.001,
                 lr_c=0.001,
                 reward_decay=0.9,
                 memory_size=10000,
                 batch_size=32,
                 tau=0.01,
                 device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.gamma = reward_decay
        self.tau = tau
        self.device = device

        self.capacity = memory_size
        self.batch_size = batch_size
        self.ReplayBuffer = ReplayBuffer(self.capacity)

        self.actor = Actor(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
        self.target_actor = Actor(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.lr_a)

        self.critic = Critic(state_dim=self.state_dim, action_dim=action_dim).to(self.device)
        self.target_critic = Critic(state_dim=self.state_dim, action_dim=action_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.lr_c)

        self.loss_func = nn.MSELoss()

        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

    def store_transition(self, s, a, r, s_, done):
        transition = (s, a, r, s_, done)
        self.ReplayBuffer.add(transition)

    def choose_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        action = self.target_actor(state).cpu().detach()
        return action.data.numpy()

    def learn(self):
        batch_memory = self.ReplayBuffer.sample(self.batch_size)

        b_s = torch.Tensor(np.asarray([e[0] for e in batch_memory])).float().to(device=self.device)
        b_a = torch.Tensor(np.asarray([e[1] for e in batch_memory])).float().reshape((-1, 1)).to(device=self.device)
        b_r = torch.Tensor(np.asarray([e[2] for e in batch_memory])).float().to(device=self.device)
        b_s_ = torch.Tensor(np.asarray([e[3] for e in batch_memory])).float().to(device=self.device)
        b_d_ = torch.Tensor(np.asarray([e[4] for e in batch_memory])).float().to(device=self.device)

        # Compute the target Q value
        b_a_ = self.target_actor(b_s).detach()
        next_eval = torch.squeeze(self.target_critic(b_s_, b_a_).detach())
        target_q = b_r + (1 - b_d_) * self.gamma * next_eval

        # Get current Q estimate
        current_q = torch.squeeze(self.critic(b_s, b_a))

        # Compute critic loss
        loss_critic = self.loss_func(target_q, current_q)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        pred_a = self.actor(b_s_)
        loss_actor = -1 * torch.mean(self.critic(b_s, pred_a))
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        soft_update(self.target_actor, self.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)
