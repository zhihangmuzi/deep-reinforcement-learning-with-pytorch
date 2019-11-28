import torch
import torch.nn.functional as F
import numpy as np

from TD3.version1.ReplayBuffer import ReplayBuffer
from TD3.version1.Network import Actor, Critic
from TD3.version1.utils import hard_update, soft_update


class TD3(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 learning_rate=0.001,
                 memory_size=50000,
                 tau=0.005,
                 reward_decay=0.99,
                 batch_size=100,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_delay=2,
                 device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.lr = learning_rate
        self.gamma = reward_decay
        self.tau = tau
        self.capacity = memory_size
        self.batch_size = batch_size
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.device = device

        self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(self.device)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.max_action).to(self.device)
        self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, self.action_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        self.max_action = max_action
        self.ReplayBuffer = ReplayBuffer(self.capacity)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).float().to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def store_transition(self, s, a, r, s_, done):
        transition = (s, a, r, s_, done)
        self.ReplayBuffer.add(transition)

    def update(self, num_iteration):

        if self.num_training % 500 == 0:
            print("====================================")
            print("model has been trained for {} times...".format(self.num_training))
            print("====================================")
        for i in range(num_iteration):
            batch_memory = self.ReplayBuffer.sample(self.batch_size)

            state = torch.Tensor(np.asarray([e[0] for e in batch_memory])).float().to(self.device)
            action = torch.Tensor(np.asarray([e[1] for e in batch_memory])).float().reshape((-1, 1)).to(self.device)
            reward = torch.Tensor(np.asarray([e[2] for e in batch_memory])).float().reshape((-1, 1)).to(self.device)
            next_state = torch.Tensor(np.asarray([e[3] for e in batch_memory])).float().to(self.device)
            done = torch.Tensor(np.asarray([e[4] for e in batch_memory])).float().reshape((-1, 1)).to(self.device)

            # Select next action according to target policy:
            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip).to(self.device)

                next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

                # Compute the target Q value
                target_q1, target_q2 = self.critic_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)
                target_q = reward + (1 - done) * self.gamma * target_q

            # Get current Q estimates
            current_q1, current_q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates:
            if i % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                soft_update(self.actor_target, self.actor, self.tau)
                soft_update(self.critic_target, self.critic, self.tau)

                self.num_actor_update_iteration += 1
        self.num_critic_update_iteration += 1
        self.num_training += 1







