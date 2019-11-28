import torch
import numpy as np
import torch.nn.functional as F

from TD3.version2.Network import Actor, Critic
from TD3.version2.ReplayBuffer import ReplayBuffer
from TD3.version2.utils import hard_update, soft_update


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
        self.critic_1 = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_1_target = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_2 = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_2_target = Critic(self.state_dim, self.action_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters())
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters())

        # self.actor_target.load_state_dict(self.actor.state_dict())
        # self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        # self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_1_target, self.critic_1)
        hard_update(self.critic_2_target, self.critic_2)

        self.max_action = max_action
        self.ReplayBuffer = ReplayBuffer(self.capacity)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.tensor(state.reshape(1, -1)).float().to(self.device)
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
            noise = torch.ones_like(action).data.normal_(0, self.policy_noise).to(self.device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute target Q-value:
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + ((1 - done) * self.gamma * target_q).detach()

            # Optimize Critic 1:
            current_q1 = self.critic_1(state, action)
            loss_q1 = F.mse_loss(current_q1, target_q)
            self.critic_1_optimizer.zero_grad()
            loss_q1.backward()
            self.critic_1_optimizer.step()

            # Optimize Critic 2:
            current_q2 = self.critic_2(state, action)
            loss_q2 = F.mse_loss(current_q2, target_q)
            self.critic_2_optimizer.zero_grad()
            loss_q2.backward()
            self.critic_2_optimizer.step()

            # Delayed policy updates:
            if i % self.policy_delay == 0:
                # Compute actor loss:
                actor_loss = - self.critic_1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                #     target_param.data.copy_(((1 - self.tau) * target_param.data) + self.tau * param.data)
                #
                # for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                #     target_param.data.copy_(((1 - self.tau) * target_param.data) + self.tau * param.data)
                #
                # for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                #     target_param.data.copy_(((1 - self.tau) * target_param.data) + self.tau * param.data)

                soft_update(self.actor_target, self.actor, self.tau)
                soft_update(self.critic_1_target, self.critic_1, self.tau)
                soft_update(self.critic_2_target, self.critic_2, self.tau)

                self.num_actor_update_iteration += 1
        self.num_critic_update_iteration += 1
        self.num_training += 1
