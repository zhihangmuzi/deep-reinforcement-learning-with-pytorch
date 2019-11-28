import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.nn.utils import clip_grad_norm_

from PPO_Continuous.version2.Network import Actor, Critic


class PPO(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 lr_a=0.001,
                 lr_c=0.001,
                 reward_decay=0.99,
                 memory_size=1000,
                 batch_size=32,
                 update_times=10,
                 clip_param=0.2,
                 max_grad_norm=0.5,
                 device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.gamma = reward_decay
        self.capacity = memory_size
        self.batch_size = batch_size
        self.update_times = update_times
        self.clip_param = clip_param
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.ReplayBuffer = []
        self.counter = 0

        self.actor = Actor(self.state_dim, self.action_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)

        self.critic = Critic(self.state_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        self.counter = 0
        self.training_step = 0

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        mu, sigma = self.actor(state)
        dist = Normal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        action = action.clamp(-2, 2)
        return action.item(), action_log_prob.item()

    def store_transition(self, transition):
        self.ReplayBuffer.append(transition)
        self.counter += 1
        return self.counter % self.capacity == 0

    def update(self):
        old_state = torch.Tensor([t.state for t in self.ReplayBuffer]).float()
        old_action = torch.Tensor([t.action for t in self.ReplayBuffer]).float().view(-1, 1)
        reward = torch.Tensor([t.reward for t in self.ReplayBuffer]).float().view(-1, 1)
        next_state = torch.Tensor([t.next_state for t in self.ReplayBuffer]).float()
        old_action_log_prob = torch.Tensor([t.a_log_prob for t in self.ReplayBuffer]).float().view(-1, 1)

        reward = (reward - reward.mean()) / (reward.std() + 1e-10)
        with torch.no_grad():
            gt = reward + self.gamma * self.critic(next_state)

        for i in range(self.update_times):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.ReplayBuffer))), self.batch_size, False):
                gt_value = gt[index].view(-1, 1)
                value = self.critic(old_state[index])
                advantage = (gt_value - value).detach()

                mu, sigma = self.actor(old_state[index])
                m = Normal(torch.squeeze(mu), torch.squeeze(sigma))
                action_log_prob = m.log_prob(torch.squeeze(old_action[index])).view(-1, 1)
                ratio = torch.exp(action_log_prob - old_action_log_prob[index])

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                action_loss = -torch.min(surr1, surr2).mean()
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(gt_value, value)
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                self.training_step += 1

        del self.ReplayBuffer[:]

