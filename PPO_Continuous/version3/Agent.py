import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from PPO_Continuous.version3.Network import ActorCritic


class PPO(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 learning_rate=0.0003,
                 memory_size=1000,
                 reward_decay=0.99,
                 update_times=10,
                 batch_size=32,
                 clip_param=0.2,
                 max_grad_norm=0.5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = reward_decay
        self.capacity = memory_size
        self.update_times = update_times
        self.batch_size = batch_size
        self.clip_param = clip_param
        self.max_grad_norm = max_grad_norm
        self.ReplayBuffer = []

        self.counter = 0
        self.training_step = 0

        self.policy = ActorCritic(state_dim=self.state_dim, action_dim=self.action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action, log_prob = self.policy.act(state)
        return action, log_prob

    def store_transition(self, transition):
        self.ReplayBuffer.append(transition)
        self.counter += 1
        return self.counter % self.capacity == 0

    def update(self):
        self.training_step += 1

        old_state = torch.Tensor([t.state for t in self.ReplayBuffer]).float()
        old_action = torch.Tensor([t.action for t in self.ReplayBuffer]).float().view(-1, 1)
        reward = torch.Tensor([t.reward for t in self.ReplayBuffer]).float().view(-1, 1)
        next_state = torch.Tensor([t.next_state for t in self.ReplayBuffer]).float()
        old_action_log_prob = torch.Tensor([t.a_log_prob for t in self.ReplayBuffer]).float().view(-1, 1)

        reward = (reward - reward.mean()) / (reward.std() + 1e-10)
        with torch.no_grad():
            gt = reward + self.gamma * self.policy.critic_layer(next_state)

        for i in range(self.update_times):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.ReplayBuffer))), self.batch_size, False):
                action_log_prob, value, dist_entropy = self.policy.evaluate(old_state[index], old_action[index])

                gt_value = gt[index].view(-1, 1)
                advantage = (gt_value - value).detach()

                ratio = torch.exp(action_log_prob - old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage
                loss = -torch.min(surr1, surr2) + 0.5 * F.mse_loss(value, gt_value) - 0.01 * dist_entropy

                self.optimizer.zero_grad()
                loss.mean().backward()
                clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        del self.ReplayBuffer[:]
