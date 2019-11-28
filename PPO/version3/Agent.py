import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F


from PPO.version2.Network import ActorCritic


class PPO(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 lr=0.001,
                 reward_decay=0.99,
                 memory_size=1000,
                 batch_size=32,
                 update_times=10,
                 clip_param=0.2,
                 max_grad_norm=0.5,
                 device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = reward_decay
        self.capacity = memory_size
        self.update_times = update_times
        self.clip_param = clip_param
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.batch_size = batch_size
        self.ReplayBuffer = []
        self.counter = 0

        self.actor_critic = ActorCritic(self.state_dim, self.action_dim)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.lr)

        self.counter = 0
        self.training_step = 0

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action, log_prob = self.actor_critic.act(state)
        return action, log_prob

    def store_transition(self, transition):
        self.ReplayBuffer.append(transition)
        self.counter += 1

    def update(self):
        old_state = torch.Tensor([t.state for t in self.ReplayBuffer]).float()
        old_action = torch.Tensor([t.action for t in self.ReplayBuffer]).long().view(-1, 1)
        reward = torch.Tensor([t.reward for t in self.ReplayBuffer]).float().view(-1, 1)
        next_state = torch.Tensor([t.next_state for t in self.ReplayBuffer]).float()
        old_action_log_prob = torch.Tensor([t.a_log_prob for t in self.ReplayBuffer]).float().view(-1, 1)

        reward = (reward - reward.mean()) / (reward.std() + 1e-10)
        with torch.no_grad():
            gt = reward + self.gamma * self.actor_critic.critic_layer(next_state)

        # print("The agent is updating....")
        for i in range(self.update_times):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.ReplayBuffer))), self.batch_size, False):
                action_log_prob, value, dist_entropy = self.actor_critic.evaluate(old_state[index], old_action[index])

                gt_value = gt[index].view(-1, 1)
                advantage = (gt_value - value).detach()

                ratio = torch.exp(action_log_prob - old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                loss = -torch.min(surr1, surr2) + 0.5 * F.mse_loss(gt_value.detach(), value) - 0.01 * dist_entropy
                self.optimizer.zero_grad()
                loss.mean().backward()
                clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                self.training_step += 1

        del self.ReplayBuffer[:]
