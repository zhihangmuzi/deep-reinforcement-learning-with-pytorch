import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal

from SAC_RNN.Network import ActorLSTM, QLSTMNet, ActorGRU, QGRUNet
from SAC_RNN.ReplayBuffer import ReplayBufferLSTM2, ReplayBufferGRU
from SAC_RNN.utils import hard_update, soft_update


class SACRNN(object):
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
                 batch_size=2,
                 mode='lstm',
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
        self.mode = mode
        self.device = device

        if self.mode == "lstm":
            self.actor = ActorLSTM(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
            self.soft_q1_net = QLSTMNet(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
            self.soft_q2_net = QLSTMNet(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
            self.target_soft_q1_net = QLSTMNet(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
            self.target_soft_q2_net = QLSTMNet(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
        else:
            self.actor = ActorGRU(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
            self.soft_q1_net = QGRUNet(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
            self.soft_q2_net = QGRUNet(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
            self.target_soft_q1_net = QGRUNet(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
            self.target_soft_q2_net = QGRUNet(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)

        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.soft_q1_optimizer = torch.optim.Adam(self.soft_q1_net.parameters(), lr=self.lr_q)
        self.soft_q2_optimizer = torch.optim.Adam(self.soft_q2_net.parameters(), lr=self.lr_q)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr_alpha)

        if self.mode == 'lstm':
            self.ReplayBuffer = ReplayBufferLSTM2(self.capacity)
        else:
            self.ReplayBuffer = ReplayBufferGRU(self.capacity)
        self.num_training = 1

        hard_update(self.target_soft_q1_net, self.soft_q1_net)
        hard_update(self.target_soft_q2_net, self.soft_q2_net)

        self.target_entropy = -2
        self.alpha = 1.

    def choose_action(self, state, last_action, hidden_in):
        state = torch.Tensor(state).float().unsqueeze(0).unsqueeze(0).to(self.device)
        last_action = torch.Tensor(last_action).unsqueeze(0).unsqueeze(0).to(self.device)
        mu, log_std, hidden_out = self.actor(state, last_action, hidden_in)

        std = torch.exp(log_std)
        m = Normal(mu, std)
        action_val = m.sample()
        action = torch.tanh(action_val).detach().cpu().numpy()
        return action[0][0], hidden_out

    def store(self, h_i, h_o, s, a, la, r, s_, d):
        transition = (h_i, h_o, s, a, la, r, s_, np.float(d))
        self.ReplayBuffer.add(transition)

    def evaluate(self, state, last_action, hidden_in, epsilon=1e-6):
        mu, log_sigma, hidden_out = self.actor(state, last_action, hidden_in)
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        noise = Normal(0, 1)

        z = noise.sample()
        action = torch.tanh(mu + sigma * z.to(self.device))
        log_prob = dist.log_prob(mu + sigma * z.to(self.device)) - torch.log(1 - action.pow(2) + 1e-5)
        return action, log_prob, z, mu, log_sigma, hidden_out

    def update(self,):
        if self.num_training % 500 == 0:
            print("Training ... {} times ".format(self.num_training))

        hidden_in, hidden_out, state, action, last_action, reward, next_state, done = self.ReplayBuffer.sample(
            self.batch_size)

        state = torch.Tensor(state).float().to(self.device)
        next_state = torch.Tensor(next_state).float().to(self.device)
        action = torch.Tensor(action).float().to(self.device)
        last_action = torch.Tensor(last_action).float().to(self.device)
        reward = torch.Tensor(reward).float().unsqueeze(-1).to(self.device)
        done = torch.Tensor(np.float32(done)).float().unsqueeze(-1).to(self.device)

        predicted_q_value1, _ = self.soft_q1_net(state, action, last_action, hidden_in)
        predicted_q_value2, _ = self.soft_q2_net(state, action, last_action, hidden_in)
        new_action, log_prob, z, mean, log_std, _ = self.evaluate(state, last_action, hidden_in)
        new_next_action, next_log_prob, _, _, _, _ = self.evaluate(next_state, action, hidden_out)
        reward = 10 * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6)

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        predict_target_q1, _ = self.target_soft_q1_net(next_state, new_next_action, action, hidden_out)
        predict_target_q2, _ = self.target_soft_q2_net(next_state, new_next_action, action, hidden_out)
        target_q_min = torch.min(predict_target_q1, predict_target_q2) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * self.gamma * target_q_min  # if done==1, only reward
        q_value_loss1 = F.mse_loss(predicted_q_value1, target_q_value.detach())
        q_value_loss2 = F.mse_loss(predicted_q_value2, target_q_value.detach())

        self.soft_q1_optimizer.zero_grad()
        q_value_loss1.backward()
        self.soft_q1_optimizer.step()
        self.soft_q2_optimizer.zero_grad()
        q_value_loss2.backward()
        self.soft_q2_optimizer.step()

        predict_q1, _ = self.soft_q1_net(state, new_action, last_action, hidden_in)
        predict_q2, _ = self.soft_q2_net(state, new_action, last_action, hidden_in)
        predicted_new_q_value = torch.min(predict_q1, predict_q2)
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        soft_update(self.target_soft_q1_net, self.soft_q1_net, self.tau)
        soft_update(self.target_soft_q2_net, self.soft_q2_net, self.tau)

        self.num_training += 1
