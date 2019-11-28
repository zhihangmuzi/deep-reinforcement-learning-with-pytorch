import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from PER.prioritized_memory import Memory
from PER.Network import Net


class PERDQN(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 learning_rate=0.001,
                 reward_decay=0.9,
                 max_epsilon=0.99,
                 replace_target_iter=500,
                 memory_size=10000,
                 batch_size=32,
                 e_greedy_increment=None,
                 device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.device = device

        self.capacity = memory_size
        self.batch_size = batch_size
        self.ReplayBuffer = Memory(self.capacity)

        self.max_epsilon = max_epsilon  # 最大探索率
        self.epsilon_increment = e_greedy_increment  # 探索率增量
        self.epsilon = 0 if e_greedy_increment is not None else self.max_epsilon  # 是否选择电影增量

        self.learn_step_counter = 0

        self.eval_net = Net(self.state_dim, self.action_dim)
        self.target_net = Net(self.state_dim, self.action_dim)

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def update_target_model(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s, [a, r], s_))
        self.ReplayBuffer.store(transition)

    def choose_action(self, state):
        state = torch.unsqueeze(torch.Tensor(state).float(), 0)
        if np.random.uniform() < self.epsilon:
            actions_value = self.eval_net(state)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, self.action_dim)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            # print('\ntarget_params_replaced\n')

        tree_idx, batch_memory, ISWeights = self.ReplayBuffer.sample(self.batch_size)

        b_s = torch.Tensor(batch_memory[:, 0:self.state_dim]).float()
        b_a = torch.Tensor(batch_memory[:, self.state_dim:self.state_dim + 1]).long()
        b_r = torch.Tensor(batch_memory[:, self.state_dim + 1:self.state_dim + 2]).float()
        b_s_ = torch.Tensor(batch_memory[:, -self.state_dim:]).float()

        q_eval = self.eval_net(Variable(b_s)).gather(1, b_a)
        q_next = self.target_net(Variable(b_s_)).detach().max(1)[0]
        q_target = b_r + self.gamma * q_next.view(self.batch_size, 1)

        abs_errors = torch.abs(q_target - q_eval).detach().numpy()
        loss = torch.mean(torch.Tensor(ISWeights).float() * (q_eval - q_target).pow(2))
        self.ReplayBuffer.batch_update(tree_idx, abs_errors)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.max_epsilon else self.max_epsilon
        self.learn_step_counter += 1


