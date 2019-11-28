import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from DuelingDQN.Network import QNet
from DuelingDQN.ReplayBuffer import ReplayBuffer


class DuelDQN(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 max_epsilon=0.9,
                 replace_target_iter=100,
                 memory_size=2000,
                 batch_size=32,
                 e_greedy_increment=None,
                 device='cpu'):
        self.state_dim = state_dim     # 状态维度
        self.action_dim = action_dim   # 动作维度
        self.lr = learning_rate        # 学习率
        self.gamma = reward_decay      # 奖励折扣
        self.device = device
        self.replace_target_iter = replace_target_iter

        self.capacity = memory_size     # 经验池大小
        self.batch_size = batch_size    # batch_size
        self.ReplayBuffer = ReplayBuffer(self.capacity)

        self.max_epsilon = max_epsilon  # 最大探索率
        self.epsilon_increment = e_greedy_increment  # 探索率增量
        self.epsilon = 0 if e_greedy_increment is not None else self.max_epsilon  # 是否选择电影增量

        self.eval_net = QNet(state_dim, action_dim).to(self.device)    # eval_net
        self.target_net = QNet(state_dim, action_dim).to(self.device)  # target_net

        self.learn_step_counter = 0  # 学习计数

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def store_transition(self, s, a, r, s_, done):
        transition = (s, a, r, s_, done)
        self.ReplayBuffer.add(transition)

    def choose_action(self, state):
        state = torch.Tensor(state).float().unsqueeze(0).to(self.device)
        if np.random.uniform() < self.epsilon:
            actions_value = self.eval_net(state)
            action = torch.max(actions_value.cpu(), 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, self.action_dim)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('\ntarget_params_replaced\n')
        self.learn_step_counter += 1

        batch_memory = self.ReplayBuffer.sample(self.batch_size)

        b_s = torch.Tensor(np.asarray([e[0] for e in batch_memory])).float().to(device=self.device)
        b_a = torch.Tensor(np.asarray([e[1] for e in batch_memory])).long().reshape((-1, 1)).to(device=self.device)
        b_r = torch.Tensor(np.asarray([e[2] for e in batch_memory])).float().reshape((-1, 1)).to(device=self.device)
        b_s_ = torch.Tensor(np.asarray([e[3] for e in batch_memory])).float().to(device=self.device)

        q_eval = self.eval_net(Variable(b_s)).gather(1, b_a)
        q_next = self.target_net(Variable(b_s_)).detach().max(1)[0]
        q_target = b_r + self.gamma * q_next.view(self.batch_size, 1)

        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.max_epsilon else self.max_epsilon











