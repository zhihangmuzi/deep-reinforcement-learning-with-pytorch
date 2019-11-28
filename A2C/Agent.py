import torch
from torch.autograd import Variable

from A2C.Network import Actor_NET, Value_NET
from A2C.utils import entropy, discount_reward


class A2C(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 lr_a=0.01,
                 lr_v=0.01,
                 reward_decay=0.99,
                 device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr_a = lr_a
        self.lr_v = lr_v
        self.gamma = reward_decay
        self.device = device

        self.actor = Actor_NET(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)

        self.value = Value_NET(state_dim=self.state_dim).to(self.device)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.lr_v)

    def learn(self, states, actions, rewards, final_r):
        actions_var = Variable(torch.Tensor(actions).view(-1, self.action_dim)).to(self.device)
        states_var = Variable(torch.Tensor(states).view(-1, self.state_dim)).to(self.device)

        entropy_loss = torch.sum(entropy(self.actor(states_var)))

        self.actor_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        log_probs = torch.log(self.actor(states_var))
        vs = self.value(states_var).detach()
        qs = Variable(torch.Tensor(discount_reward(rewards, 0.99, final_r))).view(-1, 1).to(self.device)
        advantages = qs - vs
        actor_network_loss = -torch.mean(torch.sum(log_probs * actions_var, 1) * advantages)

        target_values = qs.view(-1, 1)
        values = self.value(states_var)
        value_network_loss = (target_values - values).pow(2).mean()

        loss = actor_network_loss + value_network_loss - 0.0001 * entropy_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
        self.actor_optimizer.step()
        self.value_optimizer.step()
