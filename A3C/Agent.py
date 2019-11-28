from A3C.Network import Actor_Value_Net


class A3C(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.actor_value = Actor_Value_Net(state_dim=self.state_dim, action_dim=self.action_dim)

    def choose_action(self, state):
        m, _ = self.actor_value.eval()(state)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        m, values = self.actor_value.train()(s)

        td = v_t - values
        c_loss = td.pow(2)

        exp_v = m.log_prob(a) * td.detach().squeeze()

        a_loss = -exp_v  # actor_loss
        total_loss = (c_loss + a_loss).mean()
        return total_loss
