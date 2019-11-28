import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Actor_Value_Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor_Value_Net, self).__init__()
        self.fca1 = nn.Linear(state_dim, 200)
        self.fca2 = nn.Linear(200, action_dim)
        self.fcv1 = nn.Linear(state_dim, 100)
        self.fcv2 = nn.Linear(100, 1)

    def forward(self, x):
        out1 = F.relu6(self.fca1(x))
        logits = self.fca2(out1)
        probs = F.softmax(logits, dim=1)
        m = Categorical(probs)

        out2 = F.relu6(self.fcv1(x))
        values = self.fcv2(out2)
        return m, values
