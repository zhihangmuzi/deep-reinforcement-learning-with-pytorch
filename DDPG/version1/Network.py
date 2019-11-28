import torch
import torch.nn as nn
import torch.nn.functional as F


class A_NET(nn.Module):
    def __init__(self):
        super(A_NET, self).__init__()
        self.fc1 = nn.Linear(3, 30)
        self.fc1.weight.data.normal_(0, 0.3)
        self.fc2 = nn.Linear(30, 1)
        self.fc2.weight.data.normal_(0, 0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        out = x * 2
        return out


class C_NET(nn.Module):
    def __init__(self):
        super(C_NET, self).__init__()
        self.fcs = nn.Linear(3, 30)
        self.fcs.weight.data.normal_(0, 0.1)
        self.fca = nn.Linear(1, 30)
        self.fca.weight.data.normal_(0, 0.1)
        self.fcq = nn.Linear(30, 1)
        self.fcq.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        xs = self.fcs(s)
        xa = self.fca(a)
        xnet = F.relu(xs + xa)
        out = self.fcq(xnet)
        return out
