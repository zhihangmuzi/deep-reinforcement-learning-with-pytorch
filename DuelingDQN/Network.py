import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 20)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fcv = nn.Linear(20, 1)
        self.fcv.weight.data.normal_(0, 0.1)
        self.fca = nn.Linear(20, action_dim)
        self.fca.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.fcv(x)
        advan = self.fca(x)
        out = value + (advan - torch.mean(advan))
        return out
