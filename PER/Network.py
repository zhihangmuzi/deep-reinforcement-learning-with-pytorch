import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(s_dim, 24)
        self.fc1.weight.data.normal_(0, 0.3)
        self.fc2 = nn.Linear(24, 24)
        self.fc2.weight.data.normal_(0, 0.3)
        self.fc3 = nn.Linear(24, a_dim)
        self.fc3.weight.data.normal_(0, 0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out
