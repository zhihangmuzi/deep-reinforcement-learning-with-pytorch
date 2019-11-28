import torch.nn as nn
import torch.nn.functional as F


class Actor_NET(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor_NET, self).__init__()
        self.fc1 = nn.Linear(state_dim, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, action_dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.softmax(self.fc3(out), dim=1)
        return out


class Value_NET(nn.Module):
    def __init__(self, state_dim):
        super(Value_NET, self).__init__()
        self.fc1 = nn.Linear(state_dim, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
