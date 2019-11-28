import torch.nn as nn
import torch.nn.functional as F


class A_Net(nn.Module):
    def __init__(self):
        super(A_Net, self).__init__()
        self.fc1 = nn.Linear(4, 20)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(20, 2)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        out = F.softmax(x, dim=1)
        return out


class C_Net(nn.Module):
    def __init__(self):
        super(C_Net, self).__init__()
        self.fc1 = nn.Linear(4, 20)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(20, 1)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out

