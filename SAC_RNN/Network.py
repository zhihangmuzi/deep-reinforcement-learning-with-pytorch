import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorLSTM(nn.Module):
    def __init__(self, state_dim, action_dim, min_log_std=-10, max_log_std=2):
        super(ActorLSTM, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(state_dim+action_dim, 512)
        self.lstm1 = nn.LSTM(512, 512)
        self.fc3 = nn.Linear(2*512, 512)
        self.fc4 = nn.Linear(512, 512)

        self.mu_head = nn.Linear(512, action_dim)
        self.log_sigma_head = nn.Linear(512, action_dim)

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, state, last_action, hidden_in):
        """
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, action_dim)
        for lstm needs to be permuted as: (sequence_length, batch_size, -1)
        """
        state = state.permute(1, 0, 2)
        last_action = last_action.permute(1, 0, 2)

        # branch 1
        fc_branch = F.relu(self.fc1(state))
        # branch 2
        lstm_branch = torch.cat([state, last_action], -1)
        lstm_branch = F.relu(self.fc2(lstm_branch))
        lstm_branch, lstm_hidden = self.lstm1(lstm_branch, hidden_in)  # no activation after lstm
        # merged
        merged_branch = torch.cat([fc_branch, lstm_branch], -1)
        x = F.relu(self.fc3(merged_branch))
        x = F.relu(self.fc4(x))
        x = x.permute(1, 0, 2)  # permute back

        mu = self.mu_head(x)
        log_sigma = self.log_sigma_head(x)
        log_sigma = torch.clamp(log_sigma, self.min_log_std, self.max_log_std)
        return mu, log_sigma, lstm_hidden


class QLSTMNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QLSTMNet, self).__init__()
        self.fc1 = nn.Linear(state_dim+action_dim, 512)
        self.fc2 = nn.Linear(state_dim+action_dim, 512)
        self.lstm1 = nn.LSTM(512, 512)
        self.fc3 = nn.Linear(2 * 512, 512)
        self.fc4 = nn.Linear(512, action_dim)

    def forward(self, state, action, last_action, hidden_in):
        """
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, 1)
        for lstm needs to be permuted as: (sequence_length, batch_size, state_dim)
        """
        state = state.permute(1, 0, 2)
        action = action.permute(1, 0, 2)
        last_action = last_action.permute(1, 0, 2)

        # branch1
        fc_branch = torch.cat([state, action], -1)
        fc_branch = F.relu(self.fc1(fc_branch))

        # branch2
        lstm_branch = torch.cat([state, action], -1)
        lstm_branch = F.relu(self.fc2(lstm_branch))
        lstm_branch, lstm_hidden = self.lstm1(lstm_branch, hidden_in)

        # merge
        merge_branch = torch.cat([fc_branch, lstm_branch], -1)

        x = F.relu(self.fc3(merge_branch))
        x = self.fc4(x)
        qval = x.permute(1, 0, 2)
        return qval, lstm_hidden


class ActorGRU(nn.Module):
    def __init__(self, state_dim, action_dim, min_log_std=-10, max_log_std=2):
        super(ActorGRU, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(state_dim+action_dim, 512)
        self.lstm1 = nn.GRU(512, 512)
        self.fc3 = nn.Linear(2*512, 512)
        self.fc4 = nn.Linear(512, 512)

        self.mu_head = nn.Linear(512, action_dim)
        self.log_sigma_head = nn.Linear(512, action_dim)

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, state, last_action, hidden_in):
        """
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, action_dim)
        for lstm needs to be permuted as: (sequence_length, batch_size, -1)
        """
        state = state.permute(1, 0, 2)
        last_action = last_action.permute(1, 0, 2)

        # branch 1
        fc_branch = F.relu(self.fc1(state))
        # branch 2
        lstm_branch = torch.cat([state, last_action], -1)
        lstm_branch = F.relu(self.fc2(lstm_branch))
        lstm_branch, lstm_hidden = self.lstm1(lstm_branch, hidden_in)  # no activation after lstm
        # merged
        merged_branch = torch.cat([fc_branch, lstm_branch], -1)
        x = F.relu(self.fc3(merged_branch))
        x = F.relu(self.fc4(x))
        x = x.permute(1, 0, 2)  # permute back

        mu = self.mu_head(x)
        log_sigma = self.log_sigma_head(x)
        log_sigma = torch.clamp(log_sigma, self.min_log_std, self.max_log_std)
        return mu, log_sigma, lstm_hidden


class QGRUNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QGRUNet, self).__init__()
        self.fc1 = nn.Linear(state_dim+action_dim, 512)
        self.fc2 = nn.Linear(state_dim+action_dim, 512)
        self.lstm1 = nn.GRU(512, 512)
        self.fc3 = nn.Linear(2 * 512, 512)
        self.fc4 = nn.Linear(512, action_dim)

    def forward(self, state, action, last_action, hidden_in):
        """
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, 1)
        for lstm needs to be permuted as: (sequence_length, batch_size, state_dim)
        """
        state = state.permute(1, 0, 2)
        action = action.permute(1, 0, 2)
        last_action = last_action.permute(1, 0, 2)

        # branch1
        fc_branch = torch.cat([state, action], -1)
        fc_branch = F.relu(self.fc1(fc_branch))

        # branch2
        lstm_branch = torch.cat([state, action], -1)
        lstm_branch = F.relu(self.fc2(lstm_branch))
        lstm_branch, lstm_hidden = self.lstm1(lstm_branch, hidden_in)

        # merge
        merge_branch = torch.cat([fc_branch, lstm_branch], -1)

        x = F.relu(self.fc3(merge_branch))
        x = self.fc4(x)
        qval = x.permute(1, 0, 2)
        return qval, lstm_hidden

