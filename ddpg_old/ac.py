import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def weights_init_by_std(layer):
    input_d = layer.weight.data.size()[0]
    std = 1./np.sqrt(input_d)
    return (-std, std)

class Critic(nn.Module):
    def __init__(self, input_d, action_d, hidden_d1=128, hidden_d2=256):
        super(Critic, self).__init__()
        
        self.fc1s = nn.Linear(input_d, hidden_d1)
        self.bn1 = nn.BatchNorm1d(hidden_d1)

        self.fc1m = nn.Linear(hidden_d1 + action_d, hidden_d2)
        self.fc2m = nn.Linear(hidden_d2, 1)
        self.reset_params()

    def reset_params(self):
        self.fc1s.weight.data.uniform_(*weights_init_by_std(self.fc1s))
        self.fc1m.weight.data.uniform_(*weights_init_by_std(self.fc1m))
        self.fc2m.weight.data.uniform_(-0.003, 0.003)

    def forward(self, state, action):
        x = F.relu(self.bn1(self.fc1s(state)))
        merged = torch.cat((x, action), dim=1)
        merged = F.relu(self.fc1m(merged))
        out = self.fc2m(merged)
        return out


class Actor(nn.Module):
    def __init__(self, input_d, action_d, hidden_d1=128, hidden_d2=256):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(input_d, hidden_d1)
        self.bn1 = nn.BatchNorm1d(hidden_d1)
        self.fc2 = nn.Linear(hidden_d1, hidden_d2)
        self.fc3 = nn.Linear(hidden_d2, action_d)
        self.tanh = nn.Tanh()
        self.reset_params()

    def reset_params(self):
        self.fc1.weight.data.uniform_(*weights_init_by_std(self.fc1))
        self.fc2.weight.data.uniform_(*weights_init_by_std(self.fc2))
        self.fc3.weight.data.uniform_(*weights_init_by_std(self.fc3))

    def forward(self, state):
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.fc2(x))
        out = self.tanh(self.fc3(x))
        return out
