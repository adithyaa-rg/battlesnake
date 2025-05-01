import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 15, kernel_size=3, dtype = torch.float32)
        self.pool1 = nn.MaxPool2d(2, 2)


        self.layer1 = nn.Linear(240, 128)
        self.layer2 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = torch.flatten(x, start_dim=0)
        x = F.relu(self.layer1(x))
        return self.layer2(x)
    
