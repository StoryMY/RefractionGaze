import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.2)
        m.bias.data.fill_(0.0)

class MLPGazeNet(nn.Module):
    def __init__(self):
        super(MLPGazeNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        self.fc = nn.Linear(8, 2)
        self.apply(init_weights)

    def forward(self, cpvalue, head):
        feature = self.mlp(cpvalue)
        out = self.fc(feature) + head

        return out

