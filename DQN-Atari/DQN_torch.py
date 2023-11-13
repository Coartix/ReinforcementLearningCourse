import torch
import torch.nn as nn
import numpy as np

# Define the Q-network model using PyTorch
class DQN(nn.Module):
    def __init__(self, state_shape, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self._conv2d_output(state_shape), 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _conv2d_output(self, shape):
        x = torch.zeros(1, *shape)
        x = self._forward_conv(x)
        return int(np.prod(x.size()))

    def _forward_conv(self, x):
        x = nn.functional.relu(self.conv1(x), inplace=True)
        x = nn.functional.relu(self.conv2(x), inplace=True)
        x = nn.functional.relu(self.conv3(x), inplace=True)
        return x.view(x.size(0), -1)

    def forward(self, x):
        x = self._forward_conv(x)
        x = nn.functional.relu(self.fc1(x), inplace=True)
        x = self.fc2(x)
        return x