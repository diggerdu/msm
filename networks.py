import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

class XUANet(nn.Module):
    def __init__(self):
        super(XUANet, self).__init__()
        self.conv1 = nn.Conv2d(3, 2, kernel_size = 1, stride=1)
        init.xavier_uniform(self.conv1.weight, gain=nn.init.calculate_gain('relu'))
        self.conv2 = nn.Conv2d(2, 1, kernel_size = 1, stride=1)
        init.xavier_uniform(self.conv2.weight, gain=nn.init.calculate_gain('relu'))
        self.fc1 = nn.Linear(1877, 4096)
        self.fc2 = nn.Linear(4096, 64)
        self.fc3 = nn.Linear(64, 1)
        self.ceriation = nn.MSELoss()

    def forward(self, x, target):
        x = self.conv1(x)
        x = F.leaky_relu(self.conv2(x))
        x = torch.squeeze(x, 1)
        x = torch.squeeze(x, 1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x)) / 10.
        x = F.sigmoid(self.fc3(x))
        #x = F.sigmoid(self.fc3(x))
        x = torch.squeeze(x, 1)
        loss = self.ceriation(x, target)
        return x, loss

    def name(self):
        return 'XUANet'






