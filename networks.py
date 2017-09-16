import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class XUANet(nn.Module):
    def __init__(self):
        super(XUANet, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size = 1, stride=1)
        #self.conv2 = nn.Conv2d(2, 2, kernel_size = 1, stride=1)
        #self.conv3 = nn.Conv2d(2, 1, kernel_size = 1, stride=1)
        self.fc1 = nn.Linear(1877, 4096)
        self.fc2 = nn.Linear(4096, 2)
        #self.fc3 = nn.Linear(1024, 256)
        #self.fc4 = nn.Linear(256, 2)
        self.ceriation = nn.CrossEntropyLoss()

    def forward(self, x, target):
        x = F.relu(self.conv1(x))
        #x = F.relu(self.conv2(x))
        #x = F.relu(self.conv3(x))
        x = torch.squeeze(x, 1)
        x = torch.squeeze(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #x = F.sigmoid(self.fc4(x))
        loss = self.ceriation(x, target)
        return x, loss

    def name(self):
        return 'XUANet'






