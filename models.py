import torch.nn as nn
import torch.functional as F

class PrimaryModel(nn.Module):
    def __init__(self):
        super(PrimaryModel, self).__init__()

        self.name = "primary_model"
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 53 * 53, 32) # change this
        self.fc2 = nn.Linear(32, 9) # change this

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 53 * 53) # change this
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()

    def forward(self, x):
        return x
