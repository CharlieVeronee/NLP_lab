import torch
import torch.nn as nn
import torch.nn.functional as F
from model_data_loader import load_model_data

class SWEM(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(300, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x