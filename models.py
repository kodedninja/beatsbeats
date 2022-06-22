import torch as th
from torch import nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(c, c, 3, padding='same'),
            nn.BatchNorm2d(c),
            nn.ReLU(c),
            nn.Conv2d(c, c, 3, padding='same'),
            nn.BatchNorm2d(c)
        )
  
    def forward(self, x):
        return F.relu(x + self.block(x))
    
class Resi(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.embed = nn.Sequential(
            nn.Conv2d(input_dim, 16, (3, 7), padding=(0, 3)), # 16xLx78
            nn.ReLU(),
            ResBlock(16), # 16xLx78
            nn.MaxPool2d((3, 1)), # 16xLx26
            nn.Dropout2d(0.4),
            nn.Conv2d(16, 32, (3, 3), padding=(0, 1)), # 32xLx24
            nn.ReLU(),
            ResBlock(32), # 32xLx24
            nn.MaxPool2d((3, 1)), # 32xLx8
            nn.Dropout2d(0.4),
            nn.Conv2d(32, 64, (3, 3), padding=(0, 1)), # 64xLx6
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, None)), # 64xLx1
            nn.Dropout2d(0.4),
            nn.Conv2d(64, 1, 1) # 1xLx1
        )

    def forward(self, X):
        out = self.embed(X)
        out = th.flatten(out, start_dim=1)
        return out