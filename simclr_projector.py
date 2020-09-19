import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model import cf_size


class SimCLRProjector(nn.Module):

    def __init__(self, out_dim=256):
        super(SimCLRProjector, self).__init__()

        num_ftrs = 256

        # projection MLP
        self.l1 = nn.Linear(cf_size, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

    def forward(self, x):
        linenered_score = x.reshape((x.size(0), -1))  # 下接逆变换（）
        x = self.l1(linenered_score)
        x = F.relu(x)
        x = self.l2(x)
        return x
