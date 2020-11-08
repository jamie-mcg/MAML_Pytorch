import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()

    def forward(self, x, weights):
        x = F.linear(x, weights[0], weight[1])
        return x