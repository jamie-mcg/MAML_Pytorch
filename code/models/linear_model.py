import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()

        self._linear = nn.Linear()

    def forward(self, x, weights=None):
        if weights is not None:
            return F.linear(x, weights[0], weight[1])
        else:
            return self._linear(x)