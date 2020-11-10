import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.name = "Linear Regression"
        self._linear = nn.Linear(in_features=input_dim, out_features=output_dim, bias=True)

    def forward(self, x, weights=None):
        if weights is not None:
            return F.linear(x, weights[0], weights[1])
        else:
            return self._linear(x)