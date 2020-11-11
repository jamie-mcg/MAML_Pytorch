import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.name = "MLP"
        self._fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True)
        self._fc2 = nn.Linear(in_features=hidden_dim, out_features=output_dim, bias=True)

    def forward(self, x, weights=None):
        if weights is not None:
            x = F.linear(x, weights[0], weights[1])
            x = F.linear(x, weights[2], weights[3])
            return x
        else:
            x = self._fc1(x)
            x = self._fc2(x)
            return x