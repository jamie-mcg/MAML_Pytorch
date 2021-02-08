import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Model for a Multi-layer Perceptron (MLP) defined using the PyTorch backbone.

    Methods:
    - forward():
        Redefinition of the nn.Module forward() method to include the option to provide 
        manual weights not necessarily the same as the weights saved inside this model.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        # Assign a name for this model.
        self.name = "MLP"

        # Define the layers used in this model.
        self._fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True)
        self._fc2 = nn.Linear(in_features=hidden_dim, out_features=output_dim, bias=True)

    def forward(self, x, weights=None):
        """
        This method takes in some input x and passes this through the models layers.

        Inputs:
        - x: Input tensor for the model to make predictions for.
        - weights: (Optional) Tensor containing the weights which can be used to override the 
          inherent weights contained in this model.

        Ouput:
        - x: The predictions produced from the model from passing the initial x through its layers.
        """
        # If we have provided weights then override the implicitly saved weights.
        if weights is not None:
            x = F.linear(x, weights[0], weights[1])
            x = F.linear(x, weights[2], weights[3])
            return x
        else:
            x = self._fc1(x)
            x = self._fc2(x)
            return x