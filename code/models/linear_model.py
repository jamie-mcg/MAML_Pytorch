import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearRegression(nn.Module):
    """
    Model for Linear Regression defined using the PyTorch backbone.

    Methods:
    - forward():
        Redefinition of the nn.Module forward() method to include the option to provide 
        manual weights not necessarily the same as the weights saved inside this model.
    """
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()

        # Assign a name for this model.
        self.name = "Linear Regression"

        # Define the layers used in this model.
        self._linear = nn.Linear(in_features=input_dim, out_features=output_dim, bias=True)

    def forward(self, x, weights=None):
        """
        This method takes in some input x and passes this through the models layers.

        Inputs:
        - x: Input tensor for the model to make predictions for.
        - weights: (Optional) Tensor containing the weights which can be used to override the 
          inherent weights contained in this model.

        Ouput:
        - y: The predictions produced from the model from passing x through its layers.
        """
        # If we have provided weights then override the implicitly saved weights.
        if weights is not None:
            return F.linear(x, weights[0], weights[1])
        else:
            return self._linear(x)