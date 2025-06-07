import torch
import torch.nn as nn

# Activation function: ReLU
# This class implements the ReLU activation function.
#         Performs a forward pass of the affine layer using the given input.
# It replaces negative values with zero and keeps positive values unchanged.
class Activation_ReLU(nn.Module):
    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs
        self.output = torch.maximum(torch.tensor(0,dtype=inputs.dtype,device=inputs.device), inputs)
        return self.output

# Activation function: Sigmoid
# This class implements the Sigmoid activation function.
class Sigmoid(nn.Module):
    def forward(self, x):
        """
        Forward pass of the Sigmoid activation function.
        - x: Input tensor.
        Returns the output tensor after applying the Sigmoid function.
        """
        return 1 / (1 + torch.exp(-x))
