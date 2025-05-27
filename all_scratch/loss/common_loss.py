import torch


# Activation function: ReLU
# This class implements the ReLU activation function.
#         Performs a forward pass of the affine layer using the given input.
# It replaces negative values with zero and keeps positive values unchanged.
class Activation_ReLU:
    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs
        self.output = torch.maximum(torch.tensor(0,dtype=inputs.dtype,device=inputs.device), inputs)
        return self.output