import torch
from torch import nn

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
    
class CrossEntropyLoss(nn.Module):
    
    def log_softmax(self,x): 
        return x - x.exp().sum(-1).log().unsqueeze(-1)
    def nll(self,input, target): 
        return -input[range(target.shape[0]), target].mean()
    
    def forward(self, inputs, targets):
        """
        Computes the cross-entropy loss between the inputs and targets.
        - inputs: Tensor of shape (batch_size, num_classes).
        - targets: Tensor of shape (batch_size,) containing class indices.
        Returns the computed loss value.
        """
        log_probs = self.log_softmax(inputs)
        loss = self.nll(log_probs, targets)
        return loss

class Softmax(nn.Module):
    def forward(self, x):
        """
        Forward pass of the Softmax activation function.
        - x: Input tensor.
        Returns the output tensor after applying the Softmax function.
        """
        exp_x = torch.exp(x - x.max(dim=-1, keepdim=True).values)
        return exp_x / exp_x.sum(dim=-1, keepdim=True)