import torch
import torch.nn as nn
class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features)
        elif num_dims == 4:
            shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape, dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(shape, dtype=torch.float32))
        self.moving_mean = torch.zeros(shape, dtype=torch.float32)
        self.moving_var = torch.ones(shape, dtype=torch.float32)
        
    def forward(self, X):
        """
        Applies batch normalization to the input tensor X.
        - X: Input tensor of shape (batch_size, num_features) or (batch_size, num_features, height, width).
        Returns the normalized tensor and updated moving mean and variance.
        """
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, 
            self.moving_mean, self.moving_var, 
            eps=1e-5, momentum=0.1
        )
        return Y






def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    if not torch.is_grad_enabled():
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4), "Input tensor must be 2D or 4D"
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var = (X - mean).pow(2).mean(dim=0)
        else:
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = (X - mean).pow(2).mean(dim=(0, 2, 3), keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + eps)
        moving_mean = (1 - momentum) * moving_mean + momentum * mean
        moving_var = (1 - momentum) * moving_var + momentum * var
    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data

   