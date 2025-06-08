import torch
import torch.nn as nn
from utils import activations_fxn as activations
from utils import common_loss as loss

class LogisticRegressionModel(nn.Module):
    def __init__(self, num_features,num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.weights = nn.Parameter(torch.zeros(num_features, num_classes, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(num_classes, dtype=torch.float32))

        print(f'Lecun initialization SD: {1/num_classes}')
        self.weights = nn.Parameter(torch.nn.init.normal_(self.weights, mean=0, std=1/num_classes))
        self.bias = nn.Parameter(torch.nn.init.normal_(self.bias, mean=0, std=1/num_classes))

    # input shape: (batch_size, num_features)
    # output shape: (batch_size, num_classes)
    def forward(self, inp):
        output = torch.zeros((inp.shape[0], self.weights.shape[1]), dtype=inp.dtype, device=inp.device)
        for i in range(inp.shape[0]):
            # Flatten input to 1D
            tmp = inp[i].reshape(1, -1)
            out = torch.add(torch.matmul(tmp, self.weights), self.bias)
            output[i] = out
        return output