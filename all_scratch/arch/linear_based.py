import torch
import torch.nn as nn
from utils import activations_fxn as activations
from utils import common_loss as loss

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

class FeedForwardNeuralNetworkModel(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim):
    super(FeedForwardNeuralNetworkModel, self).__init__()

    #Linear Function
    self.fc1 = nn.Linear(input_dim, hidden_dim)

    # Non Linearity
    self.sigmoid = activations.Sigmoid()

    # Again a Linear Function
    self.fc2 = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    # Linear function #LINEAR
    out = self.fc1(x)

    # Non-linearity
    out = self.sigmoid(out)

    # Linearity
    out = self.fc2(out)
    return out
