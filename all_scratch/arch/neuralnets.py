import torch
import torch.nn as nn
from arch import cnn_modules,linear_based
from utils import common_loss as loss
from utils import activations_fxn as activations

class ConvNNPytorch(nn.Module):
    def __init__(self):
        super(ConvNNPytorch,self).__init__()
        '''
        Initializes the CNN model using PyTorch's nn.Module.
        '''
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.affine_softmax = nn.Linear(in_features=6 * 12 * 12, out_features=1)
    def forward(self, x):
        '''
        Forward pass of the CNN model.
        - x: Input tensor of shape (batch_size, channels, height, width).
        Returns the output tensor after passing through the CNN layers.
        '''
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)  # Flatten the tensor
        x = self.affine_softmax(x)
        return x

class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN,self).__init__()
        '''
        Initializes the CNN model.
        '''
        self.conv1 = cnn_modules.ConvolutionalLayer(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.relu1 = loss.Activation_ReLU()
        self.maxpool1 = cnn_modules.MaxPoolLayer(kernel_size=2)
        self.affine_softmax = cnn_modules.AffineAndSoftmaxLayer(affine_weight_shape=([6 , 12, 12,10]))
    def forward(self, x):
        '''
        Forward pass of the CNN model.
        - x: Input tensor of shape (batch_size, channels, height, width).
        Returns the output tensor after passing through the CNN layers.
        '''
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)    
        x = self.affine_softmax(x)
        return x
    def __repr__(self):
        '''
        Returns a string representation of the CNN model.
        '''
        return f"CNN(in_channels={self.conv1.in_channels}, out_channels={self.conv1.out_channels}, kernel_size={self.conv1.kernel_size})"
    def __str__(self):
        '''
        Returns a string representation of the CNN model.
        '''
        return f"CNN(in_channels={self.conv1.in_channels}, out_channels={self.conv1.out_channels}, kernel_size={self.conv1.kernel_size})"

class FeedForwardNeuralNetworkModel(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim):
    super(FeedForwardNeuralNetworkModel, self).__init__()

    #Linear Function
    self.fc1 = linear_based.LogisticRegressionModel(input_dim, hidden_dim)

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
  
  def __repr__(self):
    return f"FeedForwardNeuralNetworkModel(input_dim={self.fc1.in_features}, hidden_dim={self.fc1.out_features}, output_dim={self.fc2.out_features})"
  def __str__(self):
    return f"FeedForwardNeuralNetworkModel(input_dim={self.fc1.in_features}, hidden_dim={self.fc1.out_features}, output_dim={self.fc2.out_features})"

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()