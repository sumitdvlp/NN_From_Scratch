
from data_setup import minst_dataset as dsets
import unittest
from all_scratch.arch import cnn_modules as cnn_modules
from utils import common_loss 
import torch
import torch.nn as nn
import torch.nn.functional as F
from arch import neuralnets as nnets
from utils import optimizers as optimizers
import matplotlib.pyplot as plt
import weightwatcher as ww
from torchviz import make_dot, make_dot_from_trace

class TestAllScratch(unittest.TestCase):
    def test_CNN(self):
        """
        Test class for CNN Regression model.
        """

        batch_size = 1
        n_iters = 30
        dataset = dsets.MNISTDataset()
        train_dataset, test_dataset = dataset.getTestTrainDatasets()
        
        print(f'Number of classes: {dataset.num_classes}')
        print(f'Number of samples: {dataset.num_samples}')
        
        print(f'Test output_shape: {dataset.output_shape}')
        print(f'Train input_shape: {dataset.input_shape}')

        input_dim = 28*28
        output_dim = 10
        num_epochs = n_iters / (len(train_dataset) / batch_size)
        num_epochs = int(num_epochs)


        print(f'Input shape {input_dim}')
        print(f'Output shape {output_dim}')
        print(f'Number of epochs {num_epochs}')

        train_loader, test_loader = dataset.getTestTrainLoaders(batch_size=batch_size)

        '''
        for i, (single_sample, y_train) in enumerate(train_loader):
            if i > 0:
                break
            # single_sample: [batch_size, 1, 28, 28]
            # y_train: [batch_size]

            # Reshape single_sample to 2D: [batch_size, 28*28]
            single_sample = single_sample.view(single_sample.shape[0], -1)

            # Print shape of single_sample
            print('='*50)
            print(f'Single sample shape: \t {list(single_sample.shape)}')
            print('='*50)


            # Print shape of single_sample
            print('='*50)
            print(f'Single sample shape: \t {list(y_train.shape)}')
            print('='*50)
        '''
        single_sample_batched, y_train = next(iter(train_loader))

        # single_sample = single_sample.squeeze(0)  # Remove batch dimension
        # y_train = y_train.squeeze(0)  # Remove batch dimension
        # single_sample = single_sample_batched.view( 28, 28)  # Reshape to 2D
        # y_train = y_train.view(1)  # Reshape to 1D
        for _, (images, _) in enumerate(train_loader): 
            print('='*50)
            print(f'Pytorch batched shape: \t {list(single_sample_batched.shape)}')
            print('='*50)

            print('*#'*50)
            num_kernels = 8
            kernel_shape = [5, 5]
            kernel_size=5
            stride=1
            padding=2
            print('*#'*50)

            # Forward: conv
            # Scratch Layer
            conv = cnn_modules.ConvolutionalLayer(in_channels=1, out_channels=6, kernel_size=5, stride=1)
            output = conv.forward(images)
            print('=*'*50)
            print(f'Conv (f) shape: \t {list(output.shape)}')
            print('=*'*50)
            # Pytorch Layer
            pycnn1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
            out = pycnn1(images)
            print('=*'*50)
            print(f'Pytorch Conv (f) shape: \t {list(out.shape)}')
            print('=*'*50)
            assert (list(output.shape) == list(out.shape)), "Output mismatch between custom and PyTorch Conv implementation"

            # Forward: ReLU
            relu1 = common_loss.Activation_ReLU()
            output = relu1.forward(output)
            print('=*'*50)
            print(f'ReLu1 (f) shape: \t {list(output.shape)}')
            print('=*'*50)
            
            pyrelu1 = nn.ReLU()
            out = pyrelu1(out)
            print('=*'*50)
            print(f'Pytorch ReLu1 (f) shape: \t {list(out.shape)}')
            print('=*'*50)
            assert (list(output.shape) == list(out.shape)), "Output mismatch between custom and PyTorch ReLU implementation"

            # Forward: Maxpool
            pool = cnn_modules.MaxPoolLayer(kernel_size=2)
            output = pool.forward(output)
            print('=*'*50)
            print(f'Maxpool output shape: \t {list(output.shape)}')
            print('=*'*50)
            pytorchmaxpool1 = nn.MaxPool2d(kernel_size=2)
            out = pytorchmaxpool1(out)
            print('=*'*50)
            print(f'Pytorch MaxPool (f) shape: \t {list(out.shape)}')
            print('=*'*50)
            assert (list(output.shape) == list(out.shape)), "Output mismatch between custom and PyTorch MaxPool implementation"


            # Forward: Affine and Softmax
            affinesoftmax = cnn_modules.AffineAndSoftmaxLayer(affine_weight_shape=(output.shape[1], output.shape[2], output.shape[3], len(torch.unique(y_train))))
            output = affinesoftmax.forward(output)

            print('='*50)
            print(f'Affine & Soft(arg)max (f) shape: \t {list(output.shape)}')
            print('='*50)

            # Affine and Softmax
            self.affine_softmax = nn.Linear(in_features=out.shape[1] * out.shape[2] * out.shape[3], out_features=len(torch.unique(y_train)))
            out = out.view(out.shape[0], -1)
            print('='*50)
            print(f'Pytorch Affine (f) shape: \t {list(out.shape)}')
            print('='*50)
            out = self.affine_softmax(out)
            print('='*50)
            print(f'Pytorch Affine & Soft(arg)max (f) shape: \t {list(out.shape)}')
            print('='*50)
            assert list(output.shape)== list(out.shape) , "Output mismatch between custom and PyTorch Affine & Softmax implementation"

    def test_CNN_Modules(self):
        train_loader, test_loader = setup_data()
        """
        Test class for CNN Modules.
        """
        model = nnets.ConvNN()
        
        loss = common_loss.CrossEntropyLoss()
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f'Parameter: {name}, Shape: {param.shape}')
        print('='*50)
        print(f'Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
        print('='*50)

        optimizer = optimizers.CustomAdam(model.parameters(), stepsize=0.001, bias_m1=0.9, bias_m2=0.999, epsilon=10e-8, bias_correction=False)
        loss_values = []
        for iter, (images, labels) in enumerate(train_loader): 
                
                model.zero_grad()
                pred = model.forward(images)
                ls = loss.forward(pred, labels)
                print('='*50)
                print(f'Loss: \t {ls.item()}')
                loss_values.append(ls.item())
                print('='*50)
                # Backward pass
                optimizer.step()
                # model.zero_grad()
                if iter % 50 == 0:
                    break
        watcher = ww.WeightWatcher(model)
        details = watcher.analyze(plot=True)
        print('='*50)
        print(f'WeightWatcher details: \n{details}')

        # make_dot(model(images), params=dict(model.named_parameters()))

    def test_load(self):
         train_loader, test_loader = setup_data()
         for batch_idx, (inputs, labels) in enumerate(test_loader):
            # inputs is a tensor of input features for the current batch
            # labels is a tensor of corresponding labels for the current batch
            print(f"Batch: {batch_idx}, Input shape: {inputs.shape}, Label shape: {labels.shape}")

    def test_FFNNLogistic(self):
        """
        Test class for FFNN Regression model.
        """
 
        train_loader, test_loader = setup_data()
        model = nnets.FeedForwardNeuralNetworkModel(input_dim=784, hidden_dim=196, output_dim=10)
        
        loss = common_loss.CrossEntropyLoss()
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f'Parameter: {name}, Shape: {param.shape}')
                
        print('='*50)
        print(f'Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
        
        optimizer = optimizers.CustomAdam(model.parameters(), stepsize=0.001, bias_m1=0.9, bias_m2=0.999, epsilon=10e-8, bias_correction=False)
        
        loss_values = []
        
        for iter, (images, labels) in enumerate(train_loader): 
                
                model.zero_grad()
                pred = model.forward(images)
                ls = loss.forward(pred, labels)
                print('='*50)
                print(f'Loss: \t {ls.item()}')
                loss_values.append(ls.item())
                print('='*50)
                # Backward pass
                optimizer.step()
                # model.zero_grad()
                if iter % 500 == 0:
                    break

        make_dot(model(images), params=dict(model.named_parameters()))

seed_value=42
torch.manual_seed(seed_value)
## Setup data
def setup_data():
    batch_size = 10
    n_iters = 10
    dataset = dsets.MNISTDataset()
    train_dataset, test_dataset = dataset.getTestTrainDatasets()
    
    print(f'Number of classes: {dataset.num_classes}')
    print(f'Number of samples: {dataset.num_samples}')
    
    print(f'Test output_shape: {dataset.output_shape}')
    print(f'Train input_shape: {dataset.input_shape}')
    input_dim = 28*28
    output_dim = 10
    num_epochs = n_iters / (len(train_dataset) / batch_size)
    num_epochs = int(num_epochs)
    print(f'Input shape {input_dim}')
    print(f'Output shape {output_dim}')
    print(f'Number of epochs {num_epochs}')
    return dataset.getTestTrainLoaders(batch_size=batch_size)


if __name__ == '__main__':
    unittest.main()